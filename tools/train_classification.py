import argparse
from pathlib import Path

import torch
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from egcvmamba.data import build_classification_loaders
from egcvmamba.engine import evaluate_classification, make_scaler, train_classification_epoch
from egcvmamba.models import build_model
from egcvmamba.utils import (
    EMA,
    append_jsonl,
    cleanup_distributed,
    init_distributed_mode,
    is_main_process,
    load_config,
    parameter_groups,
    save_checkpoint,
    set_seed,
    unwrap_model,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser("EGCVMamba ImageNet training")
    parser.add_argument("--config", default="configs/classification/egcvmamba_tiny.yaml")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", default=None, help="Checkpoint path; omitted uses auto-resume when enabled")
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def build_mixup(cfg):
    mixup_alpha = float(cfg.get("mixup", 0.0))
    cutmix_alpha = float(cfg.get("cutmix", 0.0))
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return None, None
    mixup = Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        prob=float(cfg.get("mixup_prob", 1.0)),
        switch_prob=float(cfg.get("mixup_switch_prob", 0.5)),
        mode="batch",
        label_smoothing=float(cfg.get("label_smoothing", 0.1)),
        num_classes=int(cfg["num_classes"]),
    )
    return mixup, SoftTargetCrossEntropy()


def main():
    args = parse_args()
    distributed = init_distributed_mode()
    cfg = load_config(args.config)
    if args.data_path is not None:
        cfg["data"]["path"] = args.data_path
    if args.output is not None:
        cfg["output"] = args.output

    rank = distributed["rank"]
    world_size = distributed["world_size"]
    device = torch.device(
        f"cuda:{distributed['local_rank']}" if torch.cuda.is_available() else "cpu"
    )
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    torch.backends.cudnn.benchmark = bool(cfg["train"].get("cudnn_benchmark", True))

    channels_last = bool(cfg["train"].get("channels_last", True))
    precision = cfg["train"].get("precision", "bf16")
    model = build_model(
        cfg["model"]["variant"],
        cfg["model"]["num_classes"],
        cfg["model"]["drop_path_rate"],
    ).to(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)

    train_loader, val_loader = build_classification_loaders(
        cfg["data"]["path"],
        cfg["data"]["image_size"],
        cfg["train"]["batch_size"],
        cfg["train"]["workers"],
        distributed=distributed["distributed"],
        rank=rank,
        world_size=world_size,
        augmentation=cfg.get("augmentation", {}),
    )

    global_batch_size = int(cfg["train"]["batch_size"]) * world_size
    reference_batch_size = int(cfg["train"].get("reference_batch_size", global_batch_size))
    lr_scale = global_batch_size / reference_batch_size
    learning_rate = float(cfg["train"]["lr"]) * lr_scale
    min_lr = float(cfg["train"].get("min_lr", 1e-6)) * lr_scale
    betas = tuple(cfg["train"].get("betas", [0.9, 0.999]))
    weight_decay = float(cfg["train"]["weight_decay"])
    optimizer = AdamW(
        parameter_groups(model, weight_decay),
        lr=learning_rate,
        betas=betas,
        eps=float(cfg["train"].get("eps", 1e-8)),
        weight_decay=0.0,
    )
    warmup_epochs = int(cfg["train"]["warmup_epochs"])
    total_epochs = int(cfg["train"]["epochs"])
    warmup = LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(total_epochs - warmup_epochs, 1),
        eta_min=min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )
    scaler = make_scaler(device, precision)

    output = Path(cfg["output"])
    resume_path = args.resume
    if resume_path is None and cfg["train"].get("auto_resume", True):
        candidate = output / "last.pth"
        if candidate.is_file():
            resume_path = str(candidate)

    start_epoch = 0
    best_acc = 0.0
    checkpoint = None
    if resume_path:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint.get("model", checkpoint))
        if not args.eval_only:
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
            if "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_acc = float(checkpoint.get("best_acc", 0.0))

    if distributed["distributed"]:
        model = DistributedDataParallel(
            model,
            device_ids=[distributed["local_rank"]],
            output_device=distributed["local_rank"],
            gradient_as_bucket_view=True,
            static_graph=True,
        )

    ema = EMA(model, cfg["train"].get("ema_decay", 0.9999)) if cfg["train"].get("ema", True) else None
    if ema is not None and checkpoint is not None and "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])

    # Give each rank an independent augmentation RNG after identical model initialization.
    set_seed(seed + rank)
    mixup_cfg = dict(cfg.get("augmentation", {}))
    mixup_cfg["label_smoothing"] = cfg["train"].get("label_smoothing", 0.1)
    mixup_cfg["num_classes"] = cfg["model"]["num_classes"]
    mixup_fn, train_criterion = build_mixup(mixup_cfg)
    show_progress = is_main_process()

    if is_main_process():
        print(
            {
                "variant": cfg["model"]["variant"],
                "world_size": world_size,
                "batch_per_gpu": cfg["train"]["batch_size"],
                "global_batch_size": global_batch_size,
                "lr": learning_rate,
                "precision": precision,
                "resume": resume_path,
            }
        )

    if args.eval_only:
        if checkpoint is None:
            raise ValueError("--eval-only requires --resume or an auto-resume checkpoint.")
        evaluation_model = ema.module if ema is not None and "ema" in checkpoint else unwrap_model(model)
        stats = evaluate_classification(
            evaluation_model,
            val_loader,
            device,
            precision=precision,
            channels_last=channels_last,
            show_progress=show_progress,
        )
        if is_main_process():
            print(stats)
        cleanup_distributed()
        return

    for epoch in range(start_epoch, total_epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        train_stats = train_classification_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            smoothing=cfg["train"]["label_smoothing"],
            clip_grad=cfg["train"]["clip_grad"],
            ema=ema,
            mixup_fn=mixup_fn,
            criterion=train_criterion,
            precision=precision,
            channels_last=channels_last,
            show_progress=show_progress,
        )
        val_stats = evaluate_classification(
            model,
            val_loader,
            device,
            precision=precision,
            channels_last=channels_last,
            show_progress=show_progress,
        )
        ema_stats = None
        if ema is not None:
            ema_stats = evaluate_classification(
                ema.module,
                val_loader,
                device,
                precision=precision,
                channels_last=channels_last,
                show_progress=show_progress,
            )
        scheduler.step()

        selection_acc = ema_stats["acc1"] if ema_stats is not None else val_stats["acc1"]
        is_best = selection_acc >= best_acc
        best_acc = max(best_acc, selection_acc)
        state = {
            "epoch": epoch,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_acc": best_acc,
            "config": cfg,
        }
        if ema is not None:
            state["ema"] = ema.state_dict()
        metrics = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_stats,
            "val": val_stats,
            "ema_val": ema_stats,
            "best_acc": best_acc,
        }
        if is_main_process():
            save_checkpoint(output / "last.pth", state)
            if is_best:
                save_checkpoint(output / "best.pth", state)
            append_jsonl(output / "metrics.jsonl", metrics)
            write_json(output / "metrics.json", metrics)
            print(metrics)

    cleanup_distributed()


if __name__ == "__main__":
    main()
