import argparse
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from egcvmamba.data import build_segmentation_loaders
from egcvmamba.engine import evaluate_segmentation, make_scaler, train_segmentation_epoch
from egcvmamba.models import build_segmentation_model
from egcvmamba.utils import (
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
    parser = argparse.ArgumentParser("EGCVMamba ADE20K training")
    parser.add_argument("--config", default="configs/segmentation/ade20k_tiny.yaml")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--pretrained", default=None, help="ImageNet classification checkpoint")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def load_pretrained_backbone(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "ema" in checkpoint:
        ema_state = checkpoint["ema"]
        source = ema_state.get("module", ema_state.get("shadow", {}))
    else:
        source = checkpoint.get("model", checkpoint)
    target = model.backbone.state_dict()
    matched = {}
    for name, value in target.items():
        source_value = source.get(name, source.get(f"backbone.{name}"))
        if source_value is not None and source_value.shape == value.shape:
            matched[name] = source_value
    model.backbone.load_state_dict(matched, strict=False)
    return len(matched), len(target)


def main():
    args = parse_args()
    distributed = init_distributed_mode()
    cfg = load_config(args.config)
    if args.data_path is not None:
        cfg["data"]["path"] = args.data_path
    if args.output is not None:
        cfg["output"] = args.output

    output = Path(cfg["output"])
    resume_path = args.resume
    if resume_path is None and cfg["train"].get("auto_resume", True):
        candidate = output / "last.pth"
        if candidate.is_file():
            resume_path = str(candidate)

    rank = distributed["rank"]
    world_size = distributed["world_size"]
    device = torch.device(
        f"cuda:{distributed['local_rank']}" if torch.cuda.is_available() else "cpu"
    )
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    torch.backends.cudnn.benchmark = bool(cfg["train"].get("cudnn_benchmark", True))
    precision = cfg["train"].get("precision", "bf16")
    channels_last = bool(cfg["train"].get("channels_last", True))

    model = build_segmentation_model(
        cfg["model"]["variant"],
        cfg["model"]["num_classes"],
        cfg["model"]["drop_path_rate"],
        cfg["model"]["decoder_channels"],
    ).to(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)

    pretrained_path = args.pretrained or cfg["model"].get("pretrained")
    if pretrained_path and not resume_path:
        loaded, total = load_pretrained_backbone(model, pretrained_path)
        if is_main_process():
            print(f"Loaded {loaded}/{total} backbone tensors from {pretrained_path}")
    elif not resume_path and is_main_process():
        print("Warning: training segmentation without ImageNet pretrained backbone; mIoU will be lower.")

    global_batch_size = int(cfg["train"]["batch_size"]) * world_size
    reference_batch_size = int(cfg["train"].get("reference_batch_size", global_batch_size))
    lr_scale = global_batch_size / reference_batch_size
    learning_rate = float(cfg["train"]["lr"]) * lr_scale
    backbone_lr = learning_rate * float(cfg["train"].get("backbone_lr_multiplier", 0.1))
    weight_decay = float(cfg["train"]["weight_decay"])
    optimizer = AdamW(
        parameter_groups(model.backbone, weight_decay, backbone_lr)
        + parameter_groups(model.decode_head, weight_decay, learning_rate),
        lr=learning_rate,
        betas=tuple(cfg["train"].get("betas", [0.9, 0.999])),
        weight_decay=0.0,
    )
    warmup_epochs = int(cfg["train"]["warmup_epochs"])
    total_epochs = int(cfg["train"]["epochs"])
    warmup = LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(total_epochs - warmup_epochs, 1),
        eta_min=float(cfg["train"].get("min_lr", 1e-6)) * lr_scale,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )
    scaler = make_scaler(device, precision)

    start_epoch = 0
    best_miou = 0.0
    checkpoint = None
    if resume_path:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint.get("model", checkpoint))
        if not args.eval_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            if "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_miou = float(checkpoint.get("best_miou", 0.0))

    if distributed["distributed"]:
        model = DistributedDataParallel(
            model,
            device_ids=[distributed["local_rank"]],
            output_device=distributed["local_rank"],
            gradient_as_bucket_view=True,
            static_graph=True,
        )

    set_seed(seed + rank)
    scale_range = tuple(cfg.get("augmentation", {}).get("scale_range", [0.5, 2.0]))
    train_loader, val_loader = build_segmentation_loaders(
        cfg["data"]["path"],
        cfg["data"]["image_size"],
        cfg["train"]["batch_size"],
        cfg["train"]["workers"],
        distributed=distributed["distributed"],
        rank=rank,
        world_size=world_size,
        scale_range=scale_range,
    )
    show_progress = is_main_process()
    if is_main_process():
        print(
            {
                "variant": cfg["model"]["variant"],
                "world_size": world_size,
                "global_batch_size": global_batch_size,
                "backbone_lr": backbone_lr,
                "decoder_lr": learning_rate,
                "precision": precision,
                "resume": resume_path,
            }
        )

    if args.eval_only:
        if checkpoint is None:
            raise ValueError("--eval-only requires --resume or an auto-resume checkpoint.")
        stats = evaluate_segmentation(
            model,
            val_loader,
            device,
            cfg["model"]["num_classes"],
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
        train_stats = train_segmentation_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            precision=precision,
            channels_last=channels_last,
            clip_grad=cfg["train"].get("clip_grad", 5.0),
            show_progress=show_progress,
        )
        val_stats = evaluate_segmentation(
            model,
            val_loader,
            device,
            cfg["model"]["num_classes"],
            precision=precision,
            channels_last=channels_last,
            show_progress=show_progress,
        )
        scheduler.step()
        is_best = val_stats["miou"] >= best_miou
        best_miou = max(best_miou, val_stats["miou"])
        state = {
            "epoch": epoch,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_miou": best_miou,
            "config": cfg,
        }
        metrics = {
            "epoch": epoch,
            "lr": [group["lr"] for group in optimizer.param_groups],
            "train": train_stats,
            "val": val_stats,
            "best_miou": best_miou,
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
