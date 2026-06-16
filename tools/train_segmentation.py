import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from egcvmamba.data import build_segmentation_loaders
from egcvmamba.engine import evaluate_segmentation, make_scaler, train_segmentation_epoch
from egcvmamba.models import build_segmentation_model
from egcvmamba.utils import load_config, save_checkpoint, set_seed, write_json


def parse_args():
    parser = argparse.ArgumentParser("EGCVMamba ADE20K training")
    parser.add_argument("--config", default="configs/segmentation/ade20k_tiny.yaml")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_path is not None:
        cfg["data"]["path"] = args.data_path
    if args.output is not None:
        cfg["output"] = args.output
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_segmentation_model(
        cfg["model"]["variant"],
        cfg["model"]["num_classes"],
        cfg["model"]["drop_path_rate"],
        cfg["model"]["decoder_channels"],
    ).to(device)
    train_loader, val_loader = build_segmentation_loaders(
        cfg["data"]["path"],
        cfg["data"]["image_size"],
        cfg["train"]["batch_size"],
        cfg["train"]["workers"],
    )
    optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=cfg["train"]["warmup_epochs"])
    cosine = CosineAnnealingLR(optimizer, T_max=max(cfg["train"]["epochs"] - cfg["train"]["warmup_epochs"], 1))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[cfg["train"]["warmup_epochs"]])
    scaler = make_scaler(device)
    start_epoch = 0
    best_miou = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_miou = checkpoint.get("best_miou", 0.0)
    output = Path(cfg["output"])
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        train_stats = train_segmentation_epoch(model, train_loader, optimizer, device, scaler)
        val_stats = evaluate_segmentation(model, val_loader, device, cfg["model"]["num_classes"])
        scheduler.step()
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_miou": max(best_miou, val_stats["miou"]),
            "config": cfg,
        }
        save_checkpoint(output / "last.pth", state)
        if val_stats["miou"] >= best_miou:
            best_miou = val_stats["miou"]
            save_checkpoint(output / "best.pth", state)
        write_json(output / "metrics.json", {"epoch": epoch, "train": train_stats, "val": val_stats, "best_miou": best_miou})
        print({"epoch": epoch, "train": train_stats, "val": val_stats, "best_miou": best_miou})


if __name__ == "__main__":
    main()
