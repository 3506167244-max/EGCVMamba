import argparse

import torch

from egcvmamba.data import build_classification_loaders
from egcvmamba.engine import evaluate_classification
from egcvmamba.models import build_model
from egcvmamba.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser("EGCVMamba ImageNet evaluation")
    parser.add_argument("--config", default="configs/classification/egcvmamba_tiny.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--weights", choices=["ema", "model"], default="ema")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_path is not None:
        cfg["data"]["path"] = args.data_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]["variant"], cfg["model"]["num_classes"], cfg["model"]["drop_path_rate"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.weights == "ema" and "ema" in checkpoint:
        ema_state = checkpoint["ema"]
        state_dict = ema_state.get("module", ema_state.get("shadow"))
        if state_dict is None:
            raise KeyError("EMA weights were requested but the EMA state is invalid.")
        model.load_state_dict(state_dict, strict="shadow" not in ema_state)
    else:
        model.load_state_dict(checkpoint.get("model", checkpoint))
    _, val_loader = build_classification_loaders(
        cfg["data"]["path"],
        cfg["data"]["image_size"],
        cfg["train"]["batch_size"],
        cfg["train"]["workers"],
    )
    print(
        evaluate_classification(
            model,
            val_loader,
            device,
            precision=cfg["train"].get("precision", "bf16"),
            channels_last=cfg["train"].get("channels_last", True),
        )
    )


if __name__ == "__main__":
    main()
