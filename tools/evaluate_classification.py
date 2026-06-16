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
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_path is not None:
        cfg["data"]["path"] = args.data_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]["variant"], cfg["model"]["num_classes"], cfg["model"]["drop_path_rate"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint.get("model", checkpoint))
    _, val_loader = build_classification_loaders(
        cfg["data"]["path"],
        cfg["data"]["image_size"],
        cfg["train"]["batch_size"],
        cfg["train"]["workers"],
    )
    print(evaluate_classification(model, val_loader, device))


if __name__ == "__main__":
    main()
