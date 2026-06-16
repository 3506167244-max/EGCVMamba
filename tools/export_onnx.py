import argparse

import torch

from egcvmamba.models import build_model


def parse_args():
    parser = argparse.ArgumentParser("EGCVMamba ONNX export")
    parser.add_argument("--variant", default="tiny", choices=["tiny", "small", "base", "large"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="checkpoints/egcvmamba_tiny.onnx")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    model = build_model(args.variant, args.num_classes).eval()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint.get("model", checkpoint))
    dummy = torch.randn(1, 3, args.image_size, args.image_size)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(args.output)


if __name__ == "__main__":
    main()
