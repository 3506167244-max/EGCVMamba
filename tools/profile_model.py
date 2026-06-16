import argparse

import torch

from egcvmamba.models import build_model


def parse_args():
    parser = argparse.ArgumentParser("EGCVMamba profiler")
    parser.add_argument("--variant", default="tiny", choices=["tiny", "small", "base", "large"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    model = build_model(args.variant, args.num_classes).eval()
    params = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, 3, args.image_size, args.image_size)
    with torch.no_grad():
        y = model(x)
    result = {"variant": args.variant, "params": params, "params_m": round(params / 1e6, 3), "output_shape": list(y.shape)}
    try:
        from fvcore.nn import FlopCountAnalysis

        flops = FlopCountAnalysis(model, x).total()
        result["flops"] = flops
        result["mflops"] = round(flops / 1e6, 3)
    except Exception as exc:
        result["flops_error"] = str(exc)
    print(result)


if __name__ == "__main__":
    main()
