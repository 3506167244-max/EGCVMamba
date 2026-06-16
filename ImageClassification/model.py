from egcvmamba.models import EGCVMamba, build_model


def EGCVMamba_tiny(num_classes=1000, drop_path_rate=0.05):
    return build_model("tiny", num_classes=num_classes, drop_path_rate=drop_path_rate)


def EGCVMamba_small(num_classes=1000, drop_path_rate=0.08):
    return build_model("small", num_classes=num_classes, drop_path_rate=drop_path_rate)


def EGCVMamba_base(num_classes=1000, drop_path_rate=0.12):
    return build_model("base", num_classes=num_classes, drop_path_rate=drop_path_rate)


def EGCVMamba_large(num_classes=1000, drop_path_rate=0.16):
    return build_model("large", num_classes=num_classes, drop_path_rate=drop_path_rate)


__all__ = [
    "EGCVMamba",
    "EGCVMamba_tiny",
    "EGCVMamba_small",
    "EGCVMamba_base",
    "EGCVMamba_large",
]
