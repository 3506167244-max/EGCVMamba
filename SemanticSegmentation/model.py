from egcvmamba.models import EGCVMambaFPN, build_segmentation_model


def EGCVMamba_tiny(num_classes=150, drop_path_rate=0.05):
    return build_segmentation_model("tiny", num_classes=num_classes, drop_path_rate=drop_path_rate)


def EGCVMamba_small(num_classes=150, drop_path_rate=0.08):
    return build_segmentation_model("small", num_classes=num_classes, drop_path_rate=drop_path_rate)


class FPNSegHead(EGCVMambaFPN):
    def __init__(self, variant="tiny", num_classes=150, drop_path_rate=0.05, decoder_channels=128, pretrained_weight_path=None, input_size=None):
        super().__init__(variant=variant, num_classes=num_classes, drop_path_rate=drop_path_rate, decoder_channels=decoder_channels)


__all__ = [
    "EGCVMambaFPN",
    "FPNSegHead",
    "EGCVMamba_tiny",
    "EGCVMamba_small",
    "build_segmentation_model",
]
