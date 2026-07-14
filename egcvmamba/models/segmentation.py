import torch.nn as nn
import torch.nn.functional as F

from .egcvmamba import build_model
from .layers import ConvNormAct


class FPNHead(nn.Module):
    def __init__(self, in_channels, channels=128, num_classes=150):
        super().__init__()
        self.lateral = nn.ModuleList([ConvNormAct(c, channels, 1, 1, 0) for c in in_channels])
        self.output = nn.Sequential(
            ConvNormAct(channels, channels, 3, 1, 1),
            nn.Conv2d(channels, num_classes, 1),
        )

    def forward(self, features):
        features = [layer(feature) for layer, feature in zip(self.lateral, features)]
        x = features[-1]
        for feature in reversed(features[:-1]):
            x = F.interpolate(x, size=feature.shape[-2:], mode="bilinear", align_corners=False) + feature
        return self.output(x)


class EGCVMambaFPN(nn.Module):
    def __init__(self, variant="tiny", num_classes=150, drop_path_rate=0.1, decoder_channels=128):
        super().__init__()
        self.backbone = build_model(variant, num_classes=1000, drop_path_rate=drop_path_rate, features_only=True)
        self.decode_head = FPNHead(self.backbone.out_channels, decoder_channels, num_classes)

    def forward(self, x):
        input_size = x.shape[-2:]
        features = self.backbone(x)
        logits = self.decode_head(features)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


def build_segmentation_model(variant="tiny", num_classes=150, drop_path_rate=0.1, decoder_channels=128):
    return EGCVMambaFPN(variant, num_classes, drop_path_rate, decoder_channels)
