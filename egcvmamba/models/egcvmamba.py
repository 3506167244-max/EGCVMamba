import torch
import torch.nn as nn

from .blocks import AlphaBlock, BetaBlock, EVSSBlock, GammaBlock, Stem
from .layers import ConvNormAct, DownsampleBlock


MODEL_SPECS = {
    "tiny": {
        "stem_channels": 32,
        "stage_channels": [64, 96, 192, 384],
        "stage_blocks": [1, 1, 1, 1],
    },
    "small": {
        "stem_channels": 32,
        "stage_channels": [64, 128, 256, 512],
        "stage_blocks": [2, 3, 4, 2],
    },
    "base": {
        "stem_channels": 40,
        "stage_channels": [80, 160, 320, 640],
        "stage_blocks": [3, 4, 6, 3],
    },
    "large": {
        "stem_channels": 48,
        "stage_channels": [96, 192, 384, 768],
        "stage_blocks": [4, 6, 8, 4],
    },
}


class Stage(nn.Module):
    def __init__(self, in_channels, out_channels, depth, kind, drop_rates, stem_channels=None):
        super().__init__()
        layers = []
        if kind == "stem":
            layers.append(Stem(3, out_channels, stem_channels or out_channels // 2))
        elif in_channels != out_channels:
            layers.append(ConvNormAct(in_channels, out_channels, 1, 1, 0))
        for index in range(depth):
            drop_path = drop_rates[index] if index < len(drop_rates) else 0.0
            if kind == "alpha":
                layers.append(AlphaBlock(out_channels))
            elif kind == "stem":
                layers.append(AlphaBlock(out_channels))
            elif kind == "alpha_beta":
                layers.append(AlphaBlock(out_channels))
                layers.append(BetaBlock(out_channels, drop_path))
            elif kind == "beta_gamma":
                layers.append(BetaBlock(out_channels, drop_path))
                layers.append(GammaBlock(out_channels))
            elif kind == "evss":
                layers.append(EVSSBlock(out_channels, drop_path))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)


class EGCVMamba(nn.Module):
    def __init__(self, num_classes=1000, stem_channels=32, stage_channels=None, stage_blocks=None, drop_path_rate=0.1, features_only=False):
        super().__init__()
        stage_channels = stage_channels or [64, 128, 256, 512]
        stage_blocks = stage_blocks or [2, 3, 4, 2]
        total_depth = sum(stage_blocks)
        drop_rates = torch.linspace(0, drop_path_rate, total_depth).tolist()
        cursor = 0
        self.num_classes = num_classes
        self.features_only = features_only
        self.out_channels = stage_channels
        self.stage1 = Stage(3, stage_channels[0], stage_blocks[0], "stem", drop_rates[cursor:cursor + stage_blocks[0]], stem_channels)
        cursor += stage_blocks[0]
        self.down1 = DownsampleBlock(stage_channels[0], stage_channels[1])
        self.stage2 = Stage(stage_channels[1], stage_channels[1], stage_blocks[1], "alpha_beta", drop_rates[cursor:cursor + stage_blocks[1]])
        cursor += stage_blocks[1]
        self.down2 = DownsampleBlock(stage_channels[1], stage_channels[2])
        self.stage3 = Stage(stage_channels[2], stage_channels[2], stage_blocks[2], "beta_gamma", drop_rates[cursor:cursor + stage_blocks[2]])
        cursor += stage_blocks[2]
        self.down3 = DownsampleBlock(stage_channels[2], stage_channels[3])
        self.stage4 = Stage(stage_channels[3], stage_channels[3], stage_blocks[3], "evss", drop_rates[cursor:cursor + stage_blocks[3]])
        self.norm = nn.BatchNorm2d(stage_channels[-1])
        self.head = nn.Linear(stage_channels[-1], num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x):
        features = []
        x = self.stage1(x)
        features.append(x)
        x = self.down1(x)
        x = self.stage2(x)
        features.append(x)
        x = self.down2(x)
        x = self.stage3(x)
        features.append(x)
        x = self.down3(x)
        x = self.stage4(x)
        features.append(x)
        return features

    def forward_head(self, x):
        x = self.norm(x)
        x = x.mean(dim=(2, 3))
        return self.head(x)

    def forward(self, x):
        features = self.forward_features(x)
        if self.features_only:
            return features
        return self.forward_head(features[-1])


def build_model(name="tiny", num_classes=1000, drop_path_rate=0.1, features_only=False):
    key = name.lower().replace("egcvmamba_", "").replace("egcvmamba-", "")
    if key not in MODEL_SPECS:
        raise ValueError(f"Unknown EGCVMamba variant: {name}")
    spec = MODEL_SPECS[key]
    return EGCVMamba(
        num_classes=num_classes,
        stem_channels=spec["stem_channels"],
        stage_channels=spec["stage_channels"],
        stage_blocks=spec["stage_blocks"],
        drop_path_rate=drop_path_rate,
        features_only=features_only,
    )


def EGCVMambaTiny(num_classes=1000, drop_path_rate=0.1, features_only=False):
    return build_model("tiny", num_classes, drop_path_rate, features_only)


def EGCVMambaSmall(num_classes=1000, drop_path_rate=0.1, features_only=False):
    return build_model("small", num_classes, drop_path_rate, features_only)


def EGCVMambaBase(num_classes=1000, drop_path_rate=0.1, features_only=False):
    return build_model("base", num_classes, drop_path_rate, features_only)


def EGCVMambaLarge(num_classes=1000, drop_path_rate=0.1, features_only=False):
    return build_model("large", num_classes, drop_path_rate, features_only)
