import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from Module.GammaBlock import GammaBlock
from Module.BetaBlock import BetaBlock
from Module.AlphaBlock import AlphaBlock
from Module.EVSSBlock import EVSSBlock
import itertools
from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite
from functools import partial
import pywt
import pywt.data
from timm.layers import DropPath
import sys
import os
from vmamba import SS2D
from Component.ConvBNSiLU import ConvBNSiLU
from Component.ConvBN import ConvBN
from Component.ECA import ECA
from Module.STEM import ReDSBlockforSTEM
from Component.scalemodule import _ScaleModule
from Component.Layernorm import LayerNorm
from Component.SS2D import SS2DBlock
from Component.FFN import FFN
from Component.MBWTConv2D import MBWTConv2d

__all__ = ['EGCVMamba', 'EGCVMamba_tiny', 'EGCVMamba_small', 'EGCVMamba_base']


class ReparameterizedStageBlock(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class EGCVBlockS1(nn.Module):
    def __init__(self, channels, ls_large_kernel=7, expand_ratio=2):
        super().__init__()
        self.channels = channels
        self.AlphaBlock = AlphaBlock(channels)
        self.use_Alpha = True
        self.use_Beta = False
        self.use_Gamma = False

    def forward(self, x):
        if self.use_Alpha: x = self.AlphaBlock(x)
        if self.use_Beta: x = self.BetaBlock(x)
        if self.use_Gamma: x = self.GammaBlock(x)
        return x

    def reparameterize(self):
        layers = []
        if self.use_Alpha: layers.append(self.AlphaBlock.reparameterize())
        if self.use_Beta:
            layers.append(
                self.BetaBlock.reparameterize() if hasattr(self.BetaBlock, 'reparameterize') else self.BetaBlock)
        if self.use_Gamma:
            layers.append(
                self.GammaBlock.reparameterize() if hasattr(self.GammaBlock, 'reparameterize') else self.GammaBlock)
        return ReparameterizedStageBlock(layers)


class EGCVBlockS2(nn.Module):
    def __init__(self, channels, ls_large_kernel=7, expand_ratio=2):
        super().__init__()
        self.channels = channels
        self.AlphaBlock = AlphaBlock(channels)
        self.GammaBlock = GammaBlock(channels)
        self.use_Alpha = True
        self.use_Beta = False
        self.use_Gamma = True

    def forward(self, x):
        if self.use_Alpha: x = self.AlphaBlock(x)
        if self.use_Beta: x = self.BetaBlock(x)
        if self.use_Gamma: x = self.GammaBlock(x)
        return x

    def reparameterize(self):
        layers = []
        if self.use_Alpha: layers.append(self.AlphaBlock.reparameterize())
        if self.use_Beta:
            layers.append(
                self.BetaBlock.reparameterize() if hasattr(self.BetaBlock, 'reparameterize') else self.BetaBlock)
        if self.use_Beta:
            layers.append(
                self.GammaBlock.reparameterize() if hasattr(self.GammaBlock, 'reparameterize') else self.GammaBlock)
        return ReparameterizedStageBlock(layers)


class EGCVBlockS3(nn.Module):
    def __init__(self, channels, ls_large_kernel=7, expand_ratio=2):
        super().__init__()
        self.channels = channels
        self.BetaBlock = BetaBlock(channels)
        self.GammaBlock = GammaBlock(channels)
        self.use_Alpha = False
        self.use_Gamma = True
        self.use_Beta = True

    def forward(self, x):
        if self.use_Alpha: x = self.AlphaBlock(x)
        if self.use_Beta: x = self.BetaBlock(x)
        if self.use_Gamma: x = self.GammaBlock(x)
        return x

    def reparameterize(self):
        layers = []
        if self.use_Alpha: layers.append(self.AlphaBlock.reparameterize())
        if self.use_Beta:
            layers.append(
                self.BetaBlock.reparameterize() if hasattr(self.BetaBlock, 'reparameterize') else self.BetaBlock)
        if self.use_Gamma:
            layers.append(
                self.GammaBlock.reparameterize() if hasattr(self.GammaBlock, 'reparameterize') else self.GammaBlock)
        return ReparameterizedStageBlock(layers)


class EGCVBlockS4(nn.Module):
    def __init__(self, channels, ls_large_kernel=7, expand_ratio=2):
        super().__init__()
        self.evss_block = EVSSBlock(
            dim=channels,
            ssm_ratio=2,
            d_state=16,
            forward_type="v052d",
            dw_kernel_size=ls_large_kernel,
            mlp_ratio=4,
            drop_path=0.1
        )
        self.use_EVSS = True

    def forward(self, x):
        x = self.evss_block(x)
        return x

    def reparameterize(self):
        layers = []
        if self.use_EVSS:
            layers.append(self.evss_block.reparameterize())
        return ReparameterizedStageBlock(layers)


class ReDSBlockforSTAGE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        hidden_dim = max(in_channels // 2, 32)
        self.dw_downsample = ConvBNSiLU(
            in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels
        )
        self.pw_downsample = ConvBNSiLU(in_channels, hidden_dim, kernel_size=1)
        self.dw_feature = ConvBNSiLU(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.eca = ECA(hidden_dim)
        self.conv_expand = ConvBNSiLU(hidden_dim, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.dw_downsample(x)
        x = self.pw_downsample(x)
        identity = x
        x = self.dw_feature(x)
        x = self.eca(x)
        x = x + identity
        x = self.conv_expand(x)
        return x

    def reparameterize(self):
        for name, child in list(self.named_children()):
            if hasattr(child, 'reparameterize') and callable(child.reparameterize):
                setattr(self, name, child.reparameterize())
        return self


class EGCVMamba(nn.Module):
    def __init__(self, stem_channels=32,
                 stage_channels=[64, 128, 256, 512], stage_blocks=[2, 3, 6, 3],
                 ls_large_kernels=[5, 7, 9, 9], expand_ratio=2, drop_path_rate=0.1):
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.stem = ReDSBlockforSTEM(in_channels=3, out_channels=stem_channels)
        self.stage1 = self._make_stage1(stem_channels, stage_channels[0], stage_blocks[0], ls_large_kernels[0],
                                        expand_ratio, downsample=True)
        self.stage2 = self._make_stage2(stage_channels[0], stage_channels[1], stage_blocks[1], ls_large_kernels[1],
                                        expand_ratio, downsample=True)
        self.stage3 = self._make_stage3(stage_channels[1], stage_channels[2], stage_blocks[2], ls_large_kernels[2],
                                        expand_ratio, downsample=True)
        self.stage4 = self._make_stage4(stage_channels[2], stage_channels[3], stage_blocks[3], ls_large_kernels[3],
                                        expand_ratio, downsample=False)
        self._init_weights()

    def _make_stage1(self, in_channels, out_channels, num_blocks, ls_large_kernel, expand_ratio, downsample=True):
        layers = []
        if in_channels != out_channels:
            layers.append(ConvBNSiLU(in_channels, out_channels, kernel_size=1))
        for _ in range(num_blocks):
            layers.append(EGCVBlockS1(out_channels, ls_large_kernel=ls_large_kernel, expand_ratio=expand_ratio))
        if downsample:
            layers.append(ReDSBlockforSTAGE(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _make_stage2(self, in_channels, out_channels, num_blocks, ls_large_kernel, expand_ratio, downsample=True):
        layers = []
        if in_channels != out_channels:
            layers.append(ConvBNSiLU(in_channels, out_channels, kernel_size=1))
        for _ in range(num_blocks):
            layers.append(EGCVBlockS2(out_channels, ls_large_kernel=ls_large_kernel, expand_ratio=expand_ratio))
        if downsample:
            layers.append(ReDSBlockforSTAGE(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _make_stage3(self, in_channels, out_channels, num_blocks, ls_large_kernel, expand_ratio, downsample=True):
        layers = []
        if in_channels != out_channels:
            layers.append(ConvBNSiLU(in_channels, out_channels, kernel_size=1))
        for _ in range(num_blocks):
            layers.append(EGCVBlockS3(out_channels, ls_large_kernel=ls_large_kernel, expand_ratio=expand_ratio))
        if downsample:
            layers.append(ReDSBlockforSTAGE(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _make_stage4(self, in_channels, out_channels, num_blocks, ls_large_kernel, expand_ratio, downsample=False):
        layers = []
        if in_channels != out_channels:
            layers.append(ConvBNSiLU(in_channels, out_channels, kernel_size=1))
        for _ in range(num_blocks):
            layers.append(EGCVBlockS4(out_channels, ls_large_kernel=ls_large_kernel, expand_ratio=expand_ratio))
        if downsample:
            layers.append(ReDSBlockforSTAGE(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x

    def reparameterize(self):
        self.stem = self.stem.reparameterize()
        for stage_name in ['stage1', 'stage2', 'stage3', 'stage4']:
            stage = getattr(self, stage_name)
            new_layers = []
            for layer in stage:
                if hasattr(layer, 'reparameterize') and callable(layer.reparameterize):
                    new_layers.append(layer.reparameterize())
                else:
                    new_layers.append(layer)
            setattr(self, stage_name, nn.Sequential(*new_layers))
        return self


def EGCVMamba_tiny(drop_path_rate=0.1):
    return EGCVMamba(
        stem_channels=24,
        stage_channels=[64, 96, 192, 384], stage_blocks=[1, 1, 1, 1],
        ls_large_kernels=[3, 3, 3, 5],
        expand_ratio=2,
        drop_path_rate=drop_path_rate
    )


def EGCVMamba_small(drop_path_rate=0.1):
    return EGCVMamba(
        stem_channels=32,
        stage_channels=[64, 128, 256, 512], stage_blocks=[2, 3, 4, 2],
        ls_large_kernels=[5, 7, 7, 7],
        expand_ratio=2,
        drop_path_rate=drop_path_rate
    )


def EGCVMamba_base(drop_path_rate=0.1):
    return EGCVMamba(
        stem_channels=40,
        stage_channels=[80, 160, 320, 640], stage_blocks=[3, 4, 6, 3],
        ls_large_kernels=[7, 7, 9, 9],
        expand_ratio=2,
        drop_path_rate=drop_path_rate
    )


def EGCVMamba_large(drop_path_rate=0.1):
    return EGCVMamba(
        stem_channels=48,
        stage_channels=[96, 192, 384, 768], stage_blocks=[4, 6, 8, 4],
        ls_large_kernels=[7, 9, 9, 9],
        expand_ratio=2,
        drop_path_rate=drop_path_rate
    )


if __name__ == "__main__":
    model = EGCVMamba_small()
    model.eval()
    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        y = model(x)
    print(f"输入: {x.shape}, 输出: {y.shape}")
    print(f"NaN: {torch.isnan(y).any().item()}, Inf: {torch.isinf(y).any().item()}")


    def count_params(module, name=""):
        n = sum(p.numel() for p in module.parameters())
        print(f"{name:<12s} | {n / 1e6:>6.2f}M")
        return n


    total_before = 0
    total_before += count_params(model.stem, "Stem")
    total_before += count_params(model.stage1, "Stage1")
    total_before += count_params(model.stage2, "Stage2")
    total_before += count_params(model.stage3, "Stage3")
    total_before += count_params(model.stage4, "Stage4")
    print(f"{'总计':<12s} | {total_before / 1e6:>6.2f}M")

    evss_block = model.stage4[1].evss_block
    vss_params = sum(p.numel() for p in evss_block.parameters())
    print(f"\n VSSBlock 参数量: {vss_params / 1e6:.2f}M")

    model_rep = model.reparameterize()
    model_rep.eval()

    total_after = 0
    total_after += count_params(model_rep.stem, "Stem")
    total_after += count_params(model_rep.stage1, "Stage1")
    total_after += count_params(model_rep.stage2, "Stage2")
    total_after += count_params(model_rep.stage3, "Stage3")
    total_after += count_params(model_rep.stage4, "Stage4")
    print(f"{'总计':<12s} | {total_after / 1e6:>6.2f}M")

    import time

    model = EGCVMamba_tiny()
    model.eval()
    model = model.cuda()

    dummy_input = torch.randn(1, 3, 224, 224).cuda()

    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)

    total_time = 0.0
    total_images = 0
    test_seconds = 3

    start_time = time.time()

    with torch.no_grad():
        while time.time() - start_time < test_seconds:
            model(dummy_input)
            total_images += 1

    elapsed_time = time.time() - start_time
    fps = total_images / elapsed_time

    print(f"推理速度：{fps:.2f} 张/秒 (FPS)")
    print(f"总推理图片数：{total_images} 张 / 耗时 {elapsed_time:.2f}s")