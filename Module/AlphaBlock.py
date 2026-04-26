import torch
import torch.nn.functional as F
import torch.nn as nn
from Component.ConvBNSiLU import ConvBNSiLU
from Component.ConvBN import ConvBN

class SEBlock(nn.Module):


    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc1(scale)
        scale = self.silu(scale)
        scale = self.fc2(scale)
        scale = torch.sigmoid(scale)
        return x * scale.expand_as(x)


class AlphaBlock(nn.Module):
    def __init__(self, channels, se_reduction=8):
        super().__init__()
        self.channels = channels
        self.residual_scale = 1.0

        self.dw3x3 = ConvBNSiLU(channels, channels, kernel_size=3, groups=channels)
        self.se = SEBlock(channels, reduction=se_reduction)
        self.conv1x1_1 = ConvBNSiLU(channels, channels, kernel_size=1)
        self.conv1x1_2 = ConvBN(channels, channels, kernel_size=1)
        self.has_residual = True

    def forward(self, x):
        identity = x
        out = self.dw3x3(x)
        out = self.se(out)
        out = self.conv1x1_1(out)
        out = self.conv1x1_2(out)
        if self.has_residual:
            out = self.residual_scale * out + identity
        return out

    def reparameterize(self):
        device = next(self.parameters()).device

        fused_dw3x3 = self._fuse_conv_bn(self.dw3x3.conv, self.dw3x3.bn)

        fused_conv1x1_1 = self._fuse_conv_bn(self.conv1x1_1.conv, self.conv1x1_1.bn)

        fused_conv1x1_2 = self._fuse_conv_bn(self.conv1x1_2.conv, self.conv1x1_2.bn)

        c = self.channels
        identity_kernel = torch.eye(c, c, device=device).view(c, c, 1, 1)
        identity_bias = torch.zeros(c, device=device)

        main_kernel = fused_conv1x1_2.weight.data * self.residual_scale
        main_bias = fused_conv1x1_2.bias.data * self.residual_scale

        final_kernel = main_kernel + identity_kernel
        final_bias = main_bias + identity_bias

        fused_final_conv = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True)
        fused_final_conv.weight.data = final_kernel
        fused_final_conv.bias.data = final_bias

        return ReparameterizedAlphaBlock(fused_dw3x3, self.se, fused_conv1x1_1, fused_final_conv)

    @staticmethod
    def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
        if bn is None:
            return conv
        conv_w = conv.weight.data
        bn_rm = bn.running_mean.data
        bn_rv = bn.running_var.data
        bn_w = bn.weight.data
        bn_b = bn.bias.data
        eps = bn.eps

        std = (bn_rv + eps).sqrt()
        t = (bn_w / std).reshape(-1, 1, 1, 1)
        fused_w = conv_w * t
        fused_b = bn_b - bn_rm * bn_w / std

        fused_conv = nn.Conv2d(
            conv.in_channels, conv.out_channels,
            kernel_size=conv.kernel_size, stride=conv.stride,
            padding=conv.padding, groups=conv.groups, bias=True
        )
        fused_conv.weight.data = fused_w
        fused_conv.bias.data = fused_b
        return fused_conv


class ReparameterizedAlphaBlock(nn.Module):


    def __init__(self, fused_dw3x3, se_module, fused_conv1x1_1, fused_final_conv):
        super().__init__()
        self.dw3x3 = fused_dw3x3
        self.silu1 = nn.SiLU()
        self.se = se_module
        self.conv1x1_1 = fused_conv1x1_1
        self.silu2 = nn.SiLU()
        self.conv1x1_2 = fused_final_conv

    def forward(self, x):
        x = self.dw3x3(x)
        x = self.silu1(x)
        x = self.se(x)
        x = self.conv1x1_1(x)
        x = self.silu2(x)
        x = self.conv1x1_2(x)
        return x