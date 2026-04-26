import torch
import torch.nn as nn

class ConvBNSiLU(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.silu(x)
        return x

    def reparameterize(self):

        w = self.conv.weight.data
        bn_mean = self.bn.running_mean
        bn_var = self.bn.running_var
        bn_gamma = self.bn.weight.data
        bn_beta = self.bn.bias.data
        bn_eps = self.bn.eps

        std = torch.sqrt(bn_var + bn_eps)
        w_fused = w * (bn_gamma / std).reshape(-1, 1, 1, 1)
        b_fused = bn_beta - (bn_mean * bn_gamma) / std

        fused_conv = nn.Conv2d(
            in_channels=self.conv.in_channels,
            out_channels=self.conv.out_channels,
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            groups=self.conv.groups,
            bias=True
        )
        fused_conv.weight.data = w_fused
        fused_conv.bias.data = b_fused

        return nn.Sequential(fused_conv, self.silu)