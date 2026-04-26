import torch.nn as nn

import torch
import torch.nn as nn


class ConvBN(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

    def fuse_conv_bn(self, conv, bn):

        w_conv = conv.weight.data
        mean = bn.running_mean
        var_sqrt = torch.sqrt(bn.running_var + bn.eps)
        gamma = bn.weight
        beta = bn.bias

        w_fused = w_conv * (gamma / var_sqrt).reshape(-1, 1, 1, 1)
        b_conv = conv.bias.data if conv.bias is not None else torch.zeros_like(mean)
        b_fused = beta + (b_conv - mean) * (gamma / var_sqrt)

        return w_fused, b_fused

    def reparameterize(self):

        fused_conv = nn.Conv2d(
            self.conv.in_channels, self.conv.out_channels,
            self.conv.kernel_size, self.conv.stride, self.conv.padding,
            groups=self.conv.groups, bias=True
        )

        w_fused, b_fused = self.fuse_conv_bn(self.conv, self.bn)
        fused_conv.weight.data = w_fused
        fused_conv.bias.data = b_fused
        return fused_conv
