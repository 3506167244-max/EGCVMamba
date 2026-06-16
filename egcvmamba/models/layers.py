import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, act_layer=nn.SiLU):
        if padding is None:
            padding = kernel_size // 2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if act_layer is not None:
            layers.append(act_layer(inplace=True) if act_layer in (nn.ReLU, nn.SiLU) else act_layer())
        super().__init__(*layers)


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask


class ECA(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        y = self.pool(x).squeeze(-1).transpose(1, 2)
        y = self.conv(y).transpose(1, 2).unsqueeze(-1).sigmoid()
        return x * y


class SE(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class ReparamDWConv(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.dw = ConvNormAct(channels, channels, kernel_size, stride, padding, groups=channels)
        self.pw = ConvNormAct(channels, channels, 1, stride, 0, groups=channels, act_layer=None)
        self.use_identity = stride == 1

    def forward(self, x):
        y = self.dw(x) + self.pw(x)
        if self.use_identity:
            y = y + x
        return F.silu(y, inplace=True)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        hidden = max(in_channels // 2, 32)
        self.down = ConvNormAct(in_channels, in_channels, 3, 2, 1, groups=in_channels)
        self.reduce = ConvNormAct(in_channels, hidden, 1, 1, 0)
        self.local = ConvNormAct(hidden, hidden, 3, 1, 1, groups=hidden)
        self.eca = ECA(hidden)
        self.expand = ConvNormAct(hidden, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.down(x)
        x = self.reduce(x)
        x = self.local(x) + x
        x = self.eca(x)
        return self.expand(x)
