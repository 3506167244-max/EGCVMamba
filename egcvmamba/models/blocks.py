import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvNormAct, DropPath, ECA, LayerNorm2d, ReparamDWConv, SE


class Stem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, stem_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            ReparamDWConv(in_channels, 3, 2),
            ConvNormAct(in_channels, stem_channels, 1, 1, 0),
            ReparamDWConv(stem_channels, 3, 1),
            ReparamDWConv(stem_channels, 3, 2),
            ConvNormAct(stem_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        return self.net(x)


class AlphaBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.local = ConvNormAct(channels, channels, 3, 1, 1, groups=channels)
        self.se = SE(channels)
        self.mix = nn.Sequential(
            ConvNormAct(channels, channels, 1, 1, 0),
            ConvNormAct(channels, channels, 1, 1, 0, act_layer=None),
        )

    def forward(self, x):
        y = self.local(x)
        y = self.se(y)
        y = self.mix(y)
        return F.silu(x + y, inplace=True)


class ChannelAggregation(nn.Module):
    def __init__(self, channels, ratio=4):
        super().__init__()
        hidden = int(channels * ratio)
        self.value = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 1),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.ones(1, hidden, 1, 1) * 1e-5)
        self.proj = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        v = self.value(x)
        g = self.gate(x)
        v = v + self.scale * (v - v.mean(dim=(2, 3), keepdim=True))
        return self.proj(v * g)


class HierarchicalGatedFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        c1 = channels // 2
        c2 = channels - c1
        self.pre = nn.Conv2d(channels, channels, 1)
        self.dw1 = nn.Conv2d(c1, c1, 3, padding=1, groups=c1)
        self.dw2 = nn.Conv2d(c2, c2, 3, padding=2, dilation=2, groups=c2)
        self.shortcut = nn.Conv2d(channels, channels, 1)
        self.global_proj = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))
        self.delta = nn.Parameter(torch.ones(1, channels, 1, 1) * 1e-5)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        x = self.pre(x)
        a, b = torch.split(x, [self.dw1.in_channels, self.dw2.in_channels], dim=1)
        y = torch.cat([self.dw1(a), self.dw2(b)], dim=1)
        y = F.silu(y, inplace=True)
        y = y * self.shortcut(x)
        y = y - self.delta * self.global_proj(y)
        return self.out(y)


class BetaBlock(nn.Module):
    def __init__(self, channels, drop_path=0.0):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        self.hgf = HierarchicalGatedFusion(channels)
        self.ca = ChannelAggregation(channels)
        self.lambda1 = nn.Parameter(torch.ones(1) * 1e-2)
        self.lambda2 = nn.Parameter(torch.ones(1) * 1e-2)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.lambda1 * self.hgf(self.norm1(x)))
        x = x + self.drop_path(self.lambda2 * self.ca(self.norm2(x)))
        return x


class RecursiveGatedConv(nn.Module):
    def __init__(self, channels, ratio=0.5):
        super().__init__()
        hidden = max(int(channels * ratio), 16)
        self.pre = ConvNormAct(channels, hidden, 1, 1, 0, act_layer=nn.ReLU)
        self.dw1 = ConvNormAct(hidden, hidden, 3, 1, 1, groups=hidden, act_layer=nn.ReLU)
        self.dw2 = ConvNormAct(hidden, hidden, 3, 1, 1, groups=hidden, act_layer=nn.ReLU)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1),
            nn.Sigmoid(),
        )
        self.proj = ConvNormAct(hidden, channels, 1, 1, 0, act_layer=None)

    def forward(self, x):
        y = self.pre(x)
        y = self.dw1(y) + y
        y = self.dw2(y) + y
        y = y * self.gate(y)
        return x + self.proj(y)


class LocalKernelFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw3 = ConvNormAct(channels, channels, 3, 1, 1, groups=channels, act_layer=None)
        self.dw5 = ConvNormAct(channels, channels, 5, 1, 2, groups=channels, act_layer=None)
        self.proj = ConvNormAct(channels * 2, channels, 1, 1, 0)

    def forward(self, x):
        return self.proj(torch.cat([self.dw3(x), self.dw5(x)], dim=1))


class GammaBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.rep = ReparamDWConv(channels)
        self.eca = ECA(channels)
        self.rgc = RecursiveGatedConv(channels)
        self.lkf = LocalKernelFusion(channels)

    def forward(self, x):
        x = self.rep(x)
        x = self.eca(x)
        x = self.rgc(x)
        return x + self.lkf(x)


class SelectiveScan2D(nn.Module):
    def __init__(self, channels, state_ratio=2):
        super().__init__()
        hidden = channels * state_ratio
        self.in_proj = nn.Conv2d(channels, hidden, 1)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.gate = nn.Conv2d(channels, hidden, 1)
        self.out_proj = nn.Conv2d(hidden, channels, 1)
        self.alpha = nn.Parameter(torch.ones(1, hidden, 1, 1) * 0.25)

    def forward(self, x):
        u = F.silu(self.dwconv(self.in_proj(x)), inplace=True)
        g = torch.sigmoid(self.gate(x))
        h1 = torch.cumsum(u, dim=2)
        h2 = torch.flip(torch.cumsum(torch.flip(u, dims=[2]), dim=2), dims=[2])
        h3 = torch.cumsum(u, dim=3)
        h4 = torch.flip(torch.cumsum(torch.flip(u, dims=[3]), dim=3), dims=[3])
        h = (h1 + h2 + h3 + h4) * self.alpha
        return self.out_proj(h * g)


class EVSSBlock(nn.Module):
    def __init__(self, channels, drop_path=0.0):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.scan = SelectiveScan2D(channels)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        return x + self.drop_path(self.scan(self.norm(x)))
