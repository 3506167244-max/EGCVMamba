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
    """Four-direction, input-dependent 2D state-space scan.

    The scan is intentionally implemented with PyTorch operations instead of a
    custom CUDA extension. Stage 4 only sees a 7x7 map for a 224x224 input, so
    this implementation is fast enough for training while remaining portable to
    new CUDA architectures (including Blackwell).
    """

    def __init__(self, channels, expand_ratio=2, state_dim=8, dt_rank="auto"):
        super().__init__()
        hidden = int(channels * expand_ratio)
        dt_rank = max(channels // 16, 1) if dt_rank == "auto" else int(dt_rank)
        self.hidden = hidden
        self.state_dim = state_dim
        self.dt_rank = dt_rank
        self.num_directions = 4

        self.in_proj = nn.Conv2d(channels, hidden * 2, 1)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.x_proj_weight = nn.Parameter(
            torch.empty(self.num_directions, dt_rank + state_dim * 2, hidden)
        )
        self.dt_proj_weight = nn.Parameter(torch.empty(self.num_directions, hidden, dt_rank))
        self.dt_proj_bias = nn.Parameter(torch.empty(self.num_directions, hidden))
        self.A_log = nn.Parameter(torch.empty(self.num_directions, hidden, state_dim))
        self.D = nn.Parameter(torch.ones(self.num_directions, hidden))
        self.out_norm = LayerNorm2d(hidden)
        self.out_proj = nn.Conv2d(hidden, channels, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for direction_weight in self.x_proj_weight:
            nn.init.xavier_uniform_(direction_weight)
        bound = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj_weight, -bound, bound)

        # Start with time steps log-uniformly distributed in [1e-3, 1e-1].
        dt = torch.exp(
            torch.rand(self.num_directions, self.hidden) * (torch.log(torch.tensor(0.1)) - torch.log(torch.tensor(1e-3)))
            + torch.log(torch.tensor(1e-3))
        )
        inv_softplus = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj_bias.copy_(inv_softplus)
            base = torch.arange(1, self.state_dim + 1, dtype=torch.float32).log()
            self.A_log.copy_(base.view(1, 1, -1).expand_as(self.A_log))

    @staticmethod
    def _cross_scan(x):
        row_major = x.flatten(2)
        column_major = x.transpose(2, 3).contiguous().flatten(2)
        return torch.stack(
            [row_major, column_major, row_major.flip(-1), column_major.flip(-1)],
            dim=1,
        )

    def _selective_scan(self, u, dt, B, C):
        input_dtype = u.dtype
        u = u.float()
        dt = F.softplus(dt.float())
        B = B.float()
        C = C.float()
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        batch, directions, hidden, length = u.shape
        state = u.new_zeros(batch, directions, hidden, self.state_dim)
        outputs = []
        for index in range(length):
            dt_t = dt[..., index]
            transition = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            input_update = (
                dt_t.unsqueeze(-1)
                * B[..., index].unsqueeze(2)
                * u[..., index].unsqueeze(-1)
            )
            state = transition * state + input_update
            y = (state * C[..., index].unsqueeze(2)).sum(-1)
            outputs.append(y + D.unsqueeze(0) * u[..., index])
        return torch.stack(outputs, dim=-1).to(input_dtype)

    @staticmethod
    def _cross_merge(y, height, width):
        batch, _, hidden, _ = y.shape
        row = y[:, 0] + y[:, 2].flip(-1)
        column = y[:, 1] + y[:, 3].flip(-1)
        column = column.view(batch, hidden, width, height).transpose(2, 3).contiguous().flatten(2)
        return ((row + column) * 0.25).view(batch, hidden, height, width)

    def forward(self, x):
        _, _, height, width = x.shape
        u, gate = self.in_proj(x).chunk(2, dim=1)
        u = F.silu(self.dwconv(u), inplace=True)
        sequences = self._cross_scan(u)

        projected = torch.einsum("bkdl,kcd->bkcl", sequences, self.x_proj_weight)
        dt_features, B, C = torch.split(projected, [self.dt_rank, self.state_dim, self.state_dim], dim=2)
        dt = torch.einsum("bkrl,kdr->bkdl", dt_features, self.dt_proj_weight)
        dt = dt + self.dt_proj_bias.unsqueeze(0).unsqueeze(-1)

        scanned = self._selective_scan(sequences, dt, B, C)
        merged = self._cross_merge(scanned, height, width)
        merged = self.out_norm(merged) * F.silu(gate)
        return self.out_proj(merged)


class EVSSBlock(nn.Module):
    def __init__(self, channels, drop_path=0.0):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.scan = SelectiveScan2D(channels)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        return x + self.drop_path(self.scan(self.norm(x)))
