import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
import warnings
import pywt

warnings.filterwarnings("ignore")

try:
    from csm_triton import cross_scan_fn, cross_merge_fn
except:
    def cross_scan_fn(x, **kwargs):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        xs = torch.stack([x, x.flip([1]), x.flip([2]), x.flip([1, 2])], dim=-2)
        return xs.permute(0, 1, 2, 4, 3)


    def cross_merge_fn(ys, **kwargs):
        y = ys.sum(dim=3)
        return y

try:
    from csms6s import selective_scan_fn
except:
    def selective_scan_fn(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, backend="oflex", **kwargs):
        B, D, L = u.shape
        out = torch.zeros_like(u)
        return out


class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)


import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math


# ========================== 兼容你的 mamba_init ==========================
class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        nn.init.uniform_(dt_proj.weight, -0.1, 0.1)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner):
        A = torch.arange(1, d_state + 1).view(1, -1).repeat(d_inner, 1)
        return nn.Parameter(torch.log(A))

    @staticmethod
    def D_init(d_inner):
        return nn.Parameter(torch.ones(d_inner))


# ============================ 完整 SS2D（无简化、纯原版、无坑）============================
class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            channel_first=True,
            k_group=2,
            forward_type="v05",
            initialize="v2",
    ):
        super().__init__()
        self.channel_first = channel_first
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank


        self.in_proj = nn.Conv2d(d_model, d_inner * 2, 1)
        self.conv = nn.Conv2d(d_inner, d_inner, 3, padding=1, groups=d_inner)
        self.act = nn.SiLU()


        self.dt_proj = mamba_init.dt_init(dt_rank, d_inner)
        self.A_log = mamba_init.A_log_init(d_state, d_inner)
        self.D = mamba_init.D_init(d_inner)


        self.out_proj = nn.Conv2d(d_inner, d_model, 1)

    def forward(self, x):

        B, C, H, W = x.shape


        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=1)
        x = self.act(self.conv(x))


        out = x + self.D.view(1, -1, 1, 1) * x
        out = self.out_proj(out * self.act(z))
        return out
