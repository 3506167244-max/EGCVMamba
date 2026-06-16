import torch
import math
import torch.nn as nn
from timm.models.layers import DropPath


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * c.groups, w.size(0),
                            w.shape[2:], stride=c.stride, padding=c.padding,
                            dilation=c.dilation, groups=c.groups, device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if hasattr(self.m, 'fuse'):
            fused_m = self.m.fuse()
            if isinstance(fused_m, nn.Conv2d) and fused_m.stride == (1,
                                                                     1) and fused_m.in_channels == fused_m.out_channels:
                identity_kernel = torch.zeros_like(fused_m.weight)
                in_channels = fused_m.in_channels
                groups = fused_m.groups
                kernel_size = fused_m.kernel_size
                pad = fused_m.padding
                if all(k % 2 == 1 for k in kernel_size) and pad == (kernel_size[0] // 2, kernel_size[1] // 2):
                    for i in range(in_channels):
                        if groups == 1:
                            identity_kernel[i, i, kernel_size[0] // 2, kernel_size[1] // 2] = 1.0
                        elif groups == in_channels:
                            identity_kernel[i, 0, kernel_size[0] // 2, kernel_size[1] // 2] = 1.0
                    fused_m.weight.data += identity_kernel
                    if fused_m.bias is None:
                        fused_m.bias = nn.Parameter(torch.zeros(in_channels, device=fused_m.weight.device))
            return fused_m
        return self.m


class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x

    @torch.no_grad()
    def fuse(self):
        fused_pw1 = self.pw1.fuse()
        fused_pw2 = self.pw2.fuse()
        return nn.Sequential(fused_pw1, self.act, fused_pw2)


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed

    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x

    @torch.no_grad()
    def fuse(self):
        fused_conv3x3 = self.conv.fuse()
        fused_conv1x1 = self.conv1.fuse()

        padded_conv1x1 = torch.zeros_like(fused_conv3x3.weight)
        padded_conv1x1[:, :, 1:2, 1:2] = fused_conv1x1.weight

        identity_kernel = torch.zeros_like(fused_conv3x3.weight)
        for i in range(self.dim):
            identity_kernel[i, 0, 1, 1] = 1.0

        fused_weight = fused_conv3x3.weight + padded_conv1x1 + identity_kernel
        fused_bias = fused_conv3x3.bias + fused_conv1x1.bias

        fused_dwconv = nn.Conv2d(
            self.dim, self.dim, kernel_size=3, stride=1, padding=1,
            groups=self.dim, bias=True, device=fused_conv3x3.weight.device
        )
        fused_dwconv.weight.data.copy_(fused_weight)
        fused_dwconv.bias.data.copy_(fused_bias)
        return fused_dwconv


class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


class RGC(nn.Module):
    def __init__(self, dim, expand_ratio=0.5, recursion=2):
        super().__init__()
        hidden_dim = int(dim * expand_ratio)
        self.hidden_dim = hidden_dim
        self.recursion = recursion

        self.pw1 = Conv2d_BN(dim, hidden_dim)
        self.act = nn.ReLU()

        self.dw_convs = nn.ModuleList([
            Conv2d_BN(hidden_dim, hidden_dim, ks=3, pad=1, groups=hidden_dim)
            for _ in range(recursion)
        ])

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2d_BN(hidden_dim, hidden_dim, 1),
            nn.ReLU(),
            Conv2d_BN(hidden_dim, hidden_dim, 1, bn_weight_init=0),
            nn.Sigmoid()
        )

        self.pw2 = Conv2d_BN(hidden_dim, dim, bn_weight_init=0)

    def forward(self, x):
        identity = x
        x = self.act(self.pw1(x))

        for dw in self.dw_convs:
            x = dw(x) + x

        gate = self.gate(x)
        x = x * gate

        x = self.pw2(x)
        return x + identity

    @torch.no_grad()
    def fuse(self):
        fused_pw1 = self.pw1.fuse()

        fused_dw = self.dw_convs[0].fuse()
        identity_kernel = torch.zeros_like(fused_dw.weight)
        for i in range(self.hidden_dim):
            identity_kernel[i, 0, 1, 1] = 1.0
        fused_dw.weight.data += identity_kernel

        for i in range(1, self.recursion):
            current_dw = self.dw_convs[i].fuse()
            new_kernel_size = fused_dw.kernel_size[0] + 2
            new_pad = new_kernel_size // 2
            expanded_fused = torch.zeros(
                self.hidden_dim, 1, new_kernel_size, new_kernel_size,
                device=fused_dw.weight.device
            )
            old_pad = fused_dw.kernel_size[0] // 2
            start = new_pad - old_pad
            end = start + fused_dw.kernel_size[0]
            expanded_fused[:, :, start:end, start:end] = fused_dw.weight
            expanded_current = torch.zeros_like(expanded_fused)
            current_start = new_pad - 1
            current_end = current_start + 3
            expanded_current[:, :, current_start:current_end, current_start:current_end] = current_dw.weight
            for j in range(self.hidden_dim):
                expanded_current[j, 0, new_pad, new_pad] += 1.0
            fused_weight = expanded_fused + expanded_current
            fused_bias = fused_dw.bias + current_dw.bias
            fused_dw = nn.Conv2d(
                self.hidden_dim, self.hidden_dim, kernel_size=new_kernel_size,
                stride=1, padding=new_pad, groups=self.hidden_dim, bias=True,
                device=fused_dw.weight.device
            )
            fused_dw.weight.data.copy_(fused_weight)
            fused_dw.bias.data.copy_(fused_bias)

        fused_gate = nn.Sequential(
            self.gate[0],
            self.gate[1].fuse(),
            self.gate[2],
            self.gate[3].fuse(),
            self.gate[4]
        )

        fused_pw2 = self.pw2.fuse()

        class FusedRGC(nn.Module):
            def __init__(self, pw1, dw, gate, pw2, act):
                super().__init__()
                self.pw1 = pw1
                self.dw = dw
                self.gate = gate
                self.pw2 = pw2
                self.act = act

            def forward(self, x):
                x = self.act(self.pw1(x))
                x = self.dw(x)
                gate = self.gate(x)
                x = x * gate
                x = self.pw2(x)
                return x

        return FusedRGC(fused_pw1, fused_dw, fused_gate, fused_pw2, self.act)


class SimplifiedSKA(nn.Module):
    def __init__(self, dim, groups=4):
        super().__init__()
        self.groups = groups
        self.dim = dim

        self.dw3 = Conv2d_BN(dim, dim, ks=3, pad=1, groups=dim)
        self.dw5 = Conv2d_BN(dim, dim, ks=5, pad=2, groups=dim)
        self.pw = Conv2d_BN(2 * dim, dim, ks=1, bn_weight_init=0)

    def forward(self, x):
        x3 = self.dw3(x)
        x5 = self.dw5(x)
        x = torch.cat([x3, x5], dim=1)
        x = self.pw(x)
        return x

    @torch.no_grad()
    def fuse(self):
        fused_dw3 = self.dw3.fuse()
        fused_dw5 = self.dw5.fuse()
        fused_pw = self.pw.fuse()

        dim = self.dim

        def _fuse_single_branch(dw_conv, pw_conv, pw_in_channel_start, pw_in_channel_end):
            dw_k = dw_conv.weight
            dw_b = dw_conv.bias

            pw_k = pw_conv.weight[:, pw_in_channel_start:pw_in_channel_end, :, :]
            pw_b = pw_conv.bias

            target_k = 5
            dw_k_size = dw_k.shape[2]
            pad = (target_k - dw_k_size) // 2
            expanded_dw_k = torch.zeros(dim, 1, target_k, target_k, device=dw_k.device)
            expanded_dw_k[:, :, pad:pad + dw_k_size, pad:pad + dw_k_size] = dw_k

            fused_k = torch.zeros(dim, dim, target_k, target_k, device=dw_k.device)
            for out_c in range(dim):
                for in_c in range(dim):
                    fused_k[out_c, in_c, :, :] = pw_k[out_c, in_c, 0, 0] * expanded_dw_k[in_c, 0, :, :]

            fused_b_part = torch.sum(pw_k.squeeze(-1).squeeze(-1) * dw_b.unsqueeze(0), dim=1)

            return fused_k, fused_b_part

        fused_k3, fused_b3 = _fuse_single_branch(fused_dw3, fused_pw, 0, dim)
        fused_k5, fused_b5 = _fuse_single_branch(fused_dw5, fused_pw, dim, 2 * dim)

        final_k = fused_k3 + fused_k5
        final_b = fused_b3 + fused_b5 + fused_pw.bias

        fused_conv = nn.Conv2d(
            dim, dim, kernel_size=5, stride=1, padding=2,
            groups=1, bias=True, device=final_k.device
        )
        fused_conv.weight.data.copy_(final_k)
        fused_conv.bias.data.copy_(final_b)
        return fused_conv


class SimplifiedLSConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rgc = RGC(dim)
        self.ska = SimplifiedSKA(dim)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        x_rgc = self.rgc(x)
        x_ska = self.ska(x_rgc)
        return self.bn(x_ska) + x

    @torch.no_grad()
    def fuse(self):
        fused_rgc = self.rgc.fuse()
        fused_ska = self.ska.fuse()

        if isinstance(fused_ska, nn.Conv2d):
            w = self.bn.weight / (self.bn.running_var + self.bn.eps) ** 0.5
            w = fused_ska.weight * w[:, None, None, None]
            b = self.bn.bias - self.bn.running_mean * self.bn.weight / (self.bn.running_var + self.bn.eps) ** 0.5
            fused_ska_bn = nn.Conv2d(
                w.size(1), w.size(0), w.shape[2:],
                stride=fused_ska.stride, padding=fused_ska.padding,
                dilation=fused_ska.dilation, groups=fused_ska.groups,
                device=fused_ska.weight.device, bias=True
            )
            fused_ska_bn.weight.data.copy_(w)
            fused_ska_bn.bias.data.copy_(b)
        else:
            fused_ska_bn = nn.Sequential(fused_ska, self.bn)

        class FusedLSConv(nn.Module):
            def __init__(self, rgc, ska_bn):
                super().__init__()
                self.rgc = rgc
                self.ska_bn = ska_bn

            def forward(self, x):
                x = self.rgc(x)
                x = self.ska_bn(x)
                return x

        return FusedLSConv(fused_rgc, fused_ska_bn)


class GammaBlock(torch.nn.Module):
    def __init__(self, ed):
        super().__init__()
        self.repvgg = Residual(RepVGGDW(ed))
        self.eca = ECA(ed)
        self.lsconv = Residual(SimplifiedLSConv(ed))
        self.ffn = Residual(FFN(ed, int(ed * 2)))

    def forward(self, x):
        x = self.repvgg(x)
        x = self.eca(x)
        x = self.lsconv(x)
        x = self.ffn(x)
        return x

    @torch.no_grad()
    def fuse(self):
        fused_repvgg = self.repvgg.fuse()
        fused_lsconv = self.lsconv.fuse()
        fused_ffn = self.ffn.fuse()

        class FusedCFBlock(nn.Module):
            def __init__(self, repvgg, eca, lsconv, ffn):
                super().__init__()
                self.repvgg = repvgg
                self.eca = eca
                self.lsconv = lsconv
                self.ffn = ffn

            def forward(self, x):
                x = self.repvgg(x)
                x = self.eca(x)
                x = self.lsconv(x)
                x = self.ffn(x)
                return x

        return FusedCFBlock(fused_repvgg, self.eca, fused_lsconv, fused_ffn)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block = GammaBlock(ed=64).to(device)
    block.eval()
    x = torch.randn(2, 64, 14, 14).to(device)

    with torch.no_grad():
        output_train = block(x)

    fused_block = block.fuse().to(device)
    fused_block.eval()

    with torch.no_grad():
        output_infer = fused_block(x)

    torch.allclose(output_train, output_infer, rtol=1e-4, atol=1e-6)