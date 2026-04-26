import torch
import torch.nn as nn
from Component.ConvBN import ConvBN
from Component.ConvBNSiLU import ConvBNSiLU

class RepVGGStemBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_identity=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_identity = use_identity and (stride == 1) and (in_channels == out_channels)
        self.branched_conv3x3 = ConvBN(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.branched_conv1x1 = ConvBN(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        out = self.branched_conv3x3(x) + self.branched_conv1x1(x)
        if self.use_identity:
            out = out + x
        return self.silu(out)

    def reparameterize(self):

        fused_conv3x3 = self.branched_conv3x3.reparameterize()
        fused_conv1x1 = self.branched_conv1x1.reparameterize()

        fused_conv1x1_pad = nn.Conv2d(
            fused_conv1x1.in_channels, fused_conv1x1.out_channels,
            kernel_size=3, stride=fused_conv1x1.stride, padding=1,
            groups=fused_conv1x1.groups, bias=True
        )
        fused_conv1x1_pad.weight.data.zero_()
        fused_conv1x1_pad.weight.data[:, :, 1:2, 1:2] = fused_conv1x1.weight.data
        fused_conv1x1_pad.bias.data = fused_conv1x1.bias.data
        # 3. 融合Identity分支
        identity_conv = None
        if self.use_identity:
            identity_conv = nn.Conv2d(
                self.in_channels, self.out_channels,
                kernel_size=3, stride=1, padding=1, groups=1, bias=True
            )
            identity_conv.weight.data.zero_()
            for i in range(self.in_channels):
                identity_conv.weight.data[i, i, 1, 1] = 1.0
            identity_conv.bias.data.zero_()

        final_weight = fused_conv3x3.weight.data + fused_conv1x1_pad.weight.data
        final_bias = fused_conv3x3.bias.data + fused_conv1x1_pad.bias.data
        if identity_conv is not None:
            final_weight += identity_conv.weight.data
            final_bias += identity_conv.bias.data

        final_conv = nn.Conv2d(
            self.in_channels, self.out_channels,
            kernel_size=3, stride=self.stride, padding=1, bias=True
        )
        final_conv.weight.data = final_weight
        final_conv.bias.data = final_bias
        return nn.Sequential(final_conv, self.silu)

class ReDSBlockforSTEM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.stem = nn.Sequential(
            RepVGGStemBlock(in_channels, 64, stride=2, use_identity=False),
            RepVGGStemBlock(64, 64, stride=2, use_identity=True),
            RepVGGStemBlock(64, 64, stride=2, use_identity=False),
            ConvBNSiLU(64, out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.stem(x)

    def reparameterize(self):
        new_stem = []
        for module in self.stem:
            if hasattr(module, 'reparameterize') and callable(module.reparameterize):
                new_stem.append(module.reparameterize())
            else:
                new_stem.append(module)
        self.stem = nn.Sequential(*new_stem)
        return self