import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def build_act_layer(act_type):
    if act_type is None:
        return nn.Identity()
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        return nn.GELU()

class ElementScale(nn.Module):
    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super().__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale

class ChannelMixer(nn.Module):
    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 kernel_size=3,
                 act_type='GELU',
                 ffn_drop=0.,
                 groups=16):
        super().__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.groups = groups
        self.hidden_dim = feedforward_channels // 2

        assert embed_dims % groups == 0
        assert feedforward_channels % groups == 0
        assert feedforward_channels % 2 == 0

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1,
            groups=groups,
            bias=False
        )

        self.dwconv3 = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.hidden_dim
        )
        self.dwconv5 = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
            groups=self.hidden_dim
        )

        self.act = build_act_layer(act_type)

        self.fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1,
            groups=groups,
            bias=False
        )
        self.drop = nn.Dropout(ffn_drop)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs(math.log2(self.feedforward_channels) + 1) / 2)
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def feat_decompose(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y

    def forward(self, x):
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=1)
        x3 = self.dwconv3(x1)
        x5 = self.dwconv5(x2)
        x = torch.cat([x3, x5], dim=1)
        x = self.act(x)
        x = self.drop(x)
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

if __name__ == "__main__":
    ca_module = ChannelMixer(
        embed_dims=64,
        feedforward_channels=256,
        kernel_size=3,
        act_type='GELU',
        ffn_drop=0.1,
        groups=16
    )

    test_input = torch.randn(2, 64, 32, 32)
    output = ca_module(test_input)