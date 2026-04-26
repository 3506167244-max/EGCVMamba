import torch
from torch import nn
from scalemodule import _ScaleModule
from ImageClassification.vmamba import SS2D
import torch.nn.functional as F

class MBWTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True,
                 wt_levels=1, wt_type='db1', ssm_ratio=1, forward_type="v05"):
        super(MBWTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.stride = stride

        # ==================== 只保留 Mamba ====================
        if SS2D is not None:
            self.global_atten = SS2D(
                d_model=in_channels,
                d_state=1,
                ssm_ratio=ssm_ratio,
                initialize="v2",
                forward_type=forward_type,
                channel_first=True,
                k_group=2
            )
        else:
            self.global_atten = nn.Identity()

        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride, groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):
        x = self.base_scale(self.global_atten(x))

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x