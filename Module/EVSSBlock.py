import torch
import torch.nn as nn
from Component.SS2D import SS2DBlock
from Component.Layernorm import LayerNorm
from timm.models.layers import drop_path, DropPath


class EVSSBlock(nn.Module):

    def __init__(self, dim, ssm_ratio=1, d_state=4, forward_type="v05",
                 dw_kernel_size=3, mlp_ratio=0, drop_path=0., act_layer=nn.GELU):
        super().__init__()

        self.norm1 = LayerNorm(dim, data_format="channels_first")
        self.ss2d_block = SS2DBlock(
            dim=dim,
            ssm_ratio=1,
            d_state=4,
            forward_type=forward_type,
            dw_kernel_size=dw_kernel_size,
            drop_path=drop_path
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path2(self.ss2d_block(self.norm1(x)))
        return x
