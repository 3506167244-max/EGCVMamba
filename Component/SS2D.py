import torch
import torch.nn as nn
from ImageClassification.vmamba import SS2D
from Layernorm import LayerNorm
from timm.models.layers import drop_path, DropPath
class SS2DBlock(nn.Module):
    """SS2D 子模块（对应虚线框结构）"""

    def __init__(self, dim, ssm_ratio=2, d_state=16, forward_type="v05",
                 dw_kernel_size=3, drop_path=0.):
        super().__init__()
        # 1. 输入 Linear (1x1 Conv)
        self.fc1 = nn.Conv2d(dim, int(ssm_ratio * dim), kernel_size=1)
        # 2. DWConv
        self.dwconv = nn.Conv2d(int(ssm_ratio * dim), int(ssm_ratio * dim),
                                kernel_size=dw_kernel_size, stride=1,
                                padding=dw_kernel_size // 2, groups=int(ssm_ratio * dim))
        # 3. SiLU 激活
        self.act = nn.SiLU(inplace=True)
        # 4. SS2D 核心模块
        self.ss2d = SS2D(
            d_model=int(ssm_ratio * dim),
            d_state=d_state,
            ssm_ratio=1,
            initialize="v2",
            forward_type=forward_type,
            channel_first=True,
            k_group=2
        )
        # 5. LayerNorm
        self.norm = LayerNorm(int(ssm_ratio * dim), data_format="channels_first")
        # 6. 输出 Linear (1x1 Conv)
        self.fc2 = nn.Conv2d(int(ssm_ratio * dim), dim, kernel_size=1)
        # DropPath 残差
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        # 按结构图顺序执行
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.ss2d(x)
        x = self.norm(x)
        x = self.fc2(x)
        # 残差连接
        x = self.drop_path(x) + shortcut
        return x