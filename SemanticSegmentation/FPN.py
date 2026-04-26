import torch
import torch.nn as nn
import torch.nn.functional as F
from model import EGCVMamba_tiny,EGCVMamba_small,EGCVMamba_base,EGCVMamba_large


class FPNSegHead(nn.Module):
    def __init__(self, in_channels_list=[24, 64, 96, 192, 384], embed_dim=128, num_classes=151):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels_list[0], embed_dim, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels_list[1], embed_dim, 1, bias=False)
        self.conv3 = nn.Conv2d(in_channels_list[2], embed_dim, 1, bias=False)
        self.conv4 = nn.Conv2d(in_channels_list[3], embed_dim, 1, bias=False)
        self.conv5 = nn.Conv2d(in_channels_list[4], embed_dim, 1, bias=False)

        self.smooth1 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False)
        self.smooth2 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False)
        self.smooth3 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False)
        self.smooth4 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False)

        self.out_conv = nn.Conv2d(embed_dim, num_classes, 1)
        self.norm = nn.BatchNorm2d(embed_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, features):
        c1, c2, c3, c4, c5 = features

        p5 = self.conv5(c5)
        p4 = self.conv4(c4) + F.interpolate(p5, c4.shape[2:], mode='bilinear', align_corners=False)
        p3 = self.conv3(c3) + F.interpolate(p4, c3.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.conv2(c2) + F.interpolate(p3, c2.shape[2:], mode='bilinear', align_corners=False)
        p1 = self.conv1(c1) + F.interpolate(p2, c1.shape[2:], mode='bilinear', align_corners=False)

        p1 = self.smooth1(p1)
        p1 = self.norm(p1)
        p1 = self.act(p1)

        out = self.out_conv(p1)
        out = F.interpolate(out, size=(512, 512), mode='bilinear', align_corners=False)
        return out


class SegNet(nn.Module):
    def __init__(self, pretrained_weight_path, embed_dim=128, num_classes=151):
        super().__init__()
        self.backbone = EGCVMamba_tiny(num_classes=100, drop_path_rate=0.05)

        checkpoint = torch.load(pretrained_weight_path, map_location="cuda" if torch.cuda.is_available() else "cpu",
                                weights_only=True)
        self.backbone.load_state_dict(checkpoint)

        def extract_features(x):
            c1 = self.backbone.stem(x)
            c2 = self.backbone.stage1(c1)
            c3 = self.backbone.stage2(c2)
            c4 = self.backbone.stage3(c3)
            c5 = self.backbone.stage4(c4)
            return c1, c2, c3, c4, c5

        self.backbone.forward = extract_features
        self.seg_head = FPNSegHead(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.seg_head(features)