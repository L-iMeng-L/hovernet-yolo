# models/hover_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import CBL

class DenseUnit(nn.Module):
    def __init__(self, in_ch, grow_ch):
        super().__init__()
        self.conv = nn.Sequential(
            CBL(in_ch, grow_ch, 1, 1, 0),
            CBL(grow_ch, grow_ch, 3, 1, 1),
        )

    def forward(self, x):
        return torch.cat([x, self.conv(x)], dim=1)

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        fused_ch = in_ch + skip_ch
        self.dense = DenseUnit(fused_ch, out_ch)
        self.compress = CBL(fused_ch + out_ch, out_ch, 1, 1, 0)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.compress(self.dense(x))

class HoverDecoder(nn.Module):
    """
    三分支输出：
      np_map : (B, 1,H, W)  前景概率
      hv_map : (B, 2,           H, W)  水平/垂直距离图
      nc_map : (B, num_classes, H, W)  像素级类别 logits（未经 softmax）
    """
    def __init__(self, base_ch=64, num_classes=5):
        super().__init__()
        b = base_ch

        self.bottleneck = CBL(b * 16, b * 4, 1, 1, 0)# 1024 → 256
        self.up1 = UpBlock(b * 4, b * 8, b * 4)        # 256+512 → 256
        self.up2 = UpBlock(b * 4, b * 4, b * 2)        # 256+256 → 128

        # NP / HV 共享到这一级
        self.up3_np = UpBlock(b * 2, b * 2, b)# 128+128 → 64，NP专用
        self.up3_hv = UpBlock(b * 2, b * 2, b)          # 128+128 → 64，HV专用
        self.up3_nc = UpBlock(b * 2, b * 2, b)          # 128+128 → 64，NC专用

        # NP head
        self.np_head = nn.Sequential(
            CBL(b, b, 3, 1, 1),
            nn.Conv2d(b, 1, 1),
            nn.Sigmoid(),
        )
        # HV head
        self.hv_head = nn.Sequential(
            CBL(b, b, 3, 1, 1),
            nn.Conv2d(b, 2, 1),
            nn.Tanh(),
        )
        # NC head（输出 logits，loss里用 CrossEntropy）
        self.nc_head = nn.Sequential(
            CBL(b, b, 3, 1, 1),
            nn.Conv2d(b, num_classes, 1),   # 不加Softmax，训练时 CE内部处理
        )

    def forward(self, backbone_feats):
        x2, x3, x4 = backbone_feats          # /8-256, /16-512, /32-1024

        d = self.bottleneck(x4)              # (B, 256, H/32, W/32)
        d = self.up1(d, x3)                  # (B, 256, H/16, W/16)
        d2 = self.up2(d, x2)                 # (B, 128, H/8,  W/8)

        # 三个分支独立精炼（共享 d2，各自有独立 up3）
        # up3 的 skip 用d2 自身（相同分辨率做一次自精炼）
        d_np = self.up3_np(d2, d2)           # (B, 64, H/8, W/8)
        d_hv = self.up3_hv(d2, d2)
        d_nc = self.up3_nc(d2, d2)

        #×8 上采样到原图
        np_map = F.interpolate(
            self.np_head(d_np), scale_factor=8, mode='bilinear', align_corners=False
        )   # (B, 1, H, W)

        hv_map = F.interpolate(
            self.hv_head(d_hv), scale_factor=8, mode='bilinear', align_corners=False
        )   # (B, 2, H, W)

        nc_map = F.interpolate(
            self.nc_head(d_nc), scale_factor=8, mode='bilinear', align_corners=False
        )   # (B, num_classes, H, W)  logits

        return d2, np_map, hv_map, nc_map