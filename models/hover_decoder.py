# models/hover_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FPNFusion(nn.Module):
    def __init__(self, c2=256, c3=512, c4=512, out_ch=256):  
        super().__init__()
        self.c2_conv = nn.Conv2d(c2, out_ch, 1)
        self.c3_conv = nn.Conv2d(c3, out_ch, 1)
        self.c4_conv = nn.Conv2d(c4, out_ch, 1)
        
        self.refine = nn.Sequential(
            CBL(out_ch * 3, out_ch, 3, 1, 1),
            CBL(out_ch, out_ch, 3, 1, 1),
        )
    
    def forward(self, x2, x3, x4):
        c2_aligned = self.c2_conv(x2)
        c3_aligned = F.interpolate(
            self.c3_conv(x3), size=x2.shape[2:], 
            mode='bilinear', align_corners=False
        )
        c4_aligned = F.interpolate(
            self.c4_conv(x4), size=x2.shape[2:], 
            mode='bilinear', align_corners=False
        )
        fused = torch.cat([c2_aligned, c3_aligned, c4_aligned], dim=1)
        return self.refine(fused)

# ── ASPP 模块 ─────────────────────────────────────────────
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = CBL(in_ch, out_ch, 1, 1, 0)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )
        self.fuse = CBL(out_ch * 4, out_ch, 1, 1, 0)
    
    def forward(self, x):
        size = x.shape[2:]
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = F.interpolate(self.pool(x), size=size, mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([feat1, feat2, feat3, feat4], dim=1))

# ── 增强解码头 ────────────────────────────────────────────
class EnhancedHead(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.aspp = ASPP(in_ch, in_ch // 2)
        self.refine = nn.Sequential(
            CBL(in_ch // 2, in_ch // 2, 3, 1, 1),
            CBL(in_ch // 2, in_ch // 2, 3, 1, 1),
        )
        self.out_conv = nn.Conv2d(in_ch // 2, out_ch, 1)
    
    def forward(self, x):
        x = self.aspp(x)
        x = self.refine(x)
        return self.out_conv(x)

# ── 增强解码器 ────────────────────────────────────────────
class HoverDecoder(nn.Module):
    def __init__(self, base_ch=64, num_classes=5):
        super().__init__()
        b = base_ch
        
        # 多尺度融合
        self.fpn = FPNFusion(c2=b*4, c3=b*8, c4=b*16, out_ch=b*4)
        
        # 三个增强解码头
        self.np_head = nn.Sequential(
            EnhancedHead(b*4, 64),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        self.hv_head = nn.Sequential(
            EnhancedHead(b*4, 64),
            nn.Conv2d(64, 2, 1),
            nn.Tanh()
        )
        
        self.nc_head = nn.Sequential(
            EnhancedHead(b*4, 128),
            nn.Conv2d(128, num_classes, 1)
        )
    
    def forward(self, backbone_feats):
        x2, x3, x4 = backbone_feats  # /8-256, /16-512, /32-1024
        
        # 多尺度融合
        fused = self.fpn(x2, x3, x4)  # (B, 256, H/8, W/8)
        
        # 三个分支解码
        np_map = F.interpolate(
            self.np_head(fused), scale_factor=8, 
            mode='bilinear', align_corners=False
        )
        hv_map = F.interpolate(
            self.hv_head(fused), scale_factor=8, 
            mode='bilinear', align_corners=False
        )
        nc_map = F.interpolate(
            self.nc_head(fused), scale_factor=8, 
            mode='bilinear', align_corners=False
        )
        
        return fused, np_map, hv_map, nc_map