# models/hover_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import CBAM_Light, MultiScaleFusion


class CBL(nn.Module):
    def __init__(self, c1, c2, k, s, p):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

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

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = CBL(in_ch, out_ch, 1, 1, 0)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
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

class EnhancedHead(nn.Module):
    def __init__(self, in_ch, out_ch, use_attention=True):
        super().__init__()
        self.aspp = ASPP(in_ch, in_ch // 2)
        
        # 添加注意力
        self.use_attention = use_attention
        if use_attention:
            self.attention = CBAM_Light(in_ch // 2)
        
        self.refine = nn.Sequential(
            CBL(in_ch // 2, in_ch // 2, 3, 1, 1),
            CBL(in_ch // 2, in_ch // 2, 3, 1, 1),
        )
        self.out_conv = nn.Conv2d(in_ch // 2, out_ch, 1)
    
    def forward(self, x):
        x = self.aspp(x)
        
        if self.use_attention:
            x = self.attention(x)
        
        x = self.refine(x)
        return self.out_conv(x)

class HoverDecoder(nn.Module):
    def __init__(self, base_ch=64, num_classes=5):
        super().__init__()
        b = base_ch
        self.fpn = FPNFusion(c2=b*4, c3=b*8, c4=b*8, out_ch=b*4)
        
        # NC头使用注意力（重点增强小核分类）
        self.nc_head = nn.Sequential(
            EnhancedHead(b*4, 128, use_attention=True),
            nn.Conv2d(128, num_classes, 1)
        )
        
        # NP和HV头保持原样
        self.np_head = nn.Sequential(
            EnhancedHead(b*4, 64, use_attention=False),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        self.hv_head = nn.Sequential(
            EnhancedHead(b*4, 64, use_attention=False),
            nn.Conv2d(64, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, backbone_feats):
        x2, x3, x4 = backbone_feats
        fused = self.fpn(x2, x3, x4)
        
        np_map = F.interpolate(self.np_head(fused), scale_factor=8, mode='bilinear', align_corners=False)
        hv_map = F.interpolate(self.hv_head(fused), scale_factor=8, mode='bilinear', align_corners=False)
        nc_map = F.interpolate(self.nc_head(fused), scale_factor=8, mode='bilinear', align_corners=False)
        
        return fused, np_map, hv_map, nc_map