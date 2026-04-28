# models/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ECA(nn.Module):
    """Efficient Channel Attention (轻量级)"""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float)) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
    
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * y.sigmoid()

class SpatialAttention(nn.Module):
    """空间注意力（强化小目标）"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
    
    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial = torch.cat([max_pool, avg_pool], dim=1)
        attention = self.conv(spatial).sigmoid()
        return x * attention

class CBAM_Light(nn.Module):
    """轻量级CBAM（通道+空间注意力）"""
    def __init__(self, channels):
        super().__init__()
        self.channel_att = ECA(channels)
        self.spatial_att = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class MultiScaleFusion(nn.Module):
    """多尺度特征融合（增强小核感知）"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch1 = nn.Conv2d(in_ch, out_ch//3, 1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch//3, 3, padding=1),
            nn.BatchNorm2d(out_ch//3),
            nn.SiLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch//3, 5, padding=2),
            nn.BatchNorm2d(out_ch//3),
            nn.SiLU()
        )
        self.fuse = nn.Conv2d(out_ch, out_ch, 1)
        self.att = CBAM_Light(out_ch)
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        fused = torch.cat([b1, b2, b3], dim=1)
        fused = self.fuse(fused)
        return self.att(fused)