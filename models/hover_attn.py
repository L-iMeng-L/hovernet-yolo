# models/hover_attn.py
"""
用hover_feat 生成空间注意力权重，增强 p2。
"""
import torch.nn as nn

class HoverAttnFusion(nn.Module):
    def __init__(self, in_hover=64, in_p2=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_hover, in_p2, 1, bias=False),
            nn.BatchNorm2d(in_p2),
            nn.Sigmoid(),
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_p2, in_p2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_p2),
            nn.SiLU(),
        )

    def forward(self, p2, hover_feat):
        attn_w = self.attn(hover_feat)              # (B, 128, H, W)
        return self.fuse_conv(p2 * attn_w + p2)     # 残差