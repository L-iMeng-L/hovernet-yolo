# models/hover_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import CBL

class HoverNetHead(nn.Module):
    """
    输入: neck_feats[0]  /8,128ch (80×80 when img=640)
    输出:np_map    : (B, 1, H_img, W_img)细胞核前景概率 [0,1]
        hv_map    : (B, 2, H_img, W_img)   水平/垂直距离图[-1,1]
        hover_feat: (B, 64, H_feat, W_feat) 中间特征，供 fusion 注意力用
    """
    def __init__(self, in_ch=128, stride=8):
        super().__init__()
        self.stride = stride
        mid_ch = in_ch // 2  # 64

        # 共享编码（提取语义）
        self.shared = nn.Sequential(
            CBL(in_ch, in_ch,3, 1, 1),
            CBL(in_ch, mid_ch, 3,1, 1),
        )

        # np分支
        self.np_branch = nn.Sequential(
            nn.Conv2d(mid_ch, 1, 1),
            nn.Sigmoid(),
        )

        # hv 分支
        self.hv_branch = nn.Sequential(
            nn.Conv2d(mid_ch, 2, 1),
            nn.Tanh(),
        )

    def forward(self, neck_feats):
        feat = neck_feats[0]               # (B, 128, H, W)
        hover_feat = self.shared(feat)     # (B, 64, H, W)← 暴露给 fusion

        np_map = F.interpolate(
            self.np_branch(hover_feat),
            scale_factor=self.stride, mode='bilinear', align_corners=False
        )
        hv_map = F.interpolate(
            self.hv_branch(hover_feat),
            scale_factor=self.stride, mode='bilinear', align_corners=False
        )
        return {
            'np_map': np_map,
            'hv_map'    : hv_map,
            'hover_feat': hover_feat,   # (B, 64, H, W)不上采样
        }

class HoverAttnFusion(nn.Module):
    """
    将 hover_feat 通过空间注意力注入 p2。

    原理：
      hover_feat 编码了"此处是否是细胞核"和"距核中心的方向"
      → 生成空间注意力权重 → 加权增强 p2 的细胞核区域响应

    in_hover : HoverNetHead.shared 输出通道数 (64)
    in_p2    : PAN p2 通道数 (128)
    """
    def __init__(self, in_hover=64, in_p2=128):
        super().__init__()

        # 将 hover_feat 映射为与 p2 同通道的注意力图
        self.attn = nn.Sequential(
            nn.Conv2d(in_hover, in_p2, 1, bias=False),
            nn.BatchNorm2d(in_p2),
            nn.Sigmoid(),                # 逐通道空间注意力权重 [0,1]
        )

        # 融合后再做一次轻量卷积，对齐分布
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_p2, in_p2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_p2),
            nn.SiLU(),
        )

    def forward(self, p2, hover_feat):
        """
        p2         : (B, 128, H, W)
        hover_feat : (B,  64, H, W)与 p2 同空间尺寸
        """
        attn_weight = self.attn(hover_feat)     # (B, 128, H, W)
        p2_fused = self.fuse_conv(p2 * attn_weight + p2)  # 残差保留原始信息
        return p2_fused