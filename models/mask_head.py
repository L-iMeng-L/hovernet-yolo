# models/mask_head.py
"""
实例分割头（YOLOv8-seg风格）：
  ProtoNet  : 生成 n_proto 个原型 mask（用hover_feat 增强）
  MaskCoeff : YOLO 检测头附加系数预测分支
推理时: sigmoid(coeff @ proto) → 每个检测框的 instance mask
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import CBL

class ProtoNet(nn.Module):
    """
    在p2_fused 基础上，融合 hover_feat，生成 n_proto 个 prototype mask。
    hover_feat 携带 hv_map 的距离信息，使prototype感知细胞边界。
    输出: (B, n_proto, H/8, W/8)
    """
    def __init__(self, in_p2=128, in_hover=64, n_proto=32):
        super().__init__()
        self.n_proto = n_proto

        # 融合 p2 + hover_feat
        self.fuse = nn.Sequential(
            CBL(in_p2+ in_hover, 128, 3, 1, 1),
            CBL(128, 128, 3, 1, 1),CBL(128, n_proto, 1, 1, 0),
        )

    def forward(self, p2_fused, hover_feat):
        """
        p2_fused  : (B, 128, H/8, W/8)
        hover_feat: (B,  64, H/8, W/8)
        """
        x = torch.cat([p2_fused, hover_feat], dim=1)  # (B, 192, H/8, W/8)
        return self.fuse(x)            # (B, n_proto, H/8, W/8)

class MaskCoeffHead(nn.Module):
    """
    附加在 YOLODetectHead 每个尺度输出上，
    预测每个 anchor 点对应 n_proto 个系数。
    输出: (B, n_proto, H, W)
    """
    def __init__(self, in_chs=(128, 256, 256), n_proto=32):
        super().__init__()
        self.coeff_heads = nn.ModuleList([
            nn.Sequential(
                CBL(in_ch, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, n_proto, 1),# 无激活，推理时用tanh
                nn.Tanh(),
            )
            for in_ch in in_chs
        ])

    def forward(self, neck_feats):
        """返回 list[(B, n_proto, H_i, W_i)] 对应三个尺度"""
        return [head(feat) for head, feat in zip(self.coeff_heads, neck_feats)]