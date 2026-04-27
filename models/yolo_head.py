# models/yolo_head.py
import torch
import torch.nn as nn
from models.backbone import CBL

class YOLODetectHead(nn.Module):
    """
    解耦检测头（Decoupled Head，YOLOv8 风格）
    每个尺度独立卷积，输出格式：
      reg: (B, 4, H, W)—— ltrb 偏移
      cls: (B, num_classes, H, W) ——类别 logits（无sigmoid，损失内处理）
    anchors-free，不再使用预设anchor
    """
    def __init__(self, num_classes, in_chs=(128, 256, 256)):
        super().__init__()
        self.num_classes = num_classes

        self.reg_heads = nn.ModuleList()
        self.cls_heads = nn.ModuleList()
        for in_ch in in_chs:
            self.reg_heads.append(nn.Sequential(
                CBL(in_ch, in_ch, 3, 1, 1),
                CBL(in_ch, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, 4, 1),          # ltrb
            ))
            self.cls_heads.append(nn.Sequential(
                CBL(in_ch, in_ch, 3, 1, 1),
                CBL(in_ch, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, num_classes, 1), # class logits
            ))

    def forward(self, neck_feats):
        """
        Returns:
            list of dict,每个尺度一个 dict:
            {'reg': (B,4,H,W), 'cls': (B,nc,H,W)}
        """
        outputs = []
        for i, feat in enumerate(neck_feats):
            reg = self.reg_heads[i](feat)
            cls = self.cls_heads[i](feat)
            outputs.append({'reg': reg, 'cls': cls})
        return outputs