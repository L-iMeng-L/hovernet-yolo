# models/seg_model.py
import torch.nn as nn
from models.yolo11_backbone import YOLO11nBackbone
from models.hover_decoder import HoverDecoder

class HoverSegModel(nn.Module):
    def __init__(self, base_ch=64, num_classes=5):
        super().__init__()
        self.backbone = YOLO11nBackbone()
        self.decoder = HoverDecoder(base_ch=base_ch, num_classes=num_classes)
    
    def forward(self, x):
        feats = self.backbone(x)  # [P3(256ch), P4(512ch), P5(512ch)]
        hover_feat, np_map, hv_map, nc_map = self.decoder(feats)
        return {
            'np_map': np_map,
            'hv_map': hv_map,
            'nc_map': nc_map,
            'hover_feat': hover_feat,
        }