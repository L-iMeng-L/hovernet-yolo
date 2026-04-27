# models/seg_model.py
import torch.nn as nn
from models.backbone import CSPDarknet
from models.hover_decoder import HoverDecoder

class HoverSegModel(nn.Module):
    def __init__(self, base_ch=64, base_depth=3, num_classes=5):
        super().__init__()
        self.backbone = CSPDarknet(base_ch=base_ch, base_depth=base_depth)
        self.decoder = HoverDecoder(base_ch=base_ch, num_classes=num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        hover_feat, np_map, hv_map, nc_map = self.decoder(feats)
        return {
            'np_map': np_map,       # (B, 1, H, W)
            'hv_map'    : hv_map,       # (B, 2, H, W)
            'nc_map'    : nc_map,       # (B, num_classes, H, W)  logits
            'hover_feat': hover_feat,   # (B,128, H/8, W/8)
        }