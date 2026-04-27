# models/fusion.py
import torch.nn as nn
from models.backbone import CSPDarknet
from models.neck import PAN
from models.hover_decoder import HoverDecoder
from models.hover_attn import HoverAttnFusion
from models.yolo_head import YOLODetectHead
from models.mask_head import ProtoNet, MaskCoeffHead

class HoverNetYOLO(nn.Module):
    """
    完整数据流:
      img│ CSPDarknet → [x2:/8-256, x3:/16-512, x4:/32-1024]
       │ PAN        → [p2:/8-128, p3:/16-256, p4:/32-256]
       │
       ├─ HoverDecoder(backbone_feats, p2)
       │       → hover_feat (B,64,H/8,W/8)   ← 携带距离图语义
       │       → np_map     (B,1,H,W)
       │       → hv_map     (B,2,H,W)
       │├─ HoverAttnFusion(p2, hover_feat)
       │       → p2_fused   (B,128,H/8,W/8)
       │├─ YOLODetectHead([p2_fused, p3, p4])
       │       → {reg, cls} × 3尺度
       │├─ MaskCoeffHead([p2_fused, p3, p4])
       │       → coeff ×3尺度  (B, n_proto, H_i, W_i)
       │
       └─ ProtoNet(p2_fused, hover_feat)
               → prototypes (B, n_proto, H/8, W/8)
    推理时:
      coeff(scale0) @ prototypes → sigmoid → instance mask
    """
    def __init__(self, num_classes=2, base_ch=64, base_depth=3, n_proto=32):
        super().__init__()
        self.backbone= CSPDarknet(base_ch=base_ch, base_depth=base_depth)
        self.neck         = PAN(in_chs=(base_ch*4, base_ch*8, base_ch*16), out_ch=256)

        self.hover_decoder = HoverDecoder()
        self.attn_fusion   = HoverAttnFusion(in_hover=64, in_p2=128)

        self.yolo_head     = YOLODetectHead(num_classes=num_classes, in_chs=(128, 256, 256))
        self.mask_coeff    = MaskCoeffHead(in_chs=(128, 256, 256), n_proto=n_proto)
        self.proto_net     = ProtoNet(in_p2=128, in_hover=64, n_proto=n_proto)

    def forward(self, x):
        # ── 特征提取 ──────────────────────────────
        backbone_feats = self.backbone(x)           # [x2, x3, x4]
        neck_feats     = self.neck(backbone_feats)  # [p2, p3, p4]
        p2, p3, p4     = neck_feats

        # ── HoverNet Decoder ──────────────────────
        hover_feat, np_map, hv_map = self.hover_decoder(backbone_feats, p2)
        # hover_feat: (B,64,H/8,W/8)  携带距离图多尺度语义

        # ── 注意力融合 p2 ─────────────────────────
        p2_fused = self.attn_fusion(p2, hover_feat) # (B,128,H/8,W/8)
        fused_feats = [p2_fused, p3, p4]

        # ── 检测头 ────────────────────────────────
        yolo_out= self.yolo_head(fused_feats)# [{reg,cls} × 3]

        # ── 实例分割头 ────────────────────────────
        coeffs     = self.mask_coeff(fused_feats)   # [(B,n_proto,H_i,W_i) × 3]
        prototypes = self.proto_net(p2_fused, hover_feat)  # (B,n_proto,H/8,W/8)

        return {
            'yolo': yolo_out,
            'coeffs'    : coeffs,
            'prototypes': prototypes,
            'hover'     : {
                'np_map': np_map,
                'hv_map': hv_map,
            }
        }