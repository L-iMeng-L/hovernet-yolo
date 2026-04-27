# losses/yolo_loss.py
import torch
import torch.nn.functional as F
import math

def ciou_loss(pred_xyxy, gt_xyxy):
    """
    pred_xyxy, gt_xyxy: (N, 4)x1y1x2y2，原图像素坐标
    """
    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    gx1, gy1, gx2, gy2 = gt_xyxy.unbind(-1)

    # IoU
    inter_x1 = torch.max(px1, gx1)
    inter_y1 = torch.max(py1, gy1)
    inter_x2 = torch.min(px2, gx2)
    inter_y2 = torch.min(py2, gy2)
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    p_area = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    g_area = (gx2 - gx1).clamp(0) * (gy2 - gy1).clamp(0)
    union  = p_area + g_area - inter + 1e-7
    iou    = inter / union

    # 最小外接矩形对角线距离²
    cw = torch.max(px2, gx2) - torch.min(px1, gx1)
    ch = torch.max(py2, gy2) - torch.min(py1, gy1)
    c2 = cw ** 2 + ch ** 2 + 1e-7

    # 中心点距离²
    pcx = (px1 + px2) / 2
    pcy = (py1 + py2) / 2
    gcx = (gx1 + gx2) / 2
    gcy = (gy1 + gy2) / 2
    rho2 = (pcx - gcx) ** 2 + (pcy - gcy) ** 2

    # 宽高比一致性
    pw = (px2 - px1).clamp(0)
    ph = (py2 - py1).clamp(0)
    gw = (gx2 - gx1).clamp(0)
    gh = (gy2 - gy1).clamp(0)
    v= (4 / math.pi ** 2) * (torch.atan(gw / (gh + 1e-7)) - torch.atan(pw / (ph + 1e-7))) ** 2
    alpha = v / (1 - iou + v + 1e-7)

    return 1 - iou + rho2 / c2 + alpha * v

def yolo_total_loss(yolo_preds, yolo_gts, anchor_pts_per_scale):
    """
    yolo_preds         : list of {'reg':(B,4,H,W) ltrb, 'cls':(B,nc,H,W)}
    yolo_gts: list of {'reg':(B,4,H,W) xyxy target,  'cls':(B,nc,H,W) soft label,
                                  'mask':(B,1,H,W)}
    anchor_pts_per_scale: list of (H*W, 2) anchor point坐标，用于解码 ltrb→xyxy
    """
    total_box= 0.0
    total_cls  = 0.0
    num_scales = len(yolo_preds)

    for pred, gt, anc_pts in zip(yolo_preds, yolo_gts, anchor_pts_per_scale):
        B, _, H, W = pred['reg'].shape
        mask = gt['mask'].squeeze(1).bool()   # (B, H, W)

        # ltrb → xyxy（利用 anchor point 解码）
        # pred['reg']: (B,4,H,W) → (B,H,W,4)
        pred_ltrb = pred['reg'].permute(0, 2, 3, 1)# (B,H,W,4)
        pts = anc_pts.reshape(H, W, 2)                        # (H,W,2)
        cx= pts[..., 0]
        cy  = pts[..., 1]
        pred_xyxy = torch.stack([
            cx - pred_ltrb[..., 0],
            cy - pred_ltrb[..., 1],
            cx + pred_ltrb[..., 2],
            cy + pred_ltrb[..., 3],
        ], dim=-1)   # (B,H,W,4)

        gt_xyxy = gt['reg'].permute(0, 2, 3, 1)# (B,H,W,4)已是 xyxy

        if mask.any():
            total_box += ciou_loss(pred_xyxy[mask], gt_xyxy[mask]).mean()

        total_cls += F.binary_cross_entropy_with_logits(
            pred['cls'], gt['cls']
        )

    return total_box / num_scales + total_cls / num_scales