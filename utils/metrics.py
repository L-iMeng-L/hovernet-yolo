# utils/metrics.py
import torch

@torch.no_grad()
def compute_np_iou(hover_pred, hover_gt, thresh=0.5):
    """HoverNet 核前景像素级IoU（训练快速指标）"""
    pred= (hover_pred['np_map'] > thresh).float()
    gt    = (hover_gt['np_map']   > thresh).float()
    inter = (pred * gt).sum()
    union = (pred + gt).clamp(0, 1).sum()
    return (inter / (union + 1e-7)).item()