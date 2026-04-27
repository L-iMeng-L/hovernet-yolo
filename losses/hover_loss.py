# losses/hover_loss.py
import torch.nn.functional as F

def np_loss(pred, gt):
    """核前景 BCE（像素级）"""
    return F.binary_cross_entropy(pred, gt)

def hv_loss(pred, gt, gt_np):
    """
    水平/垂直距离图 MSE，只在核前景区域计算
    gt_np: (B,1,H,W) 前景 mask
    """
    mask = (gt_np > 0.5).float().expand_as(pred)
    return F.mse_loss(pred * mask, gt * mask)

def hover_total_loss(hover_pred, hover_gt, w_np=1.0, w_hv=2.0):
    """
    hover_pred: {'np_map': ..., 'hv_map': ...}
    hover_gt  : {'np_map': ..., 'hv_map': ...}
    """
    loss_np = np_loss(hover_pred['np_map'], hover_gt['np_map'])
    loss_hv = hv_loss(hover_pred['hv_map'], hover_gt['hv_map'], hover_gt['np_map'])
    return w_np * loss_np + w_hv * loss_hv