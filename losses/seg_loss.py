# losses/seg_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def focal_bce(pred, gt, alpha=0.75, gamma=2.0):
    bce     = F.binary_cross_entropy(pred, gt, reduction='none')
    pt      = torch.where(gt > 0.5, pred, 1 - pred)
    focal_w = alpha * (1 - pt) ** gamma
    return (focal_w * bce).mean()

def dice_loss(pred, gt, eps=1e-6):
    pred  = pred.view(pred.size(0), -1)
    gt    = gt.view(gt.size(0), -1)
    inter = (pred * gt).sum(dim=1)
    return 1 - (2 * inter + eps) / (pred.sum(dim=1) + gt.sum(dim=1) + eps)

_CLASS_WEIGHTS = torch.tensor([1.0, 2.0, 1.5, 4.0, 1.5])

def _get_nc_criterion(device):
    return nn.CrossEntropyLoss(
        weight=_CLASS_WEIGHTS.to(device),
        ignore_index=-1,
        label_smoothing=0.1,
    )

def _hv_loss_robust(hv_pred, hv_gt, np_gt):
    """
    激进修复：
      1. 背景也用 MSE（权重0.5），保证梯度流通
      2. Cosine 方向损失权重加大到 1.0
      3. 加 L1 正则防止输出全0
    """
    fg = (np_gt > 0.5).float().expand_as(hv_pred)
    bg = 1.0 - fg

    sq_err = (hv_pred - hv_gt) ** 2

    # 前景 MSE 权重 4.0，背景 MSE 权重 0.5（不再用 L1）
    loss_fg = (sq_err * fg * 4.0).sum() / (fg.sum() + 1e-7)
    loss_bg = (sq_err * bg * 0.5).sum() / (bg.sum() + 1e-7)

    # Cosine 方向损失（前景）
    eps = 1e-8
    pred_norm = hv_pred / (hv_pred.norm(dim=1, keepdim=True) + eps)
    gt_norm   = hv_gt   / (hv_gt.norm(dim=1, keepdim=True)   + eps)
    cos_sim   = (pred_norm * gt_norm).sum(dim=1, keepdim=True)
    fg1       = (np_gt > 0.5).float()
    loss_dir  = ((1 - cos_sim) * fg1).sum() / (fg1.sum() + 1e-7)

    # L1 正则：防止 HV 头输出全0（鼓励非零输出）
    loss_reg = -hv_pred.abs().mean() * 0.01

    total = loss_fg + loss_bg + 1.0 * loss_dir + loss_reg
    return total, {
        'hv_fg':  loss_fg.item(),
        'hv_bg':  loss_bg.item(),
        'hv_dir': loss_dir.item(),
        'hv_reg': loss_reg.item(),
    }

def seg_loss(model_out, hover_gt,
             w_np=1.0, w_hv=4.0, w_nc=1.0):
    """
    权重调整：
      w_np: 1.5 → 1.0  （降低 NP 梯度主导）
      w_hv: 3.0 → 4.0  （提升 HV 梯度）
    """
    np_pred = model_out['np_map']
    hv_pred = model_out['hv_map']
    nc_pred = model_out['nc_map']

    np_gt = hover_gt['np_map']
    hv_gt = hover_gt['hv_map']
    nc_gt = hover_gt['nc_map']

    device = np_pred.device

    # NP
    loss_focal = focal_bce(np_pred, np_gt, alpha=0.75, gamma=2.0)
    loss_dice  = dice_loss(np_pred, np_gt).mean()
    loss_np    = loss_focal + loss_dice

    # HV（激进修复）
    loss_hv, hv_details = _hv_loss_robust(hv_pred, hv_gt, np_gt)

    # NC
    loss_nc = _get_nc_criterion(device)(nc_pred, nc_gt)

    loss = w_np * loss_np + w_hv * loss_hv + w_nc * loss_nc
    return loss, {
        'loss_np':    loss_np.item(),
        'loss_focal': loss_focal.item(),
        'loss_dice':  loss_dice.item(),
        'loss_hv':    loss_hv.item(),
        'loss_nc':    loss_nc.item(),
        **hv_details,
    }