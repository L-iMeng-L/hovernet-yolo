# losses/seg_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Focal BCE：解决前景稀疏，抑制假阳性 ──────────────────────
def focal_bce(pred, gt, alpha=0.75, gamma=2.0):
    """
    pred, gt : (B,1,H,W)  float
    alpha    : 前景权重（前景少 → 调高至0.75）
    """
    bce  = F.binary_cross_entropy(pred, gt, reduction='none')
    pt   = torch.where(gt > 0.5, pred, 1 - pred)
    focal_w = alpha * (1 - pt) ** gamma
    return (focal_w * bce).mean()

# ── Dice Loss：直接优化重叠面积，补偿BCE的像素独立性 ─────────
def dice_loss(pred, gt, eps=1e-6):
    """pred, gt : (B,1,H,W) float"""
    pred = pred.view(pred.size(0), -1)
    gt   = gt.view(gt.size(0), -1)
    inter = (pred * gt).sum(dim=1)
    return 1 - (2 * inter + eps) / (pred.sum(dim=1) + gt.sum(dim=1) + eps)

# ── 类别加权 CE：提升少数类（Dead/Inflammatory）────────────────
# 权重与类别频率成反比，Dead×4
_CLASS_WEIGHTS = torch.tensor([1.0, 2.0, 1.5, 4.0, 1.5])

_nc_criterion_smooth = nn.CrossEntropyLoss(
    ignore_index=-1,
    label_smoothing=0.1,
)

def _get_nc_criterion(device):
    """每次动态生成，保证 weight 在正确 device"""
    return nn.CrossEntropyLoss(
        weight=_CLASS_WEIGHTS.to(device),
        ignore_index=-1,
        label_smoothing=0.1,
    )

# ── 主损失函数 ────────────────────────────────────────────────
def seg_loss(model_out, hover_gt,
             w_np=1.5, w_hv=2.0, w_nc=1.0):
    """
    改动点：
      1. NP: BCE → Focal-BCE + Dice  提升DQ（减少漏检/假阳性）
      2. HV: 前景权重 2.0→5.0，强化细胞边界梯度
      3. NC: 权重 0.5→1.0，加类别frequency权重  提升SQ+cls_acc
    """
    np_pred = model_out['np_map']   # (B,1,H,W)
    hv_pred = model_out['hv_map']   # (B,2,H,W)
    nc_pred = model_out['nc_map']   # (B,5,H,W)

    np_gt = hover_gt['np_map']      # (B,1,H,W)
    hv_gt = hover_gt['hv_map']      # (B,2,H,W)
    nc_gt = hover_gt['nc_map']      # (B,H,W)  int64

    device = np_pred.device

    # ── NP loss：Focal-BCE + Dice ────────────────────────────
    loss_focal = focal_bce(np_pred, np_gt, alpha=0.75, gamma=2.0)
    loss_dice  = dice_loss(np_pred, np_gt).mean()
    loss_np    = loss_focal + loss_dice

    # ── HV loss：前景强权重 MSE ──────────────────────────────
    fg_mask = (np_gt > 0.5).float().expand_as(hv_pred)
    sq_err  = (hv_pred - hv_gt) ** 2
    # 前景5.0，背景0.01（原来2.0/0.01）
    weight  = fg_mask * 5.0 + (1 - fg_mask) * 0.01
    loss_hv = (sq_err * weight).sum() / (weight.sum() + 1e-7)

    # ── NC loss：类别加权 CE + label_smoothing ───────────────
    nc_criterion = _get_nc_criterion(device)
    loss_nc = nc_criterion(nc_pred, nc_gt)

    loss = w_np * loss_np + w_hv * loss_hv + w_nc * loss_nc
    return loss, {
        'loss_np':    loss_np.item(),
        'loss_focal': loss_focal.item(),
        'loss_dice':  loss_dice.item(),
        'loss_hv':    loss_hv.item(),
        'loss_nc':    loss_nc.item(),
    }