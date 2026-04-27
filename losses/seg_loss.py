# losses/seg_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Label smoothing 的CrossEntropy（支持 ignore_index）
# torch >= 1.10自带label_smoothing 参数，这里做兼容封装
_nc_criterion = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)

def seg_loss(model_out, hover_gt,
             w_np=1.0, w_hv=2.0, w_nc=0.5):# ← w_nc: 1.0 → 0.5
    np_pred = model_out['np_map']    # (B,1,H,W)
    hv_pred = model_out['hv_map']    # (B,2,H,W)
    nc_pred = model_out['nc_map']    # (B,5,H,W) logits

    np_gt = hover_gt['np_map']       # (B,1,H,W)  float [0,1]
    hv_gt = hover_gt['hv_map']       # (B,2,H,W)  float [-1,1]
    nc_gt = hover_gt['nc_map']       # (B,H,W)    int64, -1=ignore

    # ── np loss（BCE）────────────────────────────────────────
    loss_np = F.binary_cross_entropy(np_pred, np_gt)

    # ── hv loss（前景加权 MSE）───────────────────────────────
    fg_mask = (np_gt > 0.5).float()# (B,1,H,W)
    fg_mask = fg_mask.expand_as(hv_pred)         # (B,2,H,W)
    sq_err  = (hv_pred - hv_gt) ** 2

    weight= fg_mask * 2.0 + (1 - fg_mask) * 0.01
    loss_hv = (sq_err * weight).sum() / (weight.sum() + 1e-7)

    # ── nc loss（CE + label_smoothing=0.1，ignore_index=-1）──
    loss_nc = _nc_criterion(nc_pred, nc_gt)      # ← 带平滑

    loss = w_np * loss_np + w_hv * loss_hv + w_nc * loss_nc
    return loss, {
        'loss_np': loss_np.item(),
        'loss_hv': loss_hv.item(),
        'loss_nc': loss_nc.item(),
    }