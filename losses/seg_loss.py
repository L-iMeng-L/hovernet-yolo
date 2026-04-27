# losses/seg_loss.py
import torch
import torch.nn.functional as F

def seg_loss(model_out, hover_gt,
             w_np=1.0, w_hv=2.0, w_nc=1.0,
             num_classes=5):
    """
    model_out: {'np_map':(B,1,H,W), 'hv_map':(B,2,H,W), 'nc_map':(B,nc,H,W) logits}
    hover_gt : {'np_map':(B,1,H,W), 'hv_map':(B,2,H,W), 'nc_map':(B,H,W) int64}

    nc_map GT 约定：
      -1  = 背景（ignore_index，不参与 nc loss）
      0~4 = 5类细胞"""
    pred_np = model_out['np_map']    # [0,1]
    pred_hv = model_out['hv_map']    # [-1,1]
    pred_nc = model_out['nc_map']    # logits (B, nc, H, W)

    gt_np   = hover_gt['np_map']
    gt_hv   = hover_gt['hv_map']
    gt_nc   = hover_gt['nc_map']     # (B, H, W) int64, -1=ignore

    # NP loss：BCE
    loss_np = F.binary_cross_entropy(pred_np, gt_np)

    # HV loss：前景 MSE
    fg = (gt_np > 0.5).float().expand_as(pred_hv)
    loss_hv = F.mse_loss(pred_hv * fg, gt_hv * fg)

    # NC loss：像素级 CrossEntropy，忽略背景（ignore_index=-1）
    # pred_nc: (B, nc, H, W)；gt_nc: (B, H, W)
    loss_nc = F.cross_entropy(pred_nc, gt_nc, ignore_index=-1)

    total = w_np * loss_np + w_hv * loss_hv + w_nc * loss_nc

    details = {
        'loss_np': loss_np.item(),
        'loss_hv'   : loss_hv.item(),
        'loss_nc'   : loss_nc.item(),
        'loss_total': total.item(),
    }
    return total, details