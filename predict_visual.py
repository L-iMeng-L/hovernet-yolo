"""
可视化预测脚本：展示 原图 | GT掩码 | 预测掩码 | Depth Map(HV能量图) | NP Map(核概率图) | Masked Energy Map(NP过滤后)
用法：
  python predict_visual.py \
    --ckpt runs/xxx/best.pth \
    --data_root /home/lwy/dataset/PanNuke/processed \
    --val_fold Fold2 \
    --num_samples 8 \
    --save_dir vis_output
"""

import os
import argparse
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from models.seg_model import HoverSegModel
from data.dataset import get_dataloader
from utils.post_process import batch_postprocess

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── 类别颜色表（BGR→RGB）────────────────────────────────────
CLASS_NAMES  = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
CLASS_COLORS = np.array([
    [255,  64,  64],   # 0 Neoplastic   红
    [ 64, 160, 255],   # 1 Inflammatory 蓝
    [ 64, 220,  64],   # 2 Connective   绿
    [220, 220,  64],   # 3 Dead         黄
    [200,  64, 200],   # 4 Epithelial   紫
], dtype=np.uint8)

def _inst_to_color(inst_map: np.ndarray,
                   cls_dict: dict,
                   alpha: float = 0.6) -> np.ndarray:
    """
    将实例图渲染为彩色掩码。
    inst_map : (H, W) int32
    cls_dict : {inst_id: class_id}
    返回     : (H, W, 3) uint8  RGB
    """
    H, W = inst_map.shape
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    for iid in np.unique(inst_map):
        if iid == 0:
            continue
        mask = inst_map == iid
        cid  = cls_dict.get(int(iid), 0)
        color = CLASS_COLORS[cid % len(CLASS_COLORS)].astype(np.float32)
        canvas[mask] = color
    return canvas.astype(np.uint8)

def _hv_to_depth(hv_map: np.ndarray) -> np.ndarray:
    """
    hv_map : (H, W, 2)  float [-1,1]
    返回   : (H, W, 3) uint8  RGB 伪彩色深度图
    depth 用 |hv| 的幅值，再 colormap 着色
    """
    mag = np.sqrt(hv_map[..., 0]**2 + hv_map[..., 1]**2)   # (H, W)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-7)
    mag_u8 = (mag * 255).astype(np.uint8)
    depth_bgr = cv2.applyColorMap(mag_u8, cv2.COLORMAP_MAGMA)
    return cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)

# NP Map 可视化
def _np_to_color(np_map: np.ndarray) -> np.ndarray:
    """
    np_map : (H, W) float [0,1]
    返回   : (H, W, 3) uint8 RGB 伪彩色概率图
    """
    np_map_u8 = (np_map * 255).astype(np.uint8)
    np_bgr = cv2.applyColorMap(np_map_u8, cv2.COLORMAP_JET) # JET  colormap，蓝→红，直观显示概率
    return cv2.cvtColor(np_bgr, cv2.COLOR_BGR2RGB)

# NP Threshold 过滤后的 Energy Map 可视化

def _hv_to_masked_energy(hv_map: np.ndarray, np_map: np.ndarray, np_thresh: float) -> np.ndarray:
    """
    生成被 NP Threshold 过滤后的 Energy Map
    Args:
        hv_map: (H, W, 2) HV 偏移图
        np_map: (H, W) NP 概率图
        np_thresh: NP 阈值
    Returns:
        (H, W, 3) uint8 RGB 伪彩色图
    """
    # 1. 计算 HV 的幅值（原始 Energy Map）
    mag = np.sqrt(hv_map[..., 0]**2 + hv_map[..., 1]**2)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-7)
    
    # 2. 用 np_thresh 生成前景 mask
    foreground_mask = (np_map > np_thresh).astype(np.float32)
    
    # 3. 把背景区域的 Energy 置 0（黑色）
    masked_mag = mag * foreground_mask
    
    # 4. 转伪彩色
    mag_u8 = (masked_mag * 255).astype(np.uint8)
    energy_bgr = cv2.applyColorMap(mag_u8, cv2.COLORMAP_MAGMA) # 用和 Depth Map 一样的 colormap，方便对比
    return cv2.cvtColor(energy_bgr, cv2.COLOR_BGR2RGB)

def _blend(img_rgb: np.ndarray, mask_rgb: np.ndarray, alpha=0.5) -> np.ndarray:
    """原图与掩码叠加"""
    fg = mask_rgb.sum(axis=-1) > 0
    out = img_rgb.copy()
    out[fg] = (img_rgb[fg] * (1 - alpha) + mask_rgb[fg] * alpha).clip(0, 255).astype(np.uint8)
    return out

def _legend_patches():
    return [
        mpatches.Patch(color=np.array(c)/255., label=n)
        for n, c in zip(CLASS_NAMES, CLASS_COLORS)
    ]

# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_visual(model, loader, device, args):
    model.eval()
    os.makedirs(args.save_dir, exist_ok=True)

    collected = 0
    for batch_idx, (imgs, bboxes, labels, hover_gts) in enumerate(loader):
        if collected >= args.num_samples:
            break

        imgs = imgs.to(device)
        out  = model(imgs)

        # ── 后处理 ────────────────────────────────────────────
        pred_insts, pred_cls_list = batch_postprocess(
        out['np_map'], out['hv_map'], out['nc_map'],
        np_thresh      = args.np_thresh,
        ksize          = args.ksize,           
        overall_thresh = args.overall_thresh,  
        marker_ksize   = args.marker_ksize,    
        min_area       = args.min_area,
        )

        # GT 实例图
        true_insts  = [m.cpu().numpy() for m in hover_gts['inst_map']]
        nc_maps_gt  = hover_gts['nc_map'].cpu().numpy()       # (B,H,W)
        # GT cls dict
        from evaluate import _build_true_cls
        true_cls_list = [
            _build_true_cls(true_insts[b], nc_maps_gt[b])
            for b in range(len(true_insts))
        ]

        # HV map numpy  (B,H,W,2)
        hv_np = out['hv_map'].permute(0, 2, 3, 1).cpu().numpy()
        
        # NP Map
        np_np = out['np_map'].squeeze(1).cpu().numpy() # (B, H, W)，去掉 channel 维度

        B = imgs.shape[0]
        for b in range(B):
            if collected >= args.num_samples:
                break

            # 原图 (H,W,3) uint8
            img_np = (imgs[b].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            H, W   = img_np.shape[:2]

            # Resize inst_map 到原图大小（INTER_NEAREST 保留标签）
            pred_inst = cv2.resize(
                pred_insts[b].astype(np.int32), (W, H),
                interpolation=cv2.INTER_NEAREST)
            true_inst = cv2.resize(
                true_insts[b].astype(np.int32), (W, H),
                interpolation=cv2.INTER_NEAREST)

            # Mask
            pred_mask = _inst_to_color(pred_inst, pred_cls_list[b])
            true_mask = _inst_to_color(true_inst, true_cls_list[b])

            # Depth map
            depth_rgb = _hv_to_depth(hv_np[b])                 # (H,W,3)
            depth_rgb = cv2.resize(depth_rgb, (W, H))
        
            # NP Map 
            np_rgb = _np_to_color(np_np[b])                    # (H,W,3)
            np_rgb = cv2.resize(np_rgb, (W, H))
            
    
            # Masked Energy Map 
            masked_energy_rgb = _hv_to_masked_energy(hv_np[b], np_np[b], args.np_thresh)
            masked_energy_rgb = cv2.resize(masked_energy_rgb, (W, H))

            # 叠加
            pred_blend = _blend(img_np, pred_mask)
            true_blend = _blend(img_np, true_mask)

            # 绘图
  
            fig, axes = plt.subplots(1, 6, figsize=(30, 5))
    
            titles = [
                'Original', 
                'GT Mask', 
                'Pred Mask', 
                'Depth Map (HV)', 
                'NP Map (Prob)',
                f'Masked Energy (np_thresh={args.np_thresh})' # 标题里显示当前阈值
            ]
            panels = [
                img_np, 
                true_blend, 
                pred_blend, 
                depth_rgb, 
                np_rgb,
                masked_energy_rgb # 新增的面板
            ]

            for ax, title, panel in zip(axes, titles, panels):
                ax.imshow(panel)
                ax.set_title(title, fontsize=11) # 稍微调小字体，避免拥挤
                ax.axis('off')

            # 图例放在最后一列下方
            axes[-1].legend(
                handles=_legend_patches(),
                loc='lower right',
                fontsize=6, # 稍微调小字体
                framealpha=0.7,
            )

            plt.suptitle(f'Sample {batch_idx * args.batch_size + b}', fontsize=14)
            plt.tight_layout()

            save_path = os.path.join(
                args.save_dir, f'vis_{batch_idx:03d}_{b:02d}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'[Saved] {save_path}')
            collected += 1

    print(f'\n[Done] {collected} images saved to {args.save_dir}')

# ──────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',          required=True)
    p.add_argument('--data_root',     default='/home/lwy/dataset/PanNuke/processed')
    p.add_argument('--val_fold',      default='Fold2')
    p.add_argument('--save_dir',      default='vis_output')
    p.add_argument('--num_samples',   type=int,   default=4)
    p.add_argument('--batch_size',    type=int,   default=4)
    p.add_argument('--img_size',      type=int,   default=640)
    p.add_argument('--base_ch',       type=int,   default=64)
    p.add_argument('--num_classes',   type=int,   default=5)
    p.add_argument('--num_workers',   type=int,   default=2)
    p.add_argument('--np_thresh',     type=float, default=0.4)
    p.add_argument('--ksize',         type=int,   default=25)
    p.add_argument('--overall_thresh',type=float, default= 0.6)
    p.add_argument('--marker_ksize',  type=int,   default=5)
    p.add_argument('--min_area',      type=int,   default=10)
    return p.parse_args()

def main():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HoverSegModel(base_ch=args.base_ch, num_classes=args.num_classes).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device, weights_only= False)
    model.load_state_dict(ckpt['model_state'])
    print(f"[Loaded] epoch={ckpt.get('epoch', '?')}")

    val_root   = os.path.join(args.data_root, args.val_fold)
    val_loader = get_dataloader(
        val_root,
        batch_size=args.batch_size,
        shuffle=False,
        img_size=args.img_size,
        num_classes=args.num_classes,
        num_workers=args.num_workers,
        is_train=False,
    )

    predict_visual(model, val_loader, device, args)

if __name__ == '__main__':
    main()