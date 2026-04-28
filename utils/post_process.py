# utils/post_process.py
import numpy as np
import cv2
from scipy.ndimage import label as scipy_label

try:
    from skimage.segmentation import watershed
except ImportError:
    def watershed(energy, markers=None, mask=None):
        energy_img = np.uint8(
            (energy - energy.min()) / (energy.max() - energy.min() + 1e-7) * 255
        )
        image = cv2.cvtColor(energy_img, cv2.COLOR_GRAY2BGR)
        markers = markers.astype(np.int32) if markers is not None else None
        if mask is not None:
            markers = markers.copy()
            markers[~mask] = 0
        labels = cv2.watershed(image, markers)
        labels[labels == -1] = 0
        return labels

def _sobel_energy(hv_map):
    """
    hv_map: (H, W, 2)
    改动：ksize 21→7，减少边界模糊；加高斯平滑抑制噪声
    """
    def _norm_sobel(m):
        m_blur = cv2.GaussianBlur(m.astype(np.float32), (5, 5), 0)
        sx = cv2.Sobel(m_blur, cv2.CV_64F, 1, 0, ksize=7)   # ← 21→7
        sy = cv2.Sobel(m_blur, cv2.CV_64F, 0, 1, ksize=7)
        mag = np.sqrt(sx**2 + sy**2)
        mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-7)
        return mag

    energy = _norm_sobel(hv_map[..., 0]) + _norm_sobel(hv_map[..., 1])
    energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-7)
    return energy

def _remove_small_instances(inst_map, min_pixels=30):
    """
    过滤面积过小的实例（假阳性噪声）
    min_pixels: 低于此面积的实例置为背景
    """
    counts = np.bincount(inst_map.ravel())   # index=inst_id
    for iid, cnt in enumerate(counts):
        if iid == 0:
            continue
        if cnt < min_pixels:
            inst_map[inst_map == iid] = 0
    return inst_map

def hover_postprocess(np_map, hv_map, nc_map=None,
                      np_thresh=0.5, energy_thresh=0.4,
                      min_pixels=30):
    """
    改动：
      1. Sobel ksize 21→7，加高斯预平滑
      2. 自适应 energy_thresh（Otsu in 前景区）
      3. 过滤小实例（min_pixels）
    """
    fg     = (np_map > np_thresh).astype(np.uint8)
    energy = _sobel_energy(hv_map)

    # ── 自适应阈值：Otsu 在前景区内计算 ─────────────────────
    if fg.sum() > 100:
        fg_energy = (energy[fg > 0] * 255).astype(np.uint8)
        otsu_val, _ = cv2.threshold(
            fg_energy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive_thresh = float(otsu_val) / 255.0
        # 限制在合理范围内，避免极端值
        energy_thresh = float(np.clip(adaptive_thresh, 0.25, 0.60))

    marker_bin = ((energy < energy_thresh) & (fg > 0)).astype(np.uint8)
    markers, _ = scipy_label(marker_bin)
    inst_map   = watershed(energy, markers=markers, mask=fg.astype(bool))
    inst_map   = inst_map.astype(np.int32)

    # ── 去除小实例 ────────────────────────────────────────────
    inst_map = _remove_small_instances(inst_map, min_pixels=min_pixels)

    # ── 类别投票 ──────────────────────────────────────────────
    inst_class = {}
    if nc_map is not None:
        for iid in np.unique(inst_map):
            if iid == 0:
                continue
            mask = inst_map == iid
            cls_votes = nc_map[mask].sum(axis=0)
            inst_class[int(iid)] = int(cls_votes.argmax())

    return inst_map, inst_class

def batch_postprocess(np_maps, hv_maps, nc_maps=None,
                      np_thresh=0.5, energy_thresh=0.4,
                      min_pixels=30):
    import torch
    if hasattr(np_maps, 'cpu'):
        np_maps_np = np_maps.squeeze(1).cpu().numpy()
        hv_maps_np = hv_maps.permute(0, 2, 3, 1).cpu().numpy()
        if nc_maps is not None:
            import torch.nn.functional as F
            nc_prob    = F.softmax(nc_maps, dim=1)
            nc_maps_np = nc_prob.permute(0, 2, 3, 1).cpu().numpy()
        else:
            nc_maps_np = [None] * np_maps_np.shape[0]
    else:
        np_maps_np = np_maps.squeeze(1)
        hv_maps_np = hv_maps
        nc_maps_np = nc_maps if nc_maps is not None else [None] * len(np_maps_np)

    inst_maps, inst_classes = [], []
    for i in range(len(np_maps_np)):
        nc = nc_maps_np[i] if nc_maps_np is not None else None
        inst, cls_dict = hover_postprocess(
            np_maps_np[i], hv_maps_np[i], nc,
            np_thresh, energy_thresh, min_pixels,
        )
        inst_maps.append(inst)
        inst_classes.append(cls_dict)

    return inst_maps, inst_classes