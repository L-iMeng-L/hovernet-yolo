# utils/post_process.py
import numpy as np
import cv2
from scipy.ndimage import label as scipy_label

try:
    from skimage.segmentation import watershed
except ImportError:
    def watershed(energy, markers=None, mask=None):
        """
        Fallback watershed implementation using OpenCV when scikit-image is unavailable.
        """
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
    """hv_map: (H, W, 2)  →归一化能量图(H, W)"""
    def _norm_sobel(m):
        sx = cv2.Sobel(m.astype(np.float32), cv2.CV_64F, 1, 0, ksize=21)
        sy = cv2.Sobel(m.astype(np.float32), cv2.CV_64F, 0, 1, ksize=21)
        mag = np.sqrt(sx**2 + sy**2)
        mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-7)
        return mag

    energy = _norm_sobel(hv_map[..., 0]) + _norm_sobel(hv_map[..., 1])
    energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-7)
    return energy

def hover_postprocess(np_map, hv_map, nc_map=None,
                      np_thresh=0.5, energy_thresh=0.4):
    """
    np_map : (H, W) float32 [0,1]
    hv_map : (H, W, 2) float32 [-1,1]
    nc_map : (H, W, num_classes) float32 softmax概率，可为None

    返回:
      inst_map  : (H, W) int32，实例id，背景=0
      inst_class: dict {inst_id: class_id}，每个实例的类别（众数投票）
    """
    fg = (np_map > np_thresh).astype(np.uint8)
    energy = _sobel_energy(hv_map)

    marker_bin = ((energy < energy_thresh) & (fg > 0)).astype(np.uint8)
    markers, _ = scipy_label(marker_bin)
    inst_map = watershed(energy, markers=markers, mask=fg.astype(bool))
    inst_map = inst_map.astype(np.int32)

    # 类别投票
    inst_class = {}
    if nc_map is not None:
        inst_ids = np.unique(inst_map)
        inst_ids = inst_ids[inst_ids > 0]
        for iid in inst_ids:
            mask = inst_map == iid
            # 对该实例区域的 nc_map 求和，取argmax
            cls_votes = nc_map[mask].sum(axis=0)   # (num_classes,)
            inst_class[int(iid)] = int(cls_votes.argmax())

    return inst_map, inst_class

def batch_postprocess(np_maps, hv_maps, nc_maps=None,
                      np_thresh=0.5, energy_thresh=0.4):
    """
    np_maps : (B,1,H,W) Tensor 或 numpy
    hv_maps : (B,2,H,W) Tensor 或 numpy
    nc_maps : (B,num_classes,H,W) Tensor 或 numpy，可为None

    返回:
      inst_maps  : list of (H,W) int32
      inst_classes: list of dict {inst_id: class_id}
    """
    import torch
    if hasattr(np_maps, 'cpu'):
        np_maps_np = np_maps.squeeze(1).cpu().numpy()              # (B,H,W)
        hv_maps_np = hv_maps.permute(0, 2, 3, 1).cpu().numpy()    # (B,H,W,2)
        if nc_maps is not None:
            import torch.nn.functional as F
            nc_prob = F.softmax(nc_maps, dim=1)
            nc_maps_np = nc_prob.permute(0, 2, 3, 1).cpu().numpy()  # (B,H,W,nc)
        else:
            nc_maps_np = [None] * np_maps_np.shape[0]
    else:
        np_maps_np = np_maps.squeeze(1)
        hv_maps_np = np_maps   # 已经是 numpy
        nc_maps_np = nc_maps if nc_maps is not None else [None] * len(np_maps_np)

    inst_maps, inst_classes = [], []
    for i in range(len(np_maps_np)):
        nc = nc_maps_np[i] if nc_maps_np is not None else None
        inst, cls_dict = hover_postprocess(
            np_maps_np[i], hv_maps_np[i], nc,
            np_thresh, energy_thresh
        )
        inst_maps.append(inst)
        inst_classes.append(cls_dict)

    return inst_maps, inst_classes