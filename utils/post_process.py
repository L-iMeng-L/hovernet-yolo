# utils/post_process.py
import cv2
import numpy as np
from scipy.ndimage import measurements, binary_fill_holes
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects

def process_instance(np_map, hv_map, min_area=5, np_thresh=0.5, ksize=21,
                     overall_thresh=0.4, marker_ksize=5):
    """
    Args:
        np_map: (H, W)  核概率图
        hv_map: (2, H, W)  HV 偏移图，其中 hv_map[0] 是水平分量，hv_map[1] 是垂直分量

    Returns:
        inst_map: (H, W) 实例分割结果
    """
    h_dir_raw = hv_map[0]
    v_dir_raw = hv_map[1]

    # 1. 二值化 + 连通域过滤
    blb = (np_map >= np_thresh).astype(np.int32)
    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=min_area)
    blb = (blb > 0).astype(np.int32)

    # 2. HV 归一化到 [0, 1]
    h_dir = cv2.normalize(h_dir_raw, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    v_dir = cv2.normalize(v_dir_raw, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    # 3. Sobel 梯度
    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)

    sobelh = 1 - cv2.normalize(sobelh, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    sobelv = 1 - cv2.normalize(sobelv, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    # 4. 距离图
    dist = (1.0 - overall) * blb
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall = (overall >= overall_thresh).astype(np.int32)

    # 5. 生成 marker
    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (marker_ksize, marker_ksize))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=min_area)

    # 6. 分水岭
    inst_map = watershed(dist, markers=marker, mask=blb, compactness=0.05)

    return inst_map

def batch_postprocess(np_maps, hv_maps, nc_maps,
                      np_thresh=0.5, min_area=10,
                      ksize=21, overall_thresh=0.4, marker_ksize=5, **kwargs):
    """
    批量后处理。

    约定输入 shape：
        np_maps:  Tensor, shape = [B, 1, H, W]
        hv_maps:  Tensor, shape = [B, 2, H, W]
        nc_maps:  Tensor, shape = [B, C, H, W]

    其中 nc_maps 是每个像素的类别 logits，C = num_classes。
    这里会先在 channel 维度上做 softmax，再对每个实例区域做平均投票得到实例类别。

    Returns:
        inst_maps: List[np.ndarray], 每个元素 shape = [H, W]
        cls_dicts: List[dict], 每个元素为 {inst_id: class_id}
    """
    assert np_maps.ndim == 4, f"np_maps should be [B,1,H,W], got shape={tuple(np_maps.shape)}"
    assert hv_maps.ndim == 4, f"hv_maps should be [B,2,H,W], got shape={tuple(hv_maps.shape)}"
    assert nc_maps.ndim == 4, f"nc_maps should be [B,C,H,W], got shape={tuple(nc_maps.shape)}"

    assert np_maps.shape[1] == 1, f"np_maps channel should be 1, got shape={tuple(np_maps.shape)}"
    assert hv_maps.shape[1] == 2, f"hv_maps channel should be 2, got shape={tuple(hv_maps.shape)}"
    assert nc_maps.shape[1] >= 2, f"nc_maps channel should be >=2, got shape={tuple(nc_maps.shape)}"

    B = np_maps.shape[0]

    np_maps = np_maps.detach().cpu().numpy()[:, 0]
    hv_maps = hv_maps.detach().cpu().numpy()
    nc_maps = nc_maps.detach().cpu().numpy()

    # nc_maps: [B, C, H, W] -> softmax over C
    nc_maps = np.exp(nc_maps - np.max(nc_maps, axis=1, keepdims=True))
    nc_maps = nc_maps / (np.sum(nc_maps, axis=1, keepdims=True) + 1e-8)

    inst_maps = []
    cls_dicts = []

    for b in range(B):
        inst_map = process_instance(
            np_maps[b], hv_maps[b],
            min_area=min_area,
            np_thresh=np_thresh,
            ksize=ksize,
            overall_thresh=overall_thresh,
            marker_ksize=marker_ksize
        )
        inst_maps.append(inst_map)

        nc_pred = nc_maps[b]  # [C, H, W]
        cls_dict = {}
        for inst_id in np.unique(inst_map):
            if inst_id == 0:
                continue
            mask = (inst_map == inst_id)
            # nc_pred[:, mask] -> [C, Npix]
            inst_probs = nc_pred[:, mask].mean(axis=1)
            cls_dict[int(inst_id)] = int(inst_probs.argmax())
        cls_dicts.append(cls_dict)

    return inst_maps, cls_dicts

def classify_instances(inst_map, nc_map):
    """
    根据实例 mask 和类别概率图做实例级分类。

    nc_map 约定 shape:
        (C, H, W)

    Returns:
        inst_type_map: (H, W)
        inst_info: dict
    """
    assert nc_map.ndim == 3, f"nc_map should be [C,H,W], got shape={tuple(nc_map.shape)}"
    assert inst_map.ndim == 2, f"inst_map should be [H,W], got shape={tuple(inst_map.shape)}"

    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]

    inst_type_map = np.zeros_like(inst_map, dtype=np.int32)
    inst_info = {}

    for inst_id in inst_ids:
        inst_mask = (inst_map == inst_id)
        inst_probs = nc_map[:, inst_mask]
        avg_prob = inst_probs.mean(axis=1)
        pred_type = np.argmax(avg_prob)

        inst_type_map[inst_mask] = pred_type

        inst_info[int(inst_id)] = {
            "type": int(pred_type),
            "type_prob": avg_prob.tolist(),
            "centroid": measurements.center_of_mass(inst_mask),
            "area": int(inst_mask.sum()),
            "bbox": get_bbox(inst_mask),
        }

    return inst_type_map, inst_info

def get_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return [0, 0, 0, 0]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax), int(rmax)]

def post_process_hovernet(outputs, min_area=10):
    """
    单样本后处理。

    outputs 约定:
        outputs['np_map']: shape = [1, H, W] 或 [H, W]
        outputs['hv_map']: shape = [2, H, W]
        outputs['nc_map']: shape = [C, H, W]

    Returns:
        dict:
            inst_map
            inst_type_map
            inst_info
            np_map
            hv_map
    """
    np_map = outputs["np_map"]
    hv_map = outputs["hv_map"]
    nc_map = outputs["nc_map"]

    if hasattr(np_map, "detach"):
        np_map = np_map.detach().cpu().numpy()
    if hasattr(hv_map, "detach"):
        hv_map = hv_map.detach().cpu().numpy()
    if hasattr(nc_map, "detach"):
        nc_map = nc_map.detach().cpu().numpy()

    # 允许 np_map 为 [1,H,W] 或 [H,W]
    if np_map.ndim == 3:
        np_map = np_map.squeeze(0)

    assert np_map.ndim == 2, f"np_map should be [H,W], got shape={tuple(np_map.shape)}"
    assert hv_map.ndim == 3 and hv_map.shape[0] == 2, f"hv_map should be [2,H,W], got shape={tuple(hv_map.shape)}"
    assert nc_map.ndim == 3, f"nc_map should be [C,H,W], got shape={tuple(nc_map.shape)}"

    inst_map = process_instance(np_map, hv_map, min_area)
    inst_type_map, inst_info = classify_instances(inst_map, nc_map)

    return {
        "inst_map": inst_map,
        "inst_type_map": inst_type_map,
        "inst_info": inst_info,
        "np_map": np_map,
        "hv_map": hv_map,
    }