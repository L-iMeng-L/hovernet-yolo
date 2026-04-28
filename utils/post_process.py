# utils/post_process.py
import cv2
import numpy as np
from scipy.ndimage import measurements
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects

# utils/post_process.py

from skimage.morphology import remove_small_holes  # 添加导入

def process_instance(np_map, hv_map, min_area=10, np_thresh=0.6, min_distance=8, peak_thresh=0.4):
    """
    实例分割（可调参数版本）
    
    Args:
        np_map: (H,W) 核概率图
        hv_map: (2,H,W) HV距离图
        min_area: 最小细胞面积（增大减少细胞）
        np_thresh: NP二值化阈值（增大减少细胞）
        min_distance: 种子点最小间距（增大减少细胞）
        peak_thresh: 种子点能量阈值（增大减少细胞）
    """
    # 1. 二值化
    np_binary = (np_map > np_thresh).astype(np.uint8)
    
    # 2. 能量图
    h_dir = hv_map[0]
    v_dir = hv_map[1]
    energy = np.sqrt(h_dir**2 + v_dir**2)
    energy = 1.0 - energy
    
    # 3. 找种子点
    from skimage.feature import peak_local_max
    from skimage.morphology import dilation, disk
    
    energy_smooth = cv2.GaussianBlur(energy, (5, 5), 0)
    
    coordinates = peak_local_max(
        energy_smooth,
        min_distance=min_distance,
        threshold_abs=peak_thresh,
        exclude_border=False
    )
    
    markers = np.zeros_like(np_binary, dtype=np.int32)
    for idx, (y, x) in enumerate(coordinates, start=1):
        if np_binary[y, x] > 0:
            markers[y, x] = idx
    
    markers = dilation(markers, disk(2))
    
    # 4. 分水岭
    inst_map = watershed(-energy, markers, mask=np_binary)
    
    # 5. 移除小对象（修复警告）
    inst_map = remove_small_holes(
        remove_small_holes(inst_map.astype(bool), area_threshold=min_area),
        area_threshold=min_area
    ).astype(np.int32) * inst_map
    inst_map = np.where(
        np.isin(inst_map, [i for i in np.unique(inst_map) if (inst_map == i).sum() >= min_area]),
        inst_map, 0
    )
    
    # 重新编号
    inst_map = measurements.label(inst_map > 0)[0]
    
    return inst_map

def batch_postprocess(np_maps, hv_maps, nc_maps, np_thresh=0.6, energy_thresh=0.4, 
                      min_area=30, min_distance=8, peak_thresh=0.4):
    """批量后处理（可调参数）"""
    B = np_maps.shape[0]
    np_maps = np_maps.cpu().numpy()[:, 0]
    hv_maps = hv_maps.cpu().numpy()
    nc_maps = nc_maps.cpu().numpy()
    
    nc_maps = np.exp(nc_maps) / np.exp(nc_maps).sum(axis=1, keepdims=True)
    
    inst_maps = []
    cls_dicts = []
    
    for b in range(B):
        inst_map = process_instance(
            np_maps[b], hv_maps[b], 
            min_area=min_area,
            np_thresh=np_thresh,
            min_distance=min_distance,
            peak_thresh=peak_thresh
        )
        inst_maps.append(inst_map)
        
        nc_pred = nc_maps[b]
        cls_dict = {}
        for inst_id in np.unique(inst_map):
            if inst_id == 0:
                continue
            mask = inst_map == inst_id
            inst_probs = nc_pred[:, mask].mean(axis=1)
            cls_dict[int(inst_id)] = int(inst_probs.argmax())
        cls_dicts.append(cls_dict)
    
    return inst_maps, cls_dicts

def classify_instances(inst_map, nc_map):
    """
    对每个实例进行分类投票
    
    Args:
        inst_map: (H, W) 实例ID图
        nc_map: (num_classes, H, W) 分类概率图
    
    Returns:
        inst_type_map: (H, W) 每个像素的实例类别
        inst_info: dict, 每个实例的详细信息
    """
    num_classes = nc_map.shape[0]
    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]  # 排除背景
    
    inst_type_map = np.zeros_like(inst_map, dtype=np.int32)
    inst_info = {}
    
    for inst_id in inst_ids:
        # 获取该实例的mask
        inst_mask = (inst_map == inst_id)
        
        # 提取该实例内所有像素的分类概率
        inst_probs = nc_map[:, inst_mask]  # (num_classes, N_pixels)
        
        # 方法1: 平均投票（推荐）
        avg_prob = inst_probs.mean(axis=1)  # (num_classes,)
        pred_type = np.argmax(avg_prob)
        
        # 方法2: 多数投票（备选）
        # pixel_types = np.argmax(inst_probs, axis=0)
        # pred_type = np.bincount(pixel_types).argmax()
        
        # 填充类别
        inst_type_map[inst_mask] = pred_type
        
        # 保存实例信息
        inst_info[int(inst_id)] = {
            'type': int(pred_type),
            'type_prob': avg_prob.tolist(),
            'centroid': measurements.center_of_mass(inst_mask),
            'area': int(inst_mask.sum()),
            'bbox': get_bbox(inst_mask)
        }
    
    return inst_type_map, inst_info

def get_bbox(mask):
    """计算mask的边界框"""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return [0, 0, 0, 0]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax), int(rmax)]

def post_process_hovernet(outputs, min_area=10):
    """
    完整的HoverNet后处理
    
    Args:
        outputs: dict, 包含 'np_map', 'hv_map', 'nc_map'
        min_area: 最小核面积
    
    Returns:
        results: dict, 包含实例分割和分类结果
    """
    np_map = outputs['np_map'].squeeze().cpu().numpy()  # (H, W)
    hv_map = outputs['hv_map'].squeeze().cpu().numpy()  # (2, H, W)
    nc_map = outputs['nc_map'].squeeze().cpu().numpy()  # (C, H, W)
    
    # Softmax归一化分类概率
    nc_map = np.exp(nc_map) / np.exp(nc_map).sum(axis=0, keepdims=True)
    
    # 1. 实例分割
    inst_map = process_instance(np_map, hv_map, min_area)
    
    # 2. 实例分类
    inst_type_map, inst_info = classify_instances(inst_map, nc_map)
    
    return {
        'inst_map': inst_map,
        'inst_type_map': inst_type_map,
        'inst_info': inst_info,
        'np_map': np_map,
        'hv_map': hv_map
    }

