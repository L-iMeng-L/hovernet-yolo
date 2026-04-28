# utils/post_process.py
import cv2
import numpy as np
from scipy.ndimage import measurements
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects

def process_instance(np_map, hv_map, min_area=10):
    """
    使用 NP + HV 生成实例分割图
    
    Args:
        np_map: (H, W) 核概率图
        hv_map: (2, H, W) 水平/垂直距离图
        min_area: 最小核面积
    
    Returns:
        inst_map: (H, W) 实例ID图，0为背景
    """
    # 1. 二值化NP
    np_binary = (np_map > 0.5).astype(np.uint8)
    
    # 2. 计算能量图（距离到质心的负值）
    h_dir = hv_map[0]
    v_dir = hv_map[1]
    energy = np.sqrt(h_dir**2 + v_dir**2)
    energy = 1.0 - energy  # 质心处能量最高
    
    # 3. 找种子点（局部极大值）
    from skimage.feature import peak_local_max
    from skimage.morphology import dilation, disk
    
    # 平滑能量图
    energy_smooth = cv2.GaussianBlur(energy, (5, 5), 0)
    
    # 找局部极大值作为种子
    coordinates = peak_local_max(
        energy_smooth,
        min_distance=5,
        threshold_abs=0.3,
        exclude_border=False
    )
    
    # 生成marker
    markers = np.zeros_like(np_binary, dtype=np.int32)
    for idx, (y, x) in enumerate(coordinates, start=1):
        if np_binary[y, x] > 0:  # 只在前景区域
            markers[y, x] = idx
    
    # 膨胀marker避免过分割
    markers = dilation(markers, disk(2))
    
    # 4. 分水岭分割
    inst_map = watershed(-energy, markers, mask=np_binary)
    
    # 5. 移除小对象
    inst_map = remove_small_objects(inst_map, min_size=min_area)
    
    # 重新编号
    inst_map = measurements.label(inst_map > 0)[0]
    
    return inst_map

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