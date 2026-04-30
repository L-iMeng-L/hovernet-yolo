import cv2
import numpy as np
from scipy.ndimage import measurements, binary_fill_holes
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects

def process_instance(np_map, hv_map, min_area=5, np_thresh=0.5, ksize=21, overall_thresh=0.4, marker_ksize=5):
    h_dir_raw = hv_map[0]
    v_dir_raw = hv_map[1]
    
    # 1. 二值化 + 连通域过滤
    blb = (np_map >= np_thresh).astype(np.int32)
    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=min_area)
    blb = (blb > 0).astype(np.int32)
    
    # 2. HV归一化
    h_dir = cv2.normalize(h_dir_raw, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    v_dir = cv2.normalize(v_dir_raw, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    
    # 3. Sobel梯度
    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)
    
    sobelh = 1 - cv2.normalize(sobelh, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    sobelv = 1 - cv2.normalize(sobelv, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    
    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)  # 修正
    overall[overall < 0] = 0
    
    # 4. 距离变换
    dist = (1.0 - overall) * blb
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)
    
    overall = (overall >= overall_thresh).astype(np.int32)  # 修正
    
    # 5. 生成marker
    marker = blb - overall  # 修正
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (marker_ksize, marker_ksize))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=min_area)
    
    # 6. 分水岭
    inst_map = watershed(dist, markers=marker, mask=blb)
    
    return inst_map

def batch_postprocess(np_maps, hv_maps, nc_maps, np_thresh=0.5, min_area=10, 
                      ksize=21, overall_thresh=0.4, marker_ksize=5, **kwargs):
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
            ksize=ksize,
            overall_thresh=overall_thresh,
            marker_ksize=marker_ksize
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
            'type': int(pred_type),
            'type_prob': avg_prob.tolist(),
            'centroid': measurements.center_of_mass(inst_mask),
            'area': int(inst_mask.sum()),
            'bbox': get_bbox(inst_mask)
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
    np_map = outputs['np_map'].squeeze().cpu().numpy()
    hv_map = outputs['hv_map'].squeeze().cpu().numpy()
    nc_map = outputs['nc_map'].squeeze().cpu().numpy()
    
    nc_map = np.exp(nc_map) / np.exp(nc_map).sum(axis=0, keepdims=True)
    
    inst_map = process_instance(np_map, hv_map, min_area)
    inst_type_map, inst_info = classify_instances(inst_map, nc_map)
    
    return {
        'inst_map': inst_map,
        'inst_type_map': inst_type_map,
        'inst_info': inst_info,
        'np_map': np_map,
        'hv_map': hv_map
    }