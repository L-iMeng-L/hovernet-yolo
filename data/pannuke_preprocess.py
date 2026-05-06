# data/pannuke_preprocess.py
"""
将PanNuke 原始 npy 预处理为 processed/FoldX/
    images/   *.png
    hover/    *.npz  {
        'np_map':   (256,256)   float32 [0,1]
        'hv_map':   (2,256,256) float32 [-1,1]  ← 官方gen_targets方式
        'inst_map': (256,256)   int32   实例ID图
        'type_map': (256,256)   int32   0=背景,1~5=各类
    }
    labels/   *.txt  YOLO格式
"""
import os
import cv2
import numpy as np

PANNUKE_CLASSES = {0: 'Neoplastic', 1: 'Inflammatory',
                   2: 'Connective', 3: 'Dead', 4: 'Epithelial'}
NUM_CLASSES = 5

def bounding_box(inst_map):
    """返回实例的 bounding box [y1, y2, x1, x2]"""
    rows = np.any(inst_map, axis=1)
    cols = np.any(inst_map, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [rmin, rmax + 1, cmin, cmax + 1]

def build_combined_maps(masks_i):
    """
    masks_i : (256, 256, 6) 前5通道=各类实例图，第6通道=背景(忽略)
    返回:
        combined_inst : (H, W) int32  全局唯一实例ID，背景=0
        type_map      : (H, W) int32  0=背景, 1~5=类别
    """
    H, W = masks_i.shape[:2]
    combined_inst = np.zeros((H, W), dtype=np.int32)
    type_map = np.zeros((H, W), dtype=np.int32)
    global_id = 1  # 全局实例ID从1开始

    for cls_id in range(NUM_CLASSES):
        ch = masks_i[:, :, cls_id].astype(np.int32)
        inst_ids = np.unique(ch)
        inst_ids = inst_ids[inst_ids != 0]

        for inst_id in inst_ids:
            mask = ch == inst_id
            combined_inst[mask] = global_id
            type_map[mask] = cls_id + 1  # 1-indexed
            global_id += 1               # 每个实例递增，保证全局唯一

    return combined_inst, type_map

def inst_map_to_hover(inst_map):
    """
    对齐HoverNet官方 gen_targets 实现
    inst_map : (H, W) int32
    返回:
        np_map : (H, W)   float32 [0,1]
        hv_map : (2,H,W)  float32 [-1,1]
    """
    H, W = inst_map.shape
    np_map = (inst_map > 0).astype(np.float32)

    x_map = np.zeros((H, W), dtype=np.float32)
    y_map = np.zeros((H, W), dtype=np.float32)

    for inst_id in np.unique(inst_map):
        if inst_id == 0:
            continue

        inst_mask = inst_map == inst_id

        # 获取bounding box，在crop内计算（官方做法）
        y1, y2, x1, x2 = bounding_box(inst_mask)
        inst_mask_crop = inst_mask[y1:y2, x1:x2]

        # 生成坐标网格（从1开始）
        inst_x = np.arange(1, inst_mask_crop.shape[1] + 1, dtype=np.float32)
        inst_y = np.arange(1, inst_mask_crop.shape[0] + 1, dtype=np.float32)
        inst_x, inst_y = np.meshgrid(inst_x, inst_y)

        # 只保留实例内的像素
        inst_x[inst_mask_crop == 0] = 0
        inst_y[inst_mask_crop == 0] = 0

        # x方向归一化到 [-1, 1]（官方方式）
        inst_x_vals = inst_x[inst_mask_crop > 0]
        if inst_x_vals.max() != inst_x_vals.min():
            inst_x = inst_x / (inst_x_vals.max() - inst_x_vals.min())
            inst_x[inst_mask_crop > 0] -= inst_x[inst_mask_crop > 0].mean()
        else:
            inst_x[inst_mask_crop > 0] = 0  # 单像素列

        # y方向归一化到 [-1, 1]（官方方式）
        inst_y_vals = inst_y[inst_mask_crop > 0]
        if inst_y_vals.max() != inst_y_vals.min():
            inst_y = inst_y / (inst_y_vals.max() - inst_y_vals.min())
            inst_y[inst_mask_crop > 0] -= inst_y[inst_mask_crop > 0].mean()
        else:
            inst_y[inst_mask_crop > 0] = 0  # 单像素行

        # 写回完整图（注意：inst_mask_crop外的像素已经是0）
        x_map[y1:y2, x1:x2] += inst_x
        y_map[y1:y2, x1:x2] += inst_y

    hv_map = np.stack([x_map, y_map], axis=0)  # (2, H, W)
    return np_map, hv_map

def inst_map_to_bbox(masks_i, img_size=256):
    """
    从各通道分别提取bbox，保证cls_id对应正确
    返回: list of [cls_id, cx_norm, cy_norm, w_norm, h_norm]
    """
    boxes = []
    for cls_id in range(NUM_CLASSES):
        ch = masks_i[:, :, cls_id].astype(np.int32)
        for inst_id in np.unique(ch):
            if inst_id == 0:
                continue
            mask = ch == inst_id
            ys, xs = np.where(mask)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            cx = (x1 + x2) / 2.0 / img_size
            cy = (y1 + y2) / 2.0 / img_size
            w = (x2 - x1) / img_size
            h = (y2 - y1) / img_size
            if w > 0 and h > 0:
                boxes.append([cls_id, cx, cy, w, h])
    return boxes

def process_fold(fold_root, out_root, fold_name):
    img_npy = os.path.join(fold_root, 'images', fold_name, 'images.npy')
    mask_npy = os.path.join(fold_root, 'masks', fold_name, 'masks.npy')

    print(f'[Loading] {fold_name}...')
    images = np.load(img_npy)   # (N, 256, 256, 3)
    masks = np.load(mask_npy)   # (N, 256, 256, 6)  最后通道=背景
    N = images.shape[0]
    print(f'  {N} samples')

    out_img = os.path.join(out_root, fold_name, 'images')
    out_hover = os.path.join(out_root, fold_name, 'hover')
    out_label = os.path.join(out_root, fold_name, 'labels')
    for d in [out_img, out_hover, out_label]:
        os.makedirs(d, exist_ok=True)

    for i in range(N):
        name = f'{fold_name}_{i:05d}'

        # ── 图像 ──────────────────────────────────────────────
        img = images[i].astype(np.uint8)
        cv2.imwrite(
            os.path.join(out_img, name + '.png'),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        )

        # ── 合并实例图 + 生成 type_map ────────────────────────
        combined_inst, type_map = build_combined_maps(masks[i])

        # ── Hover GT（对齐官方gen_targets）───────────────────
        np_map, hv_map = inst_map_to_hover(combined_inst)
        np.savez_compressed(
            os.path.join(out_hover, name + '.npz'),
            np_map=np_map,           # (256,256)   float32
            hv_map=hv_map,           # (2,256,256) float32
            inst_map=combined_inst,  # (256,256)   int32
            type_map=type_map,       # (256,256)   int32
        )

        # ── YOLO labels ───────────────────────────────────────
        boxes = inst_map_to_bbox(masks[i], img_size=256)
        with open(os.path.join(out_label, name + '.txt'), 'w') as f:
            for box in boxes:
                f.write('{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(*box))

        if (i + 1) % 500 == 0:
            print(f'  [{fold_name}] {i+1}/{N}')

    print(f'[Done] {fold_name}: {N} → {out_root}/{fold_name}')

if __name__ == '__main__':
    PANNUKE_ROOT = '/home/lwy/dataset/PanNuke'
    OUT_ROOT = '/home/lwy/dataset/PanNuke/processed'

    for fold in ['Fold1', 'Fold2', 'Fold3']:
        process_fold(os.path.join(PANNUKE_ROOT, fold), OUT_ROOT, fold)