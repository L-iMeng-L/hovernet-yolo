# data/pannuke_preprocess.py
"""
PanNuke -> HoVer-Net 官方风格预处理

输出:
    processed/FoldX/
        images/*.png
        hover/*.npz

每个 npz 包含:
    - np_map   : (H, W) uint8, 背景0，核1
    - hv_map   : (H, W, 2) float32, 官方风格 HV 图
    - inst_map : (H, W) int32, 0=背景, 1..N=实例ID
    - type_map : (H, W) int32, 0=背景, 1..5=类别
"""

import os
import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import measurements
from skimage import morphology as morph

PANNUKE_CLASSES = {
    0: 'Neoplastic',
    1: 'Inflammatory',
    2: 'Connective',
    3: 'Dead',
    4: 'Epithelial'
}
NUM_CLASSES = 5

def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def cropping_center(x, crop_shape, batch=False):
    """Crop an input image at the centre."""
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    return x


def fix_mirror_padding(ann):
    """Deal with duplicated instances due to mirroring in interpolation."""
    current_max_id = np.amax(ann)
    inst_list = list(np.unique(ann))
    inst_list.remove(0)
    for inst_id in inst_list:
        inst_map = np.array(ann == inst_id, np.uint8)
        remapped_ids = measurements.label(inst_map)[0]
        remapped_ids[remapped_ids > 1] += current_max_id
        ann[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
        current_max_id = np.amax(ann)
    return ann

def build_inst_and_type_maps(mask_6ch):
    """
    mask_6ch: (H, W, 6)
        前5通道: 类别实例图
        第6通道: 背景/保留通道，忽略
    """
    H, W = mask_6ch.shape[:2]
    inst_map = np.zeros((H, W), dtype=np.int32)
    type_map = np.zeros((H, W), dtype=np.int32)

    global_id = 1
    for cls_idx in range(NUM_CLASSES):
        ch = mask_6ch[:, :, cls_idx].astype(np.int32)
        inst_ids = np.unique(ch)
        inst_ids = inst_ids[inst_ids != 0]

        for inst_id in inst_ids:
            m = (ch == inst_id)
            inst_map[m] = global_id
            type_map[m] = cls_idx + 1
            global_id += 1

    return inst_map, type_map

def gen_instance_hv_map(ann, crop_shape):
    """严格按官方逻辑生成 hv_map。"""
    orig_ann = ann.copy()
    fixed_ann = fix_mirror_padding(orig_ann)
    crop_ann = cropping_center(fixed_ann, crop_shape)
    crop_ann = morph.remove_small_objects(crop_ann, min_size=30)

    x_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    y_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)

    inst_list = list(np.unique(crop_ann))
    if 0 in inst_list:
        inst_list.remove(0)

    for inst_id in inst_list:
        inst_map = np.array(fixed_ann == inst_id, np.uint8)
        inst_box = get_bounding_box(inst_map)

        inst_box[0] -= 2
        inst_box[2] -= 2
        inst_box[1] += 2
        inst_box[3] += 2

        inst_box[0] = max(inst_box[0], 0)
        inst_box[2] = max(inst_box[2], 0)
        inst_box[1] = min(inst_box[1], inst_map.shape[0])
        inst_box[3] = min(inst_box[3], inst_map.shape[1])

        inst_map = inst_map[inst_box[0]:inst_box[1], inst_box[2]:inst_box[3]]

        if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
            continue

        inst_com = list(measurements.center_of_mass(inst_map))
        inst_com[0] = int(inst_com[0] + 0.5)
        inst_com[1] = int(inst_com[1] + 0.5)

        inst_x_range = np.arange(1, inst_map.shape[1] + 1)
        inst_y_range = np.arange(1, inst_map.shape[0] + 1)

        inst_x_range -= inst_com[1]
        inst_y_range -= inst_com[0]

        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

        inst_x[inst_map == 0] = 0
        inst_y[inst_map == 0] = 0
        inst_x = inst_x.astype("float32")
        inst_y = inst_y.astype("float32")

        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])

        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

        x_map_box = x_map[inst_box[0]:inst_box[1], inst_box[2]:inst_box[3]]
        x_map_box[inst_map > 0] = inst_x[inst_map > 0]

        y_map_box = y_map[inst_box[0]:inst_box[1], inst_box[2]:inst_box[3]]
        y_map_box[inst_map > 0] = inst_y[inst_map > 0]

    hv_map = np.dstack([x_map, y_map]).astype(np.float32)
    return hv_map

def gen_targets(ann, crop_shape, **kwargs):
    """Generate the targets for the network."""
    hv_map = gen_instance_hv_map(ann, crop_shape)
    np_map = ann.copy()
    np_map[np_map > 0] = 1

    hv_map = cropping_center(hv_map, crop_shape)
    np_map = cropping_center(np_map, crop_shape)

    return {
        "hv_map": hv_map,
        "np_map": np_map,
    }

def process_fold(fold_root, out_root, fold_name):
    img_npy = os.path.join(fold_root, 'images', fold_name, 'images.npy')
    mask_npy = os.path.join(fold_root, 'masks', fold_name, 'masks.npy')

    print(f'[Loading] {fold_name}')
    images = np.load(img_npy)
    masks = np.load(mask_npy)
    n = images.shape[0]
    print(f'  samples: {n}')

    out_img_dir = os.path.join(out_root, fold_name, 'images')
    out_gt_dir = os.path.join(out_root, fold_name, 'hover')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_gt_dir, exist_ok=True)

    for i in range(n):
        name = f'{fold_name}_{i:05d}'

        img = images[i].astype(np.uint8)
        cv2.imwrite(
            os.path.join(out_img_dir, name + '.png'),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        )

        inst_map, type_map = build_inst_and_type_maps(masks[i])
        targets = gen_targets(inst_map, crop_shape=inst_map.shape[:2])

        np.savez_compressed(
            os.path.join(out_gt_dir, name + '.npz'),
            np_map=targets["np_map"].astype(np.uint8),
            hv_map=targets["hv_map"].astype(np.float32),
            inst_map=inst_map.astype(np.int32),
            type_map=type_map.astype(np.int32),
        )

        if (i + 1) % 500 == 0:
            print(f'  [{fold_name}] {i+1}/{n}')

    print(f'[Done] {fold_name} -> {out_root}/{fold_name}')

if __name__ == '__main__':
    PANNUKE_ROOT = '/home/lwy/dataset/PanNuke'
    OUT_ROOT = '/home/lwy/dataset/PanNuke/processed'

    for fold in ['Fold1', 'Fold2', 'Fold3']:
        process_fold(os.path.join(PANNUKE_ROOT, fold), OUT_ROOT, fold)