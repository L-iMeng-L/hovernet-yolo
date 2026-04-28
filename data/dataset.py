# data/dataset.py
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import albumentations as A

def _random_augment(img, seg_dict):
    """
    几何增强：随机翻转 + 随机旋转90°倍数
    img: (H, W, 3) uint8
    seg_dict : {'np_map': (H,W), 'hv_map': (2,H,W), 'inst_map': (H,W), 'type_map': (H,W)}
    hv_map 的水平分量在翻转/旋转时需要取反，保证方向一致性
    """
    #── 水平翻转 ─────────────────────────────────────────────
    if np.random.rand() > 0.5:
        img = img[:, ::-1, :].copy()
        seg_dict['np_map']= seg_dict['np_map'][:, ::-1].copy()
        seg_dict['inst_map'] = seg_dict['inst_map'][:, ::-1].copy()
        seg_dict['type_map'] = seg_dict['type_map'][:, ::-1].copy()
        hv = seg_dict['hv_map'].copy()          # (2,H,W)
        hv = hv[:, :, ::-1].copy()
        hv[0] = -hv[0]                # 水平分量取反
        seg_dict['hv_map'] = hv

    # ── 垂直翻转 ─────────────────────────────────────────────
    if np.random.rand() > 0.5:
        img = img[::-1, :, :].copy()
        seg_dict['np_map']   = seg_dict['np_map'][::-1, :].copy()
        seg_dict['inst_map'] = seg_dict['inst_map'][::-1, :].copy()
        seg_dict['type_map'] = seg_dict['type_map'][::-1, :].copy()
        hv = seg_dict['hv_map'].copy()
        hv = hv[:, ::-1, :].copy()
        hv[1] = -hv[1]                          # 垂直分量取反
        seg_dict['hv_map'] = hv

    # ── 随机旋转 90°倍数 ────────────────────────────────────
    k = np.random.randint(0, 4)
    if k > 0:
        img = np.rot90(img, k).copy()
        seg_dict['np_map']   = np.rot90(seg_dict['np_map'],   k).copy()
        seg_dict['inst_map'] = np.rot90(seg_dict['inst_map'], k).copy()
        seg_dict['type_map'] = np.rot90(seg_dict['type_map'], k).copy()
        # hv_map: (2,H,W) →转到(H,W,2) 再rot90 再转回
        hv = seg_dict['hv_map'].transpose(1, 2, 0).copy()  # (H,W,2)
        hv = np.rot90(hv, k).copy()
        # 旋转后分量也要跟着变换：
        # rot90 k次: (h,v) → (-v,h) per k=1，依此类推
        for _ in range(k):
            hv = np.stack([-hv[..., 1], hv[..., 0]], axis=-1)
        seg_dict['hv_map'] = hv.transpose(2, 0, 1).copy()  # (2,H,W)

    return img, seg_dict

def _color_jitter(img):
    """亮度 + 对比度 + 饱和度轻微抖动"""
    img = img.astype(np.float32)
    #亮度
    img *= np.random.uniform(0.75, 1.25)
    # 对比度
    mean = img.mean()
    img  = (img - mean) * np.random.uniform(0.85, 1.15) + mean
    img  = np.clip(img, 0, 255).astype(np.uint8)
    # 饱和度（在HSV 空间）
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= np.random.uniform(0.8, 1.2)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

_elastic_transform = A.Compose([
    A.ElasticTransform(alpha=120, sigma=6, p=0.4),
    A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
])

def _apply_elastic(img, np_map, inst_map, type_map):
    """
    弹性/网格形变，inst_map/np_map/type_map 同步变换。
    hv_map 需要重新计算，这里简化：形变后 hv 用旧值（可接受）
    """
    H, W = img.shape[:2]
    # albumentations 需要 mask 是 uint8/float32
    transformed = _elastic_transform(
        image=img,
        masks=[
            np_map.astype(np.float32),
            inst_map.astype(np.float32),
            type_map.astype(np.float32),
        ]
    )
    img2     = transformed['image']
    np_map2  = transformed['masks'][0]
    inst_map2= transformed['masks'][1].astype(np.int32)
    type_map2= transformed['masks'][2].astype(np.int32)
    return img2, np_map2, inst_map2, type_map2

class PanNukeDataset(Dataset):
    def __init__(self, root, img_size=640, num_classes=5,
                 transform=None, is_train=False):          # ← 新增 is_train
        self.root        = root
        self.img_size    = img_size
        self.num_classes = num_classes
        self.transform   = transform
        self.is_train    = is_train                # ← 记录
        self.names = sorted(
            f[:-4] for f in os.listdir(os.path.join(root, 'images'))
            if f.endswith('.png')
        )
    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        # ── 图像 ──────────────────────────────────────────────
        img = cv2.cvtColor(
            cv2.imread(os.path.join(self.root, 'images', name + '.png')),
            cv2.COLOR_BGR2RGB,
        )
        img = cv2.resize(img, (self.img_size, self.img_size))

        # ── Hover GT（npz）—— 先resize，再增强 ───────────────
        seg_npz = np.load(os.path.join(self.root, 'hover', name + '.npz'))

        def _rf(arr, interp=cv2.INTER_LINEAR):
            return cv2.resize(
                arr.astype(np.float32),
                (self.img_size, self.img_size),
                interpolation=interp,
            )

        np_map= np.clip(_rf(seg_npz['np_map']), 0.0, 1.0)       # (H,W)
        hv_map   = np.stack([
            _rf(seg_npz['hv_map'][0]),
            _rf(seg_npz['hv_map'][1]),
        ], axis=0)                # (2,H,W)
        inst_map = _rf(seg_npz['inst_map'],
                       cv2.INTER_NEAREST).astype(np.int32)           # (H,W)
        type_map = _rf(seg_npz['type_map'],
                       cv2.INTER_NEAREST).astype(np.int32)           # (H,W)

        # ── 训练时数据增强 ────────────────────────────────────
        if self.is_train:
            seg_dict = dict(np_map=np_map, hv_map=hv_map,inst_map=inst_map, type_map=type_map)
            img, seg_dict = _random_augment(img, seg_dict)
            np_map   = seg_dict['np_map']
            hv_map   = seg_dict['hv_map']
            inst_map = seg_dict['inst_map']
            type_map = seg_dict['type_map']

            if np.random.rand() > 0.5:
                img = _color_jitter(img)

            if np.random.rand() > 0.6:
                img, np_map, inst_map, type_map = _apply_elastic(
                    img, np_map, inst_map, type_map)
                # 形变后重新裁为标准大小
                img      = cv2.resize(img, (self.img_size, self.img_size))
                np_map   = cv2.resize(np_map.astype(np.float32),
                                      (self.img_size, self.img_size))
                inst_map = cv2.resize(inst_map.astype(np.float32),
                                      (self.img_size, self.img_size),
                                      interpolation=cv2.INTER_NEAREST).astype(np.int32)
                type_map = cv2.resize(type_map.astype(np.float32),
                                      (self.img_size, self.img_size),
                                      interpolation=cv2.INTER_NEAREST).astype(np.int32)

        # ── 图像 → Tensor ─────────────────────────────────────
        img_t = torch.from_numpy(
            np.ascontiguousarray(img).transpose(2, 0, 1)
        ).float() / 255.0
        if self.transform:
            img_t = self.transform(img_t)

        # ── YOLO labels ───────────────────────────────────────
        bboxes, cat_labels = [], []
        label_path = os.path.join(self.root, 'labels', name + '.txt')
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, cx, cy, w, h = map(float, parts)
                    cx *= self.img_size; cy *= self.img_size
                    w*= self.img_size; h  *= self.img_size
                    bboxes.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])    
                    cat_labels.append(int(cls))
        bboxes     = torch.tensor(bboxes,dtype=torch.float32)
        cat_labels = torch.tensor(cat_labels, dtype=torch.long)

        # ── nc_map：type_map 1~5 → 0~4，背景 0 → -1 ──────────
        nc_map = (type_map - 1).astype(np.int64)

        return {
            'img': img_t,
            'bboxes' : bboxes,
            'labels' : cat_labels,
            'hover_gt': {
                'np_map': torch.from_numpy(np_map).unsqueeze(0),# (1,H,W)
                'hv_map'  : torch.from_numpy(hv_map),               # (2,H,W)
                'inst_map': torch.from_numpy(inst_map),             # (H,W)
                'nc_map'  : torch.from_numpy(nc_map),               # (H,W)
            },
        }

def collate_fn(batch):
    imgs = torch.stack([b['img'] for b in batch])
    hover_gts = {
        'np_map'  : torch.stack([b['hover_gt']['np_map']for b in batch]),
        'hv_map'  : torch.stack([b['hover_gt']['hv_map']   for b in batch]),
        'nc_map'  : torch.stack([b['hover_gt']['nc_map']   for b in batch]),
        'inst_map': [b['hover_gt']['inst_map'] for b in batch],
    }
    bboxes = [b['bboxes'] for b in batch]
    labels = [b['labels'] for b in batch]
    return imgs, bboxes, labels, hover_gts

def get_dataloader(roots, batch_size=8, shuffle=True,
                   img_size=640, num_classes=5, num_workers=4,
                   is_train=False):# ← 新增
    if isinstance(roots, str):
        roots = [roots]
    datasets = [
        PanNukeDataset(r, img_size=img_size, num_classes=num_classes,
                       is_train=is_train)            # ← 传入
        for r in roots
    ]
    ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    return DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = shuffle,
        collate_fn  = collate_fn,
        num_workers = num_workers,
        pin_memory  = True,
    )