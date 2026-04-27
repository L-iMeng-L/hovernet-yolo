# data/dataset.py
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class PanNukeDataset(Dataset):
    def __init__(self, root, img_size=640, num_classes=5, transform=None):
        self.root        = root
        self.img_size    = img_size
        self.num_classes = num_classes
        self.transform   = transform
        self.names = sorted(
            f[:-4] for f in os.listdir(os.path.join(root, 'images'))if f.endswith('.png')
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
                    w  *= self.img_size; h  *= self.img_size
                    bboxes.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
                    cat_labels.append(int(cls))
        bboxes     = torch.tensor(bboxes,     dtype=torch.float32)
        cat_labels = torch.tensor(cat_labels, dtype=torch.long)

        # ── Hover GT（npz）────────────────────────────────────
        seg = np.load(os.path.join(self.root, 'hover', name + '.npz'))

        def _rf(arr, interp=cv2.INTER_LINEAR):
            return cv2.resize(
                arr.astype(np.float32),
                (self.img_size, self.img_size),
                interpolation=interp,
            )

        np_map = np.clip(_rf(seg['np_map']), 0.0, 1.0)# (H,W)
        hv_map = np.stack([
            _rf(seg['hv_map'][0]),
            _rf(seg['hv_map'][1]),
        ], axis=0)                                                # (2,H,W)

        # inst_map：最近邻插值保留实例ID整数值
        inst_map = _rf(seg['inst_map'], cv2.INTER_NEAREST).astype(np.int32)

        # type_map → nc_map：1~5 → 0~4，背景0→ -1（ignore_index）
        type_map = _rf(seg['type_map'], cv2.INTER_NEAREST).astype(np.int32)
        nc_map   = (type_map - 1).astype(np.int64)   # 背景: 0-1=-1 ✓, 类别: 1~5-1=0~4 ✓

        return {
            'img': img_t,
            'bboxes' : bboxes,
            'labels' : cat_labels,
            'hover_gt': {
                'np_map'  : torch.from_numpy(np_map).unsqueeze(0),   # (1,H,W)
                'hv_map'  : torch.from_numpy(hv_map),# (2,H,W)
                'inst_map': torch.from_numpy(inst_map),              # (H,W) int32
                'nc_map'  : torch.from_numpy(nc_map),                # (H,W) int64
            },
        }

def collate_fn(batch):
    imgs = torch.stack([b['img'] for b in batch])
    hover_gts = {
        'np_map'  : torch.stack([b['hover_gt']['np_map']for b in batch]),  # (B,1,H,W)
        'hv_map'  : torch.stack([b['hover_gt']['hv_map']   for b in batch]),  # (B,2,H,W)
        'nc_map'  : torch.stack([b['hover_gt']['nc_map']   for b in batch]),  # (B,H,W)
        'inst_map': [b['hover_gt']['inst_map'] for b in batch],               # list，不stack
    }
    bboxes = [b['bboxes'] for b in batch]
    labels = [b['labels'] for b in batch]
    return imgs, bboxes, labels, hover_gts

def get_dataloader(roots, batch_size=8, shuffle=True,
                   img_size=640, num_classes=5, num_workers=4):
    if isinstance(roots, str):
        roots = [roots]
    datasets = [
        PanNukeDataset(r, img_size=img_size, num_classes=num_classes)
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