
# HoverNet-YOLO: 细胞核实例分割与分类

基于 HoverNet 的细胞核实例分割模型，使用 CSPDarknet 作为 Backbone，支持 PanNuke 数据集的 5 类细胞核分类。

---

## 特性

- **三分支输出**
  - `np_map`: 核前景概率图
  - `hv_map`: 水平/垂直距离图（用于 Watershed 分割）
  - `nc_map`: 像素级类别 logits（5 类细胞核分类）

- **轻量高效**
  - Backbone: CSPDarknet（base_ch=64, 参数量 ~15M）
  - Decoder: 三分支独立精炼，共享特征提取
  - 训练速度: ~0.3s/iter (batch_size=8, V100)

- **完整流程**
  - 数据预处理（PanNuke npy → processed）
  - 训练（3-fold 交叉验证）
  - 评估（PQ/DQ/SQ/F1 + 分类准确率）

---

## 模型结构

```
输入 (B, 3, 640, 640)
    ↓
CSPDarknet
├─ /8  → x2 (B, 256,  80, 80)
├─ /16 → x3 (B, 512,  40, 40)
└─ /32 → x4 (B, 1024, 20, 20)
    ↓
HoverDecoder
├─ bottleneck: 1024 → 256
├─ up1 (+x3):  256+512 → 256
├─ up2 (+x2):  256+256 → 128 = d2
└─ 三分支独立精炼:
   ├─ up3_np → np_head → np_map (B, 1, H, W)   [0,1]
   ├─ up3_hv → hv_head → hv_map (B, 2, H, W)   [-1,1]
   └─ up3_nc → nc_head → nc_map (B, 5, H, W)   logits
```

---

## 环境配置

```bash
# Python 3.12
pip install torch torchvision
pip install opencv-python scipy scikit-image tqdm numpy
```

---

## 数据准备

### 1. 下载 PanNuke 数据集

```bash
# 下载地址: https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke
# 解压后目录结构:
PanNuke/
├── Fold1/
│   ├── images/Fold1/images.npy  # (N, 256, 256, 3)
│   └── masks/Fold1/masks.npy    # (N, 256, 256, 6)
├── Fold2/
└── Fold3/
```

### 2. 预处理

```bash
# 修改 data/pannuke_preprocess.py 中的路径
PANNUKE_ROOT = '/path/to/PanNuke'
OUT_ROOT     = '/path/to/PanNuke/processed'

# 运行预处理
python data/pannuke_preprocess.py
```

**输出结构**:
```
processed/
├── Fold1/
│   ├── images/   *.png          # 256×256 RGB 图像
│   ├── hover/    *.npz          # np_map, hv_map, inst_map, type_map
│   └── labels/   *.txt          # YOLO 格式 bbox (cls cx cy w h)
├── Fold2/
└── Fold3/
```

**npz 字段说明**:
- `np_map`: (256, 256) float32, 核前景概率 [0,1]
- `hv_map`: (2, 256, 256) float32, 水平/垂直距离图 [-1,1]
- `inst_map`: (256, 256) int32, 实例 ID 图（背景=0）
- `type_map`: (256, 256) int32, 类别图（0=背景, 1~5=细胞类别）

---

## 训练

### 基础训练

```bash
python train.py \
  --data_root /path/to/PanNuke/processed \
  --val_fold Fold2 \
  --epochs 100 \
  --batch_size 8 \
  --lr 1e-4 \
  --img_size 640 \
  --base_ch 64 \
  --num_classes 5 \
  --save_dir ./runs
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_root` | - | 预处理后的数据根目录 |
| `--val_fold` | `Fold2` | 验证集 fold（Fold1/Fold2/Fold3） |
| `--epochs` | `100` | 训练轮数 |
| `--batch_size` | `8` | 批大小 |
| `--lr` | `1e-4` | 学习率 |
| `--img_size` | `640` | 输入图像尺寸 |
| `--base_ch` | `64` | Backbone 基础通道数 |
| `--num_classes` | `5` | 细胞类别数 |
| `--save_dir` | `./runs` | 模型保存目录 |
| `--resume` | `''` | 恢复训练的 checkpoint 路径 |

### 输出

```
runs/
└── Fold1_Fold3_vs_Fold2/
    ├── best.pt           # 最佳模型（验证集 loss 最低）
    └── epoch_*.pt        # 每 10 轮保存一次
```

### 训练日志示例

```
[050/100] train loss=0.1234 np=0.0456 hv=0.0678 nc=0.0100 | val loss=0.1100 nc=0.0090 iou=0.8523 ← best
```

---

## 评估

```bash
python evaluate.py \
  --ckpt runs/Fold1_Fold3_vs_Fold2/best.pt \
  --val_fold Fold2 \
  --data_root /path/to/PanNuke/processed \
  --batch_size 8 \
  --img_size 640 \
  --np_thresh 0.5 \
  --energy_thresh 0.4 \
  --match_iou 0.5
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ckpt` | - | 模型 checkpoint 路径 |
| `--np_thresh` | `0.5` | 核前景二值化阈值 |
| `--energy_thresh` | `0.4` | Watershed marker 阈值 |
| `--match_iou` | `0.5` | 实例匹配 IoU 阈值 |

```

---

## 推理

```python
import torch
import cv2
import numpy as np
from models.seg_model import HoverSegModel
from utils.post_process import batch_postprocess

# 加载模型
device = torch.device('cuda')
model = HoverSegModel(base_ch=64, num_classes=5).to(device)
ckpt = torch.load('runs/best.pt')
model.load_state_dict(ckpt['model_state'])
model.eval()

# 读取图像
img = cv2.imread('test.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640, 640))
img_t = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
img_t = img_t.to(device)

# 推理
with torch.no_grad():
    out = model(img_t)

# 后处理
inst_maps, inst_classes = batch_postprocess(
    out['np_map'], out['hv_map'], out['nc_map'],
    np_thresh=0.5, energy_thresh=0.4
)

# inst_maps[0]: (H, W) int32, 实例 ID 图
# inst_classes[0]: dict {inst_id: class_id}
print(f"检测到 {len(inst_classes[0])} 个细胞核")
for inst_id, cls_id in inst_classes[0].items():
    print(f"  实例 {inst_id}: 类别 {cls_id}")
```

---

## 文件结构

```
hovernet-yolo/
├── data/
│   ├── pannuke_preprocess.py   # 数据预处理脚本
│   └── dataset.py              # PyTorch Dataset
├── models/
│   ├── backbone.py             # CSPDarknet
│   ├── hover_decoder.py        # HoverDecoder (三分支)
│   └── seg_model.py            # 完整模型
├── losses/
│   └── seg_loss.py             # 分割损失 (BCE + MSE + CE)
├── utils/
│   ├── metrics.py              # 训练指标 (IoU)
│   ├── seg_metrics.py          # 评估指标 (PQ/DQ/SQ)
│   └── post_process.py         # Watershed 后处理
├── train.py                    # 训练脚本
├── evaluate.py                 # 评估脚本
└── README.md
```

---

## 类别定义

PanNuke 数据集包含 5 类细胞核:

| ID | 类别 | 说明 |
|----|------|------|
| 0 | Neoplastic | 肿瘤细胞 |
| 1 | Inflammatory | 炎症细胞 |
| 2 | Connective | 结缔组织细胞 |
| 3 | Dead | 坏死细胞 |
| 4 | Epithelial | 上皮细胞 |

---

## Loss 权重

```python
# losses/seg_loss.py
seg_loss(model_out, hover_gt,
         w_np=1.0,   # 核前景 BCE
         w_hv=2.0,   # 距离图 MSE (前景区域)
         w_nc=1.0)   # 分类 CrossEntropy (ignore_index=-1)
```

可根据验证集表现调整权重，建议:
- `w_hv` 略高（距离图对分割质量影响大）
- `w_nc` 与 `w_np` 相当（分类与检测同等重要）

---

## 后处理参数调优

```python
# utils/post_process.py
hover_postprocess(np_map, hv_map, nc_map,
                  np_thresh=0.5,       # ↑ 减少假阳性，↓ 增加召回
                  energy_thresh=0.4)   # ↑ 合并粘连核，↓ 分离更细
```

**调优建议**:
1. 先在验证集上网格搜索 `np_thresh` (0.3~0.7, step=0.05)
2. 固定 `np_thresh` 后调 `energy_thresh` (0.2~0.6, step=0.05)
3. 以 F1 或 PQ 为目标指标

---
