import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models.seg_model import HoverSegModel
from data.dataset import get_dataloader
from utils.post_process import batch_postprocess
from utils.metrics import batch_binary_metrics, batch_multiclass_metrics, match_instances

CLASS_NAMES = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
CLASS_NAMES_WITH_BG = ['Background'] + CLASS_NAMES

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--val_fold', default='Fold2')
    p.add_argument('--data_root', default='/home/lwy/dataset/PanNuke/processed')
    p.add_argument('--save_dir', default='', help='结果保存目录；默认与ckpt同目录')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--img_size', type=int, default=640)
    p.add_argument('--base_ch', type=int, default=64)
    p.add_argument('--num_classes', type=int, default=5)
    p.add_argument('--num_workers', type=int, default=8)

    p.add_argument('--np_thresh', type=float, default=0.4)
    p.add_argument('--overall_thresh', type=float, default=0.7)
    p.add_argument('--ksize', type=int, default=21)
    p.add_argument('--marker_ksize', type=int, default=7)
    p.add_argument('--min_area', type=int, default=5)
    p.add_argument('--match_iou', type=float, default=0.5)

    p.add_argument('--params_json', default='', help='从json加载后处理参数')
    return p.parse_args()

def _build_true_cls(inst_map_np, nc_map_np):
    """
    从实例图和类别图构建实例ID到类别的映射
    nc_map_np: -1=背景, 0-4=五类细胞
    """
    cls_dict = {}
    for iid in np.unique(inst_map_np):
        if iid == 0:
            continue
        mask = inst_map_np == iid
        labels = nc_map_np[mask]
        labels = labels[labels >= 0]  # 过滤背景(-1)
        if len(labels) == 0:
            continue
        cls_dict[int(iid)] = int(np.bincount(labels.astype(np.int64)).argmax())
    return cls_dict

def _collect_labels_with_bg_for_confusion(all_pred_insts, all_true_insts, all_pred_cls, all_true_cls, match_iou=0.5):
    """
    统计包含背景类的混淆矩阵
    - matched instances: 正常统计类别（1-5）
    - FN (unpaired true): 真值=类别，预测=背景(0)
    - FP (unpaired pred): 真值=背景(0)，预测=类别
    """
    true_labels = []
    pred_labels = []
    BG = 0  # 背景类标签

    for pred_inst, true_inst, pred_cls, true_cls in zip(
        all_pred_insts, all_true_insts, all_pred_cls, all_true_cls
    ):
        _, _, _, _, matched_pairs = match_instances(true_inst, pred_inst, match_iou=match_iou)

        matched_true_ids = set()
        matched_pred_ids = set()

        # 1. Matched instances: 正常统计
        for t_id, p_id, _ in matched_pairs:
            t_lab = true_cls.get(t_id, None)
            p_lab = pred_cls.get(p_id, None)
            if t_lab is None or p_lab is None:
                continue
            true_labels.append(int(t_lab) + 1)  # 1-5
            pred_labels.append(int(p_lab) + 1)  # 1-5
            matched_true_ids.add(t_id)
            matched_pred_ids.add(p_id)

        # 2. FN (False Negative): GT有，但预测漏了 → 预测为背景
        for t_id, t_lab in true_cls.items():
            if t_id not in matched_true_ids:
                true_labels.append(int(t_lab) + 1)  # 真值=类别
                pred_labels.append(BG)               # 预测=背景

        # 3. FP (False Positive): 预测有，但GT没有 → 真值为背景
        for p_id, p_lab in pred_cls.items():
            if p_id not in matched_pred_ids:
                true_labels.append(BG)               # 真值=背景
                pred_labels.append(int(p_lab) + 1)  # 预测=类别

    return true_labels, pred_labels

def _collect_labels_no_bg_for_confusion(all_pred_insts, all_true_insts, all_pred_cls, all_true_cls, match_iou=0.5):
    """
    只统计matched instances的分类准确率
    不包含背景类
    """
    true_labels = []
    pred_labels = []

    for pred_inst, true_inst, pred_cls, true_cls in zip(
        all_pred_insts, all_true_insts, all_pred_cls, all_true_cls
    ):
        _, _, _, _, matched_pairs = match_instances(true_inst, pred_inst, match_iou=match_iou)

        for t_id, p_id, _ in matched_pairs:
            t_lab = true_cls.get(t_id, None)
            p_lab = pred_cls.get(p_id, None)
            if t_lab is None or p_lab is None:
                continue
            true_labels.append(int(t_lab))
            pred_labels.append(int(p_lab))

    return true_labels, pred_labels

def plot_confusion_matrix_with_bg(all_pred_insts, all_true_insts, all_pred_cls, all_true_cls,
                                   num_classes, save_dir, fold_name, match_iou=0.5):
    """绘制包含背景类的混淆矩阵（6×6）"""
    true_labels, pred_labels = _collect_labels_with_bg_for_confusion(
        all_pred_insts, all_true_insts, all_pred_cls, all_true_cls,
        match_iou=match_iou
    )

    if len(true_labels) == 0:
        print("No instances for confusion matrix with background")
        return

    # 标签: 0=背景, 1-5=五类细胞
    labels = list(range(num_classes + 1))  # [0, 1, 2, 3, 4, 5]
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # 绝对数量
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES_WITH_BG, 
                yticklabels=CLASS_NAMES_WITH_BG, ax=ax1)
    ax1.set_title('Confusion Matrix with Background (Counts)')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')

    # 归一化
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES_WITH_BG, 
                yticklabels=CLASS_NAMES_WITH_BG, ax=ax2)
    ax2.set_title('Confusion Matrix with Background (Normalized)')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'confusion_matrix_with_bg_{fold_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'[Saved] {save_path}')

def plot_confusion_matrix_no_bg(all_pred_insts, all_true_insts, all_pred_cls, all_true_cls,
                                 num_classes, save_dir, fold_name, match_iou=0.5):
    """绘制不含背景类的混淆矩阵（5×5，只统计matched instances）"""
    true_labels, pred_labels = _collect_labels_no_bg_for_confusion(
        all_pred_insts, all_true_insts, all_pred_cls, all_true_cls,
        match_iou=match_iou
    )

    if len(true_labels) == 0:
        print("No matched instances for confusion matrix")
        return

    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(num_classes)))
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax1)
    ax1.set_title('Confusion Matrix (Matched Instances Only)')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'confusion_matrix_no_bg_{fold_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'[Saved] {save_path}')

def _fmt(v):
    """格式化输出数值"""
    return "nan" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"

def _print_metrics(metrics, fold_name):
    """打印评估指标"""
    sep = '=' * 80
    print(f'\n{sep}')
    print(f'Results on {fold_name}')
    print(sep)

    print(f"  PQb            : {_fmt(metrics['PQb'])}")
    print(f"  DQb            : {_fmt(metrics['DQb'])}")
    print(f"  SQb            : {_fmt(metrics['SQb'])}")
    print(f"  Fd             : {_fmt(metrics['Fd'])}")
    print(f"  Precision_b    : {_fmt(metrics['Precision_b'])}")
    print(f"  Recall_b       : {_fmt(metrics['Recall_b'])}")

    print(f"  PQM            : {_fmt(metrics['PQM'])}")
    print(f"  DQm            : {_fmt(metrics['DQm'])}")
    print(f"  SQm            : {_fmt(metrics['SQm'])}")
    print(f"  F1m            : {_fmt(metrics['F1m'])}")
    print(f"  Precision_m    : {_fmt(metrics['Precision_m'])}")
    print(f"  Recall_m       : {_fmt(metrics['Recall_m'])}")

    print(f"  cls_acc        : {_fmt(metrics['cls_acc'])}")

    print(f'\n  Per-class metrics:')
    for cls_name, v in metrics['per_class'].items():
        print(
            f"    {cls_name:<14}: "
            f"PQ={_fmt(v['PQ'])}  DQ={_fmt(v['DQ'])}  SQ={_fmt(v['SQ'])}  "
            f"F1={_fmt(v['F1'])}  TP={v['TP']}  FP={v['FP']}  FN={v['FN']}"
        )

    print(sep)

def _save_metrics(metrics, save_dir, fold_name):
    """保存评估指标到文件"""
    os.makedirs(save_dir, exist_ok=True)

    json_path = os.path.join(save_dir, f'metrics_{fold_name}.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    txt_path = os.path.join(save_dir, f'metrics_{fold_name}.txt')
    with open(txt_path, 'w') as f:
        f.write(f'Results on {fold_name}\n')
        f.write('=' * 80 + '\n')
        for k in ['PQb', 'DQb', 'SQb', 'Fd', 'Precision_b', 'Recall_b',
                  'PQM', 'DQm', 'SQm', 'F1m', 'Precision_m', 'Recall_m', 'cls_acc']:
            f.write(f'  {k:<15}: {_fmt(metrics[k])}\n')
        f.write('\nPer-class metrics:\n')
        for cls_name, v in metrics['per_class'].items():
            f.write(
                f"  {cls_name:<14}: PQ={_fmt(v['PQ'])}  DQ={_fmt(v['DQ'])}  "
                f"SQ={_fmt(v['SQ'])}  F1={_fmt(v['F1'])}  "
                f"TP={v['TP']}  FP={v['FP']}  FN={v['FN']}\n"
            )

    print(f'[Saved] {json_path}')
    print(f'[Saved] {txt_path}')

@torch.no_grad()
def evaluate(model, loader, device, args):
    """评估模型性能"""
    model.eval()

    all_pred_insts, all_true_insts = [], []
    all_pred_cls, all_true_cls = [], []

    pbar = tqdm(loader, desc=f'Eval {args.val_fold}', bar_format='{l_bar}{bar:30}{r_bar}')

    for imgs, bboxes, labels, hover_gts in pbar:
        imgs = imgs.to(device)
        out = model(imgs)

        pred_insts, pred_cls_list = batch_postprocess(
            out['np_map'], out['hv_map'], out['nc_map'],
            np_thresh=args.np_thresh,
            overall_thresh=args.overall_thresh,
            ksize=args.ksize,
            marker_ksize=args.marker_ksize,
            min_area=args.min_area
        )

        true_insts = [m.cpu().numpy() for m in hover_gts['inst_map']]
        nc_maps_gt = hover_gts['nc_map'].cpu().numpy()
        true_cls_list = [
            _build_true_cls(true_insts[b], nc_maps_gt[b])
            for b in range(len(true_insts))
        ]

        all_pred_insts.extend(pred_insts)
        all_true_insts.extend(true_insts)
        all_pred_cls.extend(pred_cls_list)
        all_true_cls.extend(true_cls_list)

    # 二值 PQb / Fd
    binary_metrics = batch_binary_metrics(
        all_pred_insts, all_true_insts,
        match_iou=args.match_iou,
        num_workers=args.num_workers
    )

    # 多分类 PQM
    multiclass_metrics = batch_multiclass_metrics(
        all_pred_insts, all_true_insts,
        all_pred_cls, all_true_cls,
        num_classes=args.num_classes,
        match_iou=args.match_iou
    )

    # matched instances 分类准确率（不含背景）
    true_labels_no_bg, pred_labels_no_bg = _collect_labels_no_bg_for_confusion(
        all_pred_insts, all_true_insts,
        all_pred_cls, all_true_cls,
        match_iou=args.match_iou
    )
    
    cls_acc = float((np.array(true_labels_no_bg) == np.array(pred_labels_no_bg)).mean()) if len(true_labels_no_bg) > 0 else np.nan

    metrics = {
        **binary_metrics,
        **{
            k: multiclass_metrics[k]
            for k in ['TPm', 'FPm', 'FNm', 'IoU_sum_m', 'PQM', 'DQm', 'SQm', 'F1m', 'Precision_m', 'Recall_m']
        },
        'cls_acc': cls_acc,
        'per_class': {
            CLASS_NAMES[k]: v for k, v in multiclass_metrics['per_class'].items()
        }
    }

    _print_metrics(metrics, args.val_fold)

    save_dir = args.save_dir or os.path.dirname(args.ckpt)
    _save_metrics(metrics, save_dir, args.val_fold)

    # 生成两种混淆矩阵
    print("\n[Generating confusion matrices...]")
    
    # 1. 不含背景（5×5，只统计matched instances）
    plot_confusion_matrix_no_bg(
        all_pred_insts, all_true_insts,
        all_pred_cls, all_true_cls,
        args.num_classes, save_dir, args.val_fold,
        match_iou=args.match_iou
    )
    
    # 2. 含背景（6×6，包含FP/FN）
    plot_confusion_matrix_with_bg(
        all_pred_insts, all_true_insts,
        all_pred_cls, all_true_cls,
        args.num_classes, save_dir, args.val_fold,
        match_iou=args.match_iou
    )

    return metrics

def main():
    args = get_args()

    if args.params_json and os.path.exists(args.params_json):
        with open(args.params_json) as f:
            params = json.load(f)
            if 'best_params' in params:
                params = params['best_params']
            for k, v in params.items():
                if hasattr(args, k):
                    setattr(args, k, v)
        print(f"[Loaded params from] {args.params_json}")

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    model = HoverSegModel(base_ch=args.base_ch, num_classes=args.num_classes).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    print(f"[Loaded] epoch={ckpt['epoch']}  best_val_loss={ckpt['best_val_loss']:.4f}")

    val_root = os.path.join(args.data_root, args.val_fold)
    val_loader = get_dataloader(
        val_root, batch_size=args.batch_size, shuffle=False,
        img_size=args.img_size, num_classes=args.num_classes,
        num_workers=args.num_workers, is_train=False
    )

    evaluate(model, val_loader, device, args)

if __name__ == '__main__':
    main()