# evaluate.py
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
from utils.metrics import batch_seg_metrics, get_fast_pq

CLASS_NAMES = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']

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
    
    # 后处理参数
    p.add_argument('--np_thresh', type=float, default=0.35)
    p.add_argument('--overall_thresh', type=float, default=0.15)
    p.add_argument('--ksize', type=int, default=21)
    p.add_argument('--marker_ksize', type=int, default=5)
    p.add_argument('--min_area', type=int, default=5)
    p.add_argument('--match_iou', type=float, default=0.5)
    
    # 可选：从json加载参数
    p.add_argument('--params_json', default='', help='从json加载后处理参数')
    return p.parse_args()

def _build_true_cls(inst_map_np, nc_map_np):
    cls_dict = {}
    for iid in np.unique(inst_map_np):
        if iid == 0:
            continue
        mask = inst_map_np == iid
        labels = nc_map_np[mask]
        labels = labels[labels >= 0]
        if len(labels) == 0:
            continue
        cls_dict[int(iid)] = int(np.bincount(labels.astype(np.int64)).argmax())
    return cls_dict

def _per_class_pq(pred_inst_list, true_inst_list, pred_cls_list, true_cls_list,
                  num_classes, match_iou=0.5):
    results = {}
    for cls_id in range(num_classes):
        pqs, dqs, sqs = [], [], []
        for pred_inst, true_inst, pred_cls, true_cls in zip(
                pred_inst_list, true_inst_list, pred_cls_list, true_cls_list):

            def _filter(inst_map, cls_dict):
                out = np.zeros_like(inst_map)
                for iid, cid in cls_dict.items():
                    if cid == cls_id:
                        out[inst_map == iid] = iid
                return out

            p = _filter(pred_inst, pred_cls)
            t = _filter(true_inst, true_cls)
            r = get_fast_pq(t, p, match_iou=match_iou)

            if not np.isnan(r['PQ']):
                pqs.append(r['PQ'])
                dqs.append(r['DQ'])
                sqs.append(r['SQ'])

        results[CLASS_NAMES[cls_id]] = {
            'PQ': float(np.mean(pqs)) if pqs else 0.0,
            'DQ': float(np.mean(dqs)) if dqs else 0.0,
            'SQ': float(np.mean(sqs)) if sqs else 0.0,
        }
    return results

@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()

    all_pred_insts, all_true_insts = [], []
    all_pred_cls, all_true_cls = [], []

    pbar = tqdm(loader, desc=f'Eval {args.val_fold}',
                bar_format='{l_bar}{bar:30}{r_bar}')

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

    metrics = batch_seg_metrics(
        all_pred_insts, all_true_insts,
        all_pred_cls, all_true_cls,
        match_iou=args.match_iou,
    )

    per_cls = _per_class_pq(
        all_pred_insts, all_true_insts,
        all_pred_cls, all_true_cls,
        num_classes=args.num_classes,
        match_iou=args.match_iou,
    )
    metrics['per_class'] = per_cls

    _print_metrics(metrics, args.val_fold)

    save_dir = args.save_dir or os.path.dirname(args.ckpt)
    _save_metrics(metrics, save_dir, args.val_fold)

    plot_confusion_matrix(all_pred_cls, all_true_cls, args.num_classes, 
                         save_dir, args.val_fold)

    return metrics

def _print_metrics(metrics, fold_name):
    sep = '=' * 60
    print(f'\n{sep}')
    print(f'Results on {fold_name}')
    print(sep)
    print(f"  PQ        : {metrics['PQ']:.4f}")
    print(f"  DQ        : {metrics['DQ']:.4f}")
    print(f"  SQ        : {metrics['SQ']:.4f}")
    print(f"  F1        : {metrics['F1']:.4f}")
    print(f"  Precision : {metrics['Precision']:.4f}")
    print(f"  Recall    : {metrics['Recall']:.4f}")
    print(f"  Cls Acc   : {metrics['cls_acc']:.4f} (only on matched instances)")
    print(f'\n  Per-class PQ:')
    for cls_name, v in metrics['per_class'].items():
        print(f"    {cls_name:<14}: PQ={v['PQ']:.4f}  DQ={v['DQ']:.4f}  SQ={v['SQ']:.4f}")
    print(sep)

def _save_metrics(metrics, save_dir, fold_name):
    os.makedirs(save_dir, exist_ok=True)

    json_path = os.path.join(save_dir, f'metrics_{fold_name}.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    txt_path = os.path.join(save_dir, f'metrics_{fold_name}.txt')
    with open(txt_path, 'w') as f:
        f.write(f'Results on {fold_name}\n')
        f.write('=' * 60 + '\n')
        for k in ['PQ', 'DQ', 'SQ', 'F1', 'Precision', 'Recall', 'cls_acc']:
            f.write(f'  {k:<12}: {metrics[k]:.4f}\n')
        f.write('\nPer-class PQ:\n')
        for cls_name, v in metrics['per_class'].items():
            f.write(f"  {cls_name:<14}: PQ={v['PQ']:.4f}  DQ={v['DQ']:.4f}  SQ={v['SQ']:.4f}\n")

    print(f'[Saved] {json_path}')
    print(f'[Saved] {txt_path}')

def plot_confusion_matrix(all_pred_cls, all_true_cls, num_classes, save_dir, fold_name):
    all_true_labels = []
    all_pred_labels = []
    
    for pred_cls, true_cls in zip(all_pred_cls, all_true_cls):
        for inst_id in true_cls.keys():
            if inst_id in pred_cls:
                all_true_labels.append(true_cls[inst_id])
                all_pred_labels.append(pred_cls[inst_id])
    
    if len(all_true_labels) == 0:
        print("No matched instances for confusion matrix")
        return
    
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=list(range(num_classes)))
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'confusion_matrix_{fold_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'[Saved] {save_path}')

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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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