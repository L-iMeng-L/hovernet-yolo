# evaluate.py
import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm

from models.seg_model import HoverSegModel
from data.dataset import get_dataloader
from utils.post_process import batch_postprocess
from utils.seg_metrics import batch_seg_metrics, get_fast_pq

CLASS_NAMES = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']

#──────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',required=True)
    p.add_argument('--val_fold',       default='Fold2')
    p.add_argument('--data_root',      default='/home/lwy/dataset/PanNuke/processed')
    p.add_argument('--save_dir',       default='',
                   help='结果保存目录；默认与ckpt 同目录')
    p.add_argument('--batch_size',     type=int,   default=8)
    p.add_argument('--img_size',       type=int,   default=640)
    p.add_argument('--base_ch',        type=int,   default=64)
    p.add_argument('--num_classes',    type=int,   default=5)
    p.add_argument('--num_workers',    type=int,   default=4)
    p.add_argument('--np_thresh',      type=float, default=0.5)
    p.add_argument('--energy_thresh',  type=float, default=0.4)
    p.add_argument('--match_iou',      type=float, default=0.5)
    return p.parse_args()

# ──────────────────────────────────────────────────────────────
def _build_true_cls(inst_map_np, nc_map_np):
    """
    inst_map_np : (H,W) int32
    nc_map_np   : (H,W) int64, -1=背景
    返回 dict {inst_id: class_id (0-indexed)}
    """
    cls_dict = {}
    for iid in np.unique(inst_map_np):
        if iid == 0:
            continue
        mask= inst_map_np == iid
        labels = nc_map_np[mask]
        labels = labels[labels >= 0]
        if len(labels) == 0:
            continue
        cls_dict[int(iid)] = int(np.bincount(labels.astype(np.int64)).argmax())
    return cls_dict

def _per_class_pq(pred_inst_list, true_inst_list,pred_cls_list, true_cls_list,
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

            # ── Bug2 修复：NaN（两边都空）跳过，不计入均值 ──────
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

# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, args):
    """
    返回 metrics dict，同时打印并保存结果。
    可被外部脚本 import 调用。
    """
    model.eval()

    all_pred_insts, all_true_insts = [], []
    all_pred_cls,   all_true_cls   = [], []

    pbar = tqdm(loader, desc=f'Eval {args.val_fold}',
                bar_format='{l_bar}{bar:30}{r_bar}')

    for imgs, bboxes, labels, hover_gts in pbar:
        imgs = imgs.to(device)
        out= model(imgs)

        pred_insts, pred_cls_list = batch_postprocess(
            out['np_map'], out['hv_map'], out['nc_map'],
            np_thresh=args.np_thresh,
            energy_thresh=args.energy_thresh,
        )

        true_insts= [m.cpu().numpy() for m in hover_gts['inst_map']]
        nc_maps_gt    = hover_gts['nc_map'].cpu().numpy()   # (B,H,W)
        true_cls_list = [
            _build_true_cls(true_insts[b], nc_maps_gt[b])
            for b in range(len(true_insts))
        ]

        all_pred_insts.extend(pred_insts)
        all_true_insts.extend(true_insts)
        all_pred_cls.extend(pred_cls_list)
        all_true_cls.extend(true_cls_list)

    #── 整体指标 ───────────────────────────────────────────────
    metrics = batch_seg_metrics(
        all_pred_insts, all_true_insts,
        all_pred_cls,   all_true_cls,
        match_iou=args.match_iou,
    )

    # ── Per-class PQ ───────────────────────────────────────────
    per_cls = _per_class_pq(
        all_pred_insts, all_true_insts,
        all_pred_cls,   all_true_cls,
        num_classes=args.num_classes,
        match_iou=args.match_iou,
    )
    metrics['per_class'] = per_cls

    # ── 打印 ───────────────────────────────────────────────────
    _print_metrics(metrics, args.val_fold)

    # ── 保存 ───────────────────────────────────────────────────
    save_dir = args.save_dir or os.path.dirname(args.ckpt)
    _save_metrics(metrics, save_dir, args.val_fold)

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
    print(f"  Cls Acc   : {metrics['cls_acc']:.4f}  ←匹配实例中分类正确率")
    print(f'\n  Per-class PQ:')
    for cls_name, v in metrics['per_class'].items():
        print(f"    {cls_name:<14}: PQ={v['PQ']:.4f}  DQ={v['DQ']:.4f}  SQ={v['SQ']:.4f}")
    print(sep)

def _save_metrics(metrics, save_dir, fold_name):
    os.makedirs(save_dir, exist_ok=True)

    # json（机器可读）
    json_path = os.path.join(save_dir, f'metrics_{fold_name}.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # txt（人类可读）
    txt_path = os.path.join(save_dir, f'metrics_{fold_name}.txt')
    with open(txt_path, 'w') as f:
        f.write(f'Results on {fold_name}\n')
        f.write('=' * 60 + '\n')
        for k in ['PQ', 'DQ', 'SQ', 'F1', 'Precision', 'Recall', 'cls_acc']:
            f.write(f'  {k:<12}: {metrics[k]:.4f}\n')
        f.write('\nPer-class PQ:\n')
        for cls_name, v in metrics['per_class'].items():
            f.write(f"  {cls_name:<14}: PQ={v['PQ']:.4f}  "f"DQ={v['DQ']:.4f}  SQ={v['SQ']:.4f}\n")

    print(f'[Saved] {json_path}')
    print(f'[Saved] {txt_path}')

# ──────────────────────────────────────────────────────────────
def main():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HoverSegModel(base_ch=args.base_ch, num_classes=args.num_classes).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"[Loaded] epoch={ckpt['epoch']}  best_val_loss={ckpt['best_val_loss']:.4f}")

    val_root   = os.path.join(args.data_root, args.val_fold)
    val_loader = get_dataloader(
        val_root, batch_size=args.batch_size, shuffle=False,
        img_size=args.img_size, num_classes=args.num_classes,
        num_workers=args.num_workers,
    )

    evaluate(model, val_loader, device, args)

if __name__ == '__main__':
    main()