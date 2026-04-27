# evaluate.py
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

from models.seg_model import HoverSegModel
from data.dataset import get_dataloader
from utils.post_process import batch_postprocess
from utils.seg_metrics import batch_seg_metrics

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',required=True)
    p.add_argument('--val_fold',      default='Fold2')
    p.add_argument('--data_root',     default='/home/lwy/dataset/PanNuke/processed')
    p.add_argument('--batch_size',    type=int,   default=8)
    p.add_argument('--img_size',      type=int,   default=640)
    p.add_argument('--base_ch',       type=int,   default=64)
    p.add_argument('--num_classes',   type=int,   default=5)
    p.add_argument('--num_workers',   type=int,   default=4)
    p.add_argument('--np_thresh',     type=float, default=0.5)
    p.add_argument('--energy_thresh', type=float, default=0.4)
    p.add_argument('--match_iou',     type=float, default=0.5)
    return p.parse_args()

def _build_true_cls(inst_map_np, nc_map_np):
    """
    inst_map_np : (H,W) int32
    nc_map_np   : (H,W) int64，-1=背景
    返回 dict {inst_id: class_id}
    """
    cls_dict = {}
    for iid in np.unique(inst_map_np):
        if iid == 0:
            continue
        mask = inst_map_np == iid
        labels = nc_map_np[mask]
        labels = labels[labels >= 0]   # 去掉 ignore
        if len(labels) == 0:
            continue
        cls_dict[int(iid)] = int(np.bincount(labels).argmax())
    return cls_dict

@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()

    all_pred_insts, all_true_insts = [], []
    all_pred_cls,all_true_cls   = [], []

    pbar = tqdm(loader, desc='Evaluating', bar_format='{l_bar}{bar:30}{r_bar}')
    for imgs, bboxes, labels, hover_gts in pbar:
        imgs = imgs.to(device)

        out = model(imgs)

        # 后处理（含类别投票）
        pred_insts, pred_cls_list = batch_postprocess(
            out['np_map'], out['hv_map'], out['nc_map'],
            np_thresh=args.np_thresh, energy_thresh=args.energy_thresh,)

        # GT inst_map
        true_insts = [m.cpu().numpy() for m in hover_gts['inst_map']]

        # GT类别（从 inst_map + nc_map 投票）
        nc_maps_gt = hover_gts['nc_map'].cpu().numpy()   # (B,H,W)
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
        all_pred_cls,   all_true_cls,
        match_iou=args.match_iou,
    )

    CLASS_NAMES = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
    print("\n" + "="*60)
    print(f"Results on {args.val_fold}")
    print("="*60)
    print(f"  PQ        : {metrics['PQ']:.4f}")
    print(f"  DQ        : {metrics['DQ']:.4f}")
    print(f"  SQ        : {metrics['SQ']:.4f}")
    print(f"  F1        : {metrics['F1']:.4f}")
    print(f"  Precision : {metrics['Precision']:.4f}")
    print(f"  Recall    : {metrics['Recall']:.4f}")
    print(f"  Cls Acc   : {metrics['cls_acc']:.4f}←匹配实例中分类正确率")
    print("="*60)
    return metrics

def main():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HoverSegModel(base_ch=args.base_ch, num_classes=args.num_classes).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"[Loaded] epoch={ckpt['epoch']}best_val_loss={ckpt['best_val_loss']:.4f}")

    val_root= os.path.join(args.data_root, args.val_fold)
    val_loader = get_dataloader(
        val_root, batch_size=args.batch_size, shuffle=False,
        img_size=args.img_size, num_classes=args.num_classes,
        num_workers=args.num_workers,
    )

    evaluate(model, val_loader, device, args)

if __name__ == '__main__':
    main()