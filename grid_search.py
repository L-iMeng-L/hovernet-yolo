import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
import itertools

from models.seg_model import HoverSegModel
from data.dataset import get_dataloader
from utils.post_process import batch_postprocess
from utils.seg_metrics import batch_seg_metrics

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

@torch.no_grad()
def evaluate_params(model, loader, device, params):
    model.eval()
    all_pred_insts, all_true_insts = [], []
    all_pred_cls, all_true_cls = [], []
    
    for imgs, _, _, hover_gts in loader:
        imgs = imgs.to(device)
        out = model(imgs)
        
        pred_insts, pred_cls_list = batch_postprocess(
            out['np_map'], out['hv_map'], out['nc_map'], **params
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
    
    return batch_seg_metrics(
        all_pred_insts, all_true_insts,
        all_pred_cls, all_true_cls, match_iou=0.5
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--val_fold', default='Fold3')
    p.add_argument('--data_root', default='/home/lwy/dataset/PanNuke/processed')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--img_size', type=int, default=640)
    p.add_argument('--base_ch', type=int, default=64)
    p.add_argument('--num_classes', type=int, default=5)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--save_dir', default='grid_search_results')
    args = p.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = HoverSegModel(args.base_ch, args.num_classes).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    
    val_root = os.path.join(args.data_root, args.val_fold)
    val_loader = get_dataloader(
        val_root, batch_size=args.batch_size, shuffle=False,
        img_size=args.img_size, num_classes=args.num_classes,
        num_workers=args.num_workers, is_train=False
    )
    
    param_grid = {
        'np_thresh': [0.30, 0.32, 0.35],
        'overall_thresh': [0.38, 0.40, 0.42],
        'ksize': [21],
        'marker_ksize': [3, 5],
        'min_area': [3, 5]
    }
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Testing {len(combinations)} combinations on {args.val_fold}...")
    
    results = []
    best_pq = 0
    best_params = None
    
    for params in tqdm(combinations):
        metrics = evaluate_params(model, val_loader, device, params)
        result = {'params': params, **{k: metrics[k] for k in ['PQ', 'DQ', 'SQ', 'F1']}}
        results.append(result)
        
        if metrics['PQ'] > best_pq:
            best_pq = metrics['PQ']
            best_params = params
    
    results = sorted(results, key=lambda x: x['PQ'], reverse=True)
    
    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, f'best_params_{args.val_fold}.json')
    with open(out_path, 'w') as f:
        json.dump({'best_params': best_params, 'best_pq': best_pq, 
                   'top20': results[:20]}, f, indent=2)
    
    print(f"\nBest: {best_params}")
    print(f"PQ: {best_pq:.4f}")
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()