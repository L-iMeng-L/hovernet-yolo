import os, json, argparse
import torch
import numpy as np
from tqdm import tqdm
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from models.seg_model import HoverSegModel
from data.dataset import get_dataloader, collate_fn  
from utils.post_process import batch_postprocess
from utils.metrics import batch_seg_metrics

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
def evaluate_params(model, loader, device, params, num_workers=8):
    model.eval()
    all_pred_insts, all_true_insts, all_pred_cls, all_true_cls = [], [], [], []
    
    for imgs, _, _, hover_gts in tqdm(loader, desc='Evaluating', ncols=80, leave=False):
        out = model(imgs.to(device))
        pred_insts, pred_cls_list = batch_postprocess(
            out['np_map'], out['hv_map'], out['nc_map'], **params
        )
        true_insts = [m.cpu().numpy() if torch.is_tensor(m) else m for m in hover_gts['inst_map']]
        nc_maps_gt = hover_gts['nc_map'].cpu().numpy()
        true_cls_list = [_build_true_cls(true_insts[b], nc_maps_gt[b]) 
                         for b in range(len(true_insts))]
        
        all_pred_insts.extend(pred_insts)
        all_true_insts.extend(true_insts)
        all_pred_cls.extend(pred_cls_list)
        all_true_cls.extend(true_cls_list)
    
    return batch_seg_metrics(all_pred_insts, all_true_insts, 
                            all_pred_cls, all_true_cls, 
                            match_iou=0.5, num_workers=num_workers)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--val_fold', default='Fold3')
    p.add_argument('--data_root', default='/home/lwy/dataset/PanNuke/processed')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--img_size', type=int, default=640)
    p.add_argument('--base_ch', type=int, default=64)
    p.add_argument('--num_classes', type=int, default=5)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--n_calls', type=int, default=100, help='Bayesian iterations')
    p.add_argument('--sample_size', type=int, default=1000, help='Samples for search (0=all)')
    p.add_argument('--save_dir', default='bayesian_search_results')
    args = p.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HoverSegModel(args.base_ch, args.num_classes).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    
    val_root = os.path.join(args.data_root, args.val_fold)
    
    val_loader_full = get_dataloader(
        val_root, 
        batch_size=args.batch_size, 
        shuffle=False,
        img_size=args.img_size, 
        num_classes=args.num_classes,
        num_workers=args.num_workers,  
        is_train=False
    )
    val_dataset = val_loader_full.dataset
    
    if args.sample_size > 0:
        total = len(val_dataset)
        indices = np.random.choice(total, min(args.sample_size, total), replace=False)
        val_dataset = torch.utils.data.Subset(val_dataset, indices)
        print(f"Using {len(indices)}/{total} samples for search")

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn, 
        pin_memory=True
    )
    
    # 搜索空间
    space = [
        Real(0.40, 0.60, name='np_thresh'),
        Real(0.40, 0.70, name='overall_thresh'),
        Categorical([11, 13, 15, 17, 19, 21], name='ksize'),
        Integer(3, 7, name='marker_ksize'),
        Integer(2, 10, name='min_area')
    ]
    
    @use_named_args(space)
    def objective(**params):
        metrics = evaluate_params(model, val_loader, device, params, num_workers=args.num_workers)
        return -metrics['PQ']
    
    print(f"Bayesian optimization: {args.n_calls} iterations on {args.val_fold}...")
    result = gp_minimize(objective, space, n_calls=args.n_calls, 
                        random_state=42, verbose=False)
    
    best_params = {
        'np_thresh': float(result.x[0]),
        'overall_thresh': float(result.x[1]),
        'ksize': int(result.x[2]),
        'marker_ksize': int(result.x[3]),
        'min_area': int(result.x[4])
    }
    best_pq = -result.fun
    
    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, f'best_params_{args.val_fold}.json')
    with open(out_path, 'w') as f:
        json.dump({'best_params': best_params, 'best_pq': best_pq}, f, indent=2)
    
    print(f"\nBest params: {best_params}")
    print(f"Best PQ: {best_pq:.4f}")
    print(f"Saved to: {out_path}")

if __name__ == '__main__':
    main()