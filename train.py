# train.py 完整修正版

import os
import logging
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from models.seg_model import HoverSegModel
from data.dataset import get_dataloader
from utils.logger import CSVLogger

def setup_logger(save_dir):
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(os.path.join(save_dir, 'train.log'), mode='a')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

CSV_FIELDS = [
    'epoch', 'lr',
    'train_loss', 'train_loss_np', 'train_loss_hv', 'train_loss_nc',
    'val_loss', 'val_loss_np', 'val_loss_hv', 'val_loss_nc',
    'train_np_iou', 'val_np_iou', 'is_best',
]

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',default='/home/lwy/dataset/PanNuke/processed')
    p.add_argument('--val_fold',    default='Fold3')
    p.add_argument('--epochs',      type=int,   default=80)
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--img_size',    type=int,   default=640)
    p.add_argument('--base_ch',     type=int,   default=64)
    p.add_argument('--num_classes', type=int,   default=5)
    p.add_argument('--save_dir',    default='./runs')
    p.add_argument('--resume',      default='')
    p.add_argument('--num_workers', type=int,   default=16)
    p.add_argument('--patience',    type=int,   default=15)
    return p.parse_args()

def train_one_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    losses = {'total': [], 'np': [], 'hv': [], 'nc': [], 'iou': []}
    
    pbar = tqdm(loader, desc=f'Train')
    for batch in pbar:
        imgs, _, _, hover_gts = batch
        imgs = imgs.to(device)
        np_gt = hover_gts['np_map'].to(device)
        hv_gt = hover_gts['hv_map'].to(device)
        nc_gt = hover_gts['nc_map'].to(device)
        
        optimizer.zero_grad()
        out = model(imgs)
        
        loss_np = F.binary_cross_entropy(out['np_map'], np_gt)
        loss_hv = F.mse_loss(out['hv_map'], hv_gt)
        
        mask = nc_gt >= 0
        if mask.sum() > 0:
            loss_nc = F.cross_entropy(
                out['nc_map'].permute(0,2,3,1)[mask],
                nc_gt[mask],
                label_smoothing=0.1
            )
        else:
            loss_nc = torch.tensor(0.0, device=device)
        
        pred_binary = (out['np_map'] > 0.5).float()
        gt_binary = (np_gt > 0.5).float()
        intersection = (pred_binary * gt_binary).sum()
        union = pred_binary.sum() + gt_binary.sum() - intersection
        iou = intersection / (union + 1e-6)
        
        loss = loss_np + 2.0*loss_hv + 1.5*loss_nc
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()
        
        losses['total'].append(loss.item())
        losses['np'].append(loss_np.item())
        losses['hv'].append(loss_hv.item())
        losses['nc'].append(loss_nc.item())
        losses['iou'].append(iou.item())
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'iou': f"{iou.item():.4f}"})
    
    return {k: np.mean(v) for k, v in losses.items()}

@torch.no_grad()
def val_one_epoch(model, loader, device, epoch):
    model.eval()
    losses = {'total': [], 'np': [], 'hv': [], 'nc': [], 'iou': []}
    
    pbar = tqdm(loader, desc=f'Val')
    for batch in pbar:
        imgs, _, _, hover_gts = batch
        imgs = imgs.to(device)
        np_gt = hover_gts['np_map'].to(device)
        hv_gt = hover_gts['hv_map'].to(device)
        nc_gt = hover_gts['nc_map'].to(device)
        
        out = model(imgs)
        
        loss_np = F.binary_cross_entropy(out['np_map'], np_gt)
        loss_hv = F.mse_loss(out['hv_map'], hv_gt)
        
        mask = nc_gt >= 0
        if mask.sum() > 0:
            loss_nc = F.cross_entropy(
                out['nc_map'].permute(0,2,3,1)[mask],
                nc_gt[mask],
                label_smoothing=0.1
            )
        else:
            loss_nc = torch.tensor(0.0, device=device)
        
        pred_binary = (out['np_map'] > 0.5).float()
        gt_binary = (np_gt > 0.5).float()
        intersection = (pred_binary * gt_binary).sum()
        union = pred_binary.sum() + gt_binary.sum() - intersection
        iou = intersection / (union + 1e-6)
        
        loss = loss_np + 2.0*loss_hv + 1.5*loss_nc
        
        losses['total'].append(loss.item())
        losses['np'].append(loss_np.item())
        losses['hv'].append(loss_hv.item())
        losses['nc'].append(loss_nc.item())
        losses['iou'].append(iou.item())
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'iou': f"{iou.item():.4f}"})
    
    return {k: np.mean(v) for k, v in losses.items()}

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ALL_FOLDS = ['Fold1', 'Fold2', 'Fold3']
    train_folds = [f for f in ALL_FOLDS if f != args.val_fold]
    train_roots = [os.path.join(args.data_root, f) for f in train_folds]
    val_root = os.path.join(args.data_root, args.val_fold)

    save_dir = os.path.join(args.save_dir, f"{'_'.join(train_folds)}_vs_{args.val_fold}")
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger(save_dir)
    logger.info(f"[Config] {vars(args)}")

    csv_log = CSVLogger(save_dir, filename='log.csv')
    csv_log.init(CSV_FIELDS)

    train_loader = get_dataloader(
        train_roots, batch_size=args.batch_size, shuffle=True,
        img_size=args.img_size, num_classes=args.num_classes,
        num_workers=args.num_workers, is_train=True,
    )
    val_loader = get_dataloader(
        val_root, batch_size=args.batch_size, shuffle=False,
        img_size=args.img_size, num_classes=args.num_classes,
        num_workers=args.num_workers, is_train=False,
    )

    model = HoverSegModel(base_ch=args.base_ch, num_classes=args.num_classes).to(device)
    
    backbone_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            decoder_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': decoder_params, 'lr': args.lr}
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr * 0.1, args.lr],
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4
    )

    start_epoch, best_val_loss = 0, float('inf')
    no_improve = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        no_improve = ckpt.get('no_improve', 0)
        logger.info(f"[Resume] epoch={start_epoch} best_val_loss={best_val_loss:.4f}")

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_metrics = val_one_epoch(model, val_loader, device, epoch)
        
        train_loss = train_metrics['total']
        val_loss = val_metrics['total']
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1

        lr_now = scheduler.get_last_lr()[0]
        logger.info(
            f"[{epoch:03d}] lr={lr_now:.2e} "
            f"train={train_loss:.4f} val={val_loss:.4f} "
            f"nc_tr={train_metrics['nc']:.4f} nc_val={val_metrics['nc']:.4f} "
            f"{'BEST' if is_best else f'no_improve={no_improve}/{args.patience}'}"
        )

        csv_log.log({
            'epoch': epoch,
            'lr': lr_now,
            'train_loss': train_loss,
            'train_loss_np': train_metrics['np'],
            'train_loss_hv': train_metrics['hv'],
            'train_loss_nc': train_metrics['nc'],
            'val_loss': val_loss,
            'val_loss_np': val_metrics['np'],
            'val_loss_hv': val_metrics['hv'],
            'val_loss_nc': val_metrics['nc'],
            'train_np_iou': train_metrics['iou'],
            'val_np_iou': val_metrics['iou'],
            'is_best': int(is_best),
        })

        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'no_improve': no_improve,
        }
        torch.save(ckpt, os.path.join(save_dir, 'last.pth'))
        if is_best:
            torch.save(ckpt, os.path.join(save_dir, 'best.pth'))

        if no_improve >= args.patience:
            logger.info(f"[Early Stop] val_loss {args.patience} epoch未改善")
            break
    
    logger.info(f"训练结束，best_val_loss={best_val_loss:.4f}")

if __name__ == '__main__':
    main()