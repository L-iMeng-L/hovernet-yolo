# train.py
import os
import logging
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from models.seg_model import HoverSegModel
from losses.seg_loss import seg_loss
from data.dataset import get_dataloader
from utils.metrics import compute_np_iou
from utils.logger import CSVLogger
from utils.post_process import post_process_hovernet

# ── 日志 ──────────────────────────────────────────────────────
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

# ── CSV 字段 ──────────────────────────────────────────────────
CSV_FIELDS = [
    'epoch', 'lr',
    'train_loss', 'train_loss_np', 'train_loss_hv', 'train_loss_nc',
    'val_loss', 'val_loss_np', 'val_loss_hv', 'val_loss_nc',
    'train_np_iou', 'val_iou', 'val_np_iou',
    'is_best',
]

# ── args ──────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',default='/home/lwy/dataset/PanNuke/processed')
    p.add_argument('--val_fold',    default='Fold2')
    p.add_argument('--epochs',      type=int,   default=80)
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--img_size',    type=int,   default=640)
    p.add_argument('--base_ch',     type=int,   default=64)
    p.add_argument('--num_classes', type=int,   default=5)
    p.add_argument('--save_dir',    default='./runs')
    p.add_argument('--resume',      default='')
    p.add_argument('--num_workers', type=int,   default=8)
    p.add_argument('--patience',    type=int,   default=15)   # ← early stop
    return p.parse_args()

# ── device helper ─────────────────────────────────────────────
def _to_device(hover_gts, device):
    out = {}
    for k, v in hover_gts.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out

# ── train ─────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    losses = {'total': [], 'np': [], 'hv': [], 'hv_dir': [], 'nc': [], 'iou': []}
    
    pbar = tqdm(loader, desc=f'Train')
    for batch in pbar:
        imgs = batch['img'].to(device)
        np_gt = batch['np_map'].to(device)
        hv_gt = batch['hv_map'].to(device)
        nc_gt = batch['nc_map'].to(device)
        
        optimizer.zero_grad()
        out = model(imgs)
        
        # ===== 修改损失权重 =====
        loss_np = F.binary_cross_entropy(out['np_map'], np_gt)
        loss_hv_mse = F.mse_loss(out['hv_map'], hv_gt)
        
        # HV方向损失（归一化后计算余弦）
        pred_norm = F.normalize(out['hv_map'].permute(0,2,3,1), dim=-1, eps=1e-8)
        gt_norm = F.normalize(hv_gt.permute(0,2,3,1), dim=-1, eps=1e-8)
        cos_sim = (pred_norm * gt_norm).sum(dim=-1)
        loss_hv_dir = (1 - cos_sim).mean()
        
        # NC分类损失
        loss_nc = F.cross_entropy(
            out['nc_map'], 
            nc_gt.argmax(1), 
            label_smoothing=0.1  # 添加标签平滑
        )
        
        # IoU损失（辅助NP学习）
        pred_binary = (out['np_map'] > 0.5).float()
        gt_binary = (np_gt > 0.5).float()
        intersection = (pred_binary * gt_binary).sum()
        union = pred_binary.sum() + gt_binary.sum() - intersection
        iou = intersection / (union + 1e-6)
        loss_iou = 1 - iou
        
        # ===== 新权重配置 =====
        loss = (
            1.0 * loss_np +        # NP权重降低
            3.0 * loss_hv_mse +    # HV MSE权重提高
            2.0 * loss_hv_dir +    # 方向损失权重提高
            1.5 * loss_nc +        # NC权重提高
            0.5 * loss_iou         # IoU辅助
        )
        
        loss.backward()
        
        # ===== 添加梯度裁剪 =====
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        losses['total'].append(loss.item())
        losses['np'].append(loss_np.item())
        losses['hv'].append(loss_hv_mse.item())
        losses['hv_dir'].append(loss_hv_dir.item())
        losses['nc'].append(loss_nc.item())
        losses['iou'].append(iou.item())
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'np': f"{loss_np.item():.4f}",
            'hv': f"{loss_hv_mse.item():.4f}",
            'hv_dir': f"{loss_hv_dir.item():.4f}",
            'nc': f"{loss_nc.item():.4f}",
            'iou': f"{iou.item():.4f}"
        })
    
    return {k: np.mean(v) for k, v in losses.items()}

# ── val ───────────────────────────────────────────────────────
# train.py 中修改
from utils.post_process import post_process_hovernet

@torch.no_grad()
def val_one_epoch(model, loader, device, epoch):
    model.eval()
    losses = {'total': [], 'np': [], 'hv': [], 'hv_dir': [], 'nc': [], 'iou': []}
    all_inst_results = []
    
    pbar = tqdm(loader, desc=f'Val')
    for batch in pbar:
        imgs = batch['img'].to(device)
        np_gt = batch['np_map'].to(device)
        hv_gt = batch['hv_map'].to(device)
        nc_gt = batch['nc_map'].to(device)
        
        out = model(imgs)
        
        # 计算损失（保持原有逻辑）
        loss_np = F.binary_cross_entropy(out['np_map'], np_gt)
        loss_hv_mse = F.mse_loss(out['hv_map'], hv_gt)
        
        pred_norm = F.normalize(out['hv_map'].permute(0,2,3,1), dim=-1, eps=1e-8)
        gt_norm = F.normalize(hv_gt.permute(0,2,3,1), dim=-1, eps=1e-8)
        cos_sim = (pred_norm * gt_norm).sum(dim=-1)
        loss_hv_dir = (1 - cos_sim).mean()
        
        loss_nc = F.cross_entropy(out['nc_map'], nc_gt.argmax(1), label_smoothing=0.1)
        
        pred_binary = (out['np_map'] > 0.5).float()
        gt_binary = (np_gt > 0.5).float()
        intersection = (pred_binary * gt_binary).sum()
        union = pred_binary.sum() + gt_binary.sum() - intersection
        iou = intersection / (union + 1e-6)
        loss_iou = 1 - iou
        
        loss = 1.0*loss_np + 3.0*loss_hv_mse + 2.0*loss_hv_dir + 1.5*loss_nc + 0.5*loss_iou
        
        losses['total'].append(loss.item())
        losses['np'].append(loss_np.item())
        losses['hv'].append(loss_hv_mse.item())
        losses['hv_dir'].append(loss_hv_dir.item())
        losses['nc'].append(loss_nc.item())
        losses['iou'].append(iou.item())
        
        # 新增：实例级后处理评估
        for i in range(imgs.size(0)):
            result = post_process_hovernet({
                'np_map': out['np_map'][i:i+1],
                'hv_map': out['hv_map'][i:i+1],
                'nc_map': out['nc_map'][i:i+1]
            }, min_area=10)
            all_inst_results.append(result)
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'iou': f"{iou.item():.4f}",
            'nc': f"{loss_nc.item():.4f}"
        })
    
    # 计算实例级指标
    avg_inst_per_img = np.mean([len(r['inst_info']) for r in all_inst_results])
    print(f"[Val epoch={epoch}] avg_instances={avg_inst_per_img:.1f}")
    
    return {k: np.mean(v) for k, v in losses.items()}

# ── main ──────────────────────────────────────────────────────
def main():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ALL_FOLDS   = ['Fold1', 'Fold2', 'Fold3']
    train_folds = [f for f in ALL_FOLDS if f != args.val_fold]
    train_roots = [os.path.join(args.data_root, f) for f in train_folds]
    val_root    = os.path.join(args.data_root, args.val_fold)

    save_dir = os.path.join(
        args.save_dir, f"{'_'.join(train_folds)}_vs_{args.val_fold}")
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger(save_dir)
    logger.info(f"[Config] {vars(args)}")

    csv_log = CSVLogger(save_dir, filename='log.csv')
    csv_log.init(CSV_FIELDS)

    # ← is_train=True 开启增强；val不开
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

    model = HoverSegModel(
        base_ch=args.base_ch,
        num_classes=args.num_classes,
    ).to(device)
    # ===== 修改优化器：分组学习率 =====
    backbone_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            decoder_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},  # backbone用1/10学习率
        {'params': decoder_params, 'lr': args.lr}          # decoder用完整学习率
    ], weight_decay=1e-4)
    
    # ===== 修改学习率调度器 =====
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr * 0.1, args.lr],  # 对应两组参数
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,        # 前10%用于warmup
        anneal_strategy='cos',
        div_factor=25.0,      # 初始lr = max_lr/25
        final_div_factor=1e4  # 最终lr = max_lr/1e4
    )

    def _warmup_cosine(epoch):
        warmup_epochs = 10
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_warmup_cosine)

    start_epoch, best_val_loss = 0, float('inf')
    no_improve = 0                # ← early stop 计数

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        no_improve    = ckpt.get('no_improve', 0)
        logger.info(f"[Resume] epoch={start_epoch} best_val_loss={best_val_loss:.4f}")

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics   = val_one_epoch(model, val_loader, device)
        scheduler.step()

        val_loss = val_metrics['loss']
        is_best  = val_loss < best_val_loss

        # ── early stopping────────────────────────────────────
        if is_best:
            best_val_loss = val_loss
            no_improve    = 0
        else:
            no_improve += 1

        lr_now = scheduler.get_last_lr()[0]
        logger.info(
            f"[{epoch:03d}] lr={lr_now:.2e} "
            f"train={train_metrics['loss']:.4f} val={val_loss:.4f} "
            f"nc_tr={train_metrics['loss_nc']:.4f} nc_val={val_metrics['loss_nc']:.4f} "
            f"{'BEST' if is_best else f'no_improve={no_improve}/{args.patience}'}"
        )

        csv_log.log({
            'epoch': epoch,
            'lr'           : lr_now,
            'train_loss'   : train_metrics['loss'],
            'train_loss_np': train_metrics['loss_np'],
            'train_loss_hv': train_metrics['loss_hv'],
            'train_loss_nc': train_metrics['loss_nc'],
            'val_loss'     : val_metrics['loss'],
            'val_loss_np'  : val_metrics['loss_np'],
            'val_loss_hv'  : val_metrics['loss_hv'],
            'val_loss_nc'  : val_metrics['loss_nc'],
            'train_np_iou' : train_metrics['np_iou'],
            'val_iou'      : val_metrics['np_iou'],
            'val_np_iou'   : val_metrics['np_iou'],
            'is_best'      : int(is_best),
        })

        # ──保存 checkpoint ───────────────────────────────────
        ckpt = {
            'epoch'          : epoch,
            'model_state'    : model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_val_loss'  : best_val_loss,
            'no_improve'     : no_improve,
        }
        torch.save(ckpt, os.path.join(save_dir, 'last.pth'))
        if is_best:
            torch.save(ckpt, os.path.join(save_dir, 'best.pth'))

        # ── early stop ────────────────────────────────────────
        if no_improve >= args.patience:
            logger.info(f"[Early Stop] val_loss {args.patience} epoch未改善，停止训练")
            break
    logger.info(f"训练结束，best_val_loss={best_val_loss:.4f}")

if __name__ == '__main__':
    main()