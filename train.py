import os
import logging
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

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
    'epoch', 'lr_backbone', 'lr_decoder',
    'train_loss', 'train_loss_np', 'train_loss_hv', 'train_loss_nc',
    'val_loss', 'val_loss_np', 'val_loss_hv', 'val_loss_nc',
    'train_np_iou', 'val_np_iou', 'is_best',
]

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='/home/lwy/dataset/PanNuke/processed')
    p.add_argument('--val_fold', default='Fold3')
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--img_size', type=int, default=640)
    p.add_argument('--base_ch', type=int, default=64)
    p.add_argument('--num_classes', type=int, default=5)
    p.add_argument('--save_dir', default='./runs/third_try')
    p.add_argument('--resume', default='')
    p.add_argument('--num_workers', type=int, default=32)
    p.add_argument('--patience', type=int, default=20)
    return p.parse_args()

def soft_dice_loss(pred, target, eps=1e-6):
    """
    pred: (B, 1, H, W) or (B, C, H, W), probabilities in [0,1]
    target: same shape, binary labels
    """
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()

def masked_regression_loss(pred, target, mask, loss_type='smooth_l1', eps=1e-8):
    """
    pred/target: (B, C, H, W)
    mask: (B, 1, H, W) or (B, C, H, W), 1 for foreground, 0 for background
    """
    if mask.dim() == 4 and mask.shape[1] == 1 and pred.shape[1] > 1:
        mask = mask.expand_as(pred)

    if loss_type == 'smooth_l1':
        diff = F.smooth_l1_loss(pred, target, reduction='none')
    else:
        diff = (pred - target) ** 2

    loss = (diff * mask).sum() / (mask.sum() + eps)
    return loss

def get_group_lrs(optimizer):
    lrs = [group['lr'] for group in optimizer.param_groups]
    if len(lrs) >= 2:
        return lrs[0], lrs[1]
    elif len(lrs) == 1:
        return lrs[0], lrs[0]
    else:
        return 0.0, 0.0

def train_one_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    losses = {'total': [], 'np': [], 'hv': [], 'nc': [], 'iou': []}

    pbar = tqdm(loader, desc=f'Train Epoch {epoch}')
    for batch in pbar:
        imgs, _, _, hover_gts = batch
        imgs = imgs.to(device)
        np_gt = hover_gts['np_map'].to(device).float()
        hv_gt = hover_gts['hv_map'].to(device).float()
        nc_gt = hover_gts['nc_map'].to(device).long()

        optimizer.zero_grad()
        out = model(imgs)

        # ---------------------------
        # NP branch: BCEWithLogits + Soft Dice
        # ---------------------------
        np_logits = out['np_map']
        np_prob = torch.sigmoid(np_logits)

        loss_np_bce = F.binary_cross_entropy_with_logits(np_logits, np_gt)
        loss_np_dice = soft_dice_loss(np_prob, np_gt)
        loss_np = loss_np_bce + loss_np_dice

        # metric only
        pred_binary = (np_prob > 0.5).float()
        gt_binary = (np_gt > 0.5).float()
        intersection = (pred_binary * gt_binary).sum()
        union = pred_binary.sum() + gt_binary.sum() - intersection
        iou = intersection / (union + 1e-6)

        # ---------------------------
        # HV branch: foreground dominant, background weak
        # ---------------------------
        hv_pred = out['hv_map']
        nuclei_mask = (np_gt > 0.3).float()
        bg_mask = 1.0 - nuclei_mask

        hv_fg_loss = masked_regression_loss(
            hv_pred, hv_gt, nuclei_mask, loss_type='smooth_l1'
        )
        hv_bg_loss = masked_regression_loss(
            hv_pred, hv_gt, bg_mask, loss_type='smooth_l1'
        )

        loss_hv = 1.0 * hv_fg_loss + 0.1 * hv_bg_loss

        # ---------------------------
        # NC branch: CE + Dice, rebalanced
        # ---------------------------
        nc_logits = out['nc_map']  # (B, C, H, W)
        num_classes = nc_logits.shape[1]

        logits_flat = nc_logits.permute(0, 2, 3, 1).reshape(-1, num_classes)
        targets_flat = nc_gt.reshape(-1)

        valid_mask = targets_flat >= 0

        if valid_mask.sum() > 0:
            loss_nc_ce = F.cross_entropy(
                logits_flat[valid_mask],
                targets_flat[valid_mask],
                label_smoothing=0.03
            )

            nc_prob = F.softmax(nc_logits, dim=1)
            nc_prob_flat = nc_prob.permute(0, 2, 3, 1).reshape(-1, num_classes)[valid_mask]
            nc_gt_onehot = F.one_hot(targets_flat[valid_mask], num_classes=num_classes).float()

            intersection_nc = (nc_prob_flat * nc_gt_onehot).sum(dim=0)
            denom_nc = nc_prob_flat.sum(dim=0) + nc_gt_onehot.sum(dim=0)
            loss_nc_dice = 1.0 - ((2.0 * intersection_nc + 1e-6) / (denom_nc + 1e-6)).mean()

            loss_nc = 0.8 * loss_nc_ce + 0.2 * loss_nc_dice
        else:
            loss_nc = torch.tensor(0.0, device=device)

        # ---------------------------
        # Total
        # ---------------------------
        loss = 0.5 * loss_hv + 1.0 * loss_np + 0.8 * loss_nc

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        # OneCycleLR: step per iteration
        if scheduler is not None:
            scheduler.step()

        losses['total'].append(loss.item())
        losses['np'].append(loss_np.item())
        losses['hv'].append(loss_hv.item())
        losses['nc'].append(loss_nc.item())
        losses['iou'].append(iou.item())

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'np': f"{loss_np.item():.4f}",
            'hv': f"{loss_hv.item():.4f}",
            'nc': f"{loss_nc.item():.4f}",
            'iou': f"{iou.item():.4f}",
        })

    return {k: float(np.mean(v)) for k, v in losses.items()}

@torch.no_grad()
def val_one_epoch(model, loader, device, epoch):
    model.eval()
    losses = {'total': [], 'np': [], 'hv': [], 'nc': [], 'iou': []}

    for batch in tqdm(loader, desc=f'Val Epoch {epoch}'):
        imgs, _, _, hover_gts = batch
        imgs = imgs.to(device)
        np_gt = hover_gts['np_map'].to(device).float()
        hv_gt = hover_gts['hv_map'].to(device).float()
        nc_gt = hover_gts['nc_map'].to(device).long()

        out = model(imgs)

        # NP
        np_logits = out['np_map']
        np_prob = torch.sigmoid(np_logits)

        loss_np_bce = F.binary_cross_entropy_with_logits(np_logits, np_gt)
        loss_np_dice = soft_dice_loss(np_prob, np_gt)
        loss_np = loss_np_bce + loss_np_dice

        pred_binary = (np_prob > 0.5).float()
        gt_binary = (np_gt > 0.5).float()
        intersection = (pred_binary * gt_binary).sum()
        union = pred_binary.sum() + gt_binary.sum() - intersection
        iou = intersection / (union + 1e-6)

        # HV
        hv_pred = out['hv_map']
        nuclei_mask = (np_gt > 0.3).float()
        bg_mask = 1.0 - nuclei_mask

        hv_fg_loss = masked_regression_loss(
            hv_pred, hv_gt, nuclei_mask, loss_type='smooth_l1'
        )
        hv_bg_loss = masked_regression_loss(
            hv_pred, hv_gt, bg_mask, loss_type='smooth_l1'
        )

        loss_hv = 1.0 * hv_fg_loss + 0.1 * hv_bg_loss

        # NC
        nc_logits = out['nc_map']
        num_classes = nc_logits.shape[1]
        logits_flat = nc_logits.permute(0, 2, 3, 1).reshape(-1, num_classes)
        targets_flat = nc_gt.reshape(-1)
        valid_mask = targets_flat >= 0

        if valid_mask.sum() > 0:
            loss_nc_ce = F.cross_entropy(
                logits_flat[valid_mask],
                targets_flat[valid_mask],
                label_smoothing=0.03
            )

            nc_prob = F.softmax(nc_logits, dim=1)
            nc_prob_flat = nc_prob.permute(0, 2, 3, 1).reshape(-1, num_classes)[valid_mask]
            nc_gt_onehot = F.one_hot(targets_flat[valid_mask], num_classes=num_classes).float()

            intersection_nc = (nc_prob_flat * nc_gt_onehot).sum(dim=0)
            denom_nc = nc_prob_flat.sum(dim=0) + nc_gt_onehot.sum(dim=0)
            loss_nc_dice = 1.0 - ((2.0 * intersection_nc + 1e-6) / (denom_nc + 1e-6)).mean()

            loss_nc = 0.8 * loss_nc_ce + 0.2 * loss_nc_dice
        else:
            loss_nc = torch.tensor(0.0, device=device)

        loss = 0.5 * loss_hv + 1.0 * loss_np + 0.8 * loss_nc

        losses['total'].append(loss.item())
        losses['np'].append(loss_np.item())
        losses['hv'].append(loss_hv.item())
        losses['nc'].append(loss_nc.item())
        losses['iou'].append(iou.item())

    return {k: float(np.mean(v)) for k, v in losses.items()}

def main():
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tb'))

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
    other_decoder_params = []
    hv_decoder_params = []

    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        elif 'hv_head' in name or 'hv_decoder' in name:
            hv_decoder_params.append(param)
        else:
            other_decoder_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},      # 1e-5
        {'params': other_decoder_params, 'lr': args.lr},       # 1e-4
        {'params': hv_decoder_params, 'lr': 0.5*args.lr}        # 5e-5
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr * 0.1, args.lr, 0.5*args.lr],
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

        backbone_lr, decoder_lr = get_group_lrs(optimizer)

        logger.info(
            f"[{epoch:03d}] "
            f"backbone_lr={backbone_lr:.2e} decoder_lr={decoder_lr:.2e} "
            f"train={train_loss:.4f} val={val_loss:.4f} "
            f"np_tr={train_metrics['np']:.4f} np_val={val_metrics['np']:.4f} "
            f"hv_tr={train_metrics['hv']:.4f} hv_val={val_metrics['hv']:.4f} "
            f"nc_tr={train_metrics['nc']:.4f} nc_val={val_metrics['nc']:.4f} "
            f"{'BEST' if is_best else f'no_improve={no_improve}/{args.patience}'}"
        )

        csv_log.log({
            'epoch': epoch,
            'lr_backbone': backbone_lr,
            'lr_decoder': decoder_lr,
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

        writer.add_scalar('Loss/Train_Total', train_metrics['total'], epoch)
        writer.add_scalar('Loss/Train_NP', train_metrics['np'], epoch)
        writer.add_scalar('Loss/Train_HV', train_metrics['hv'], epoch)
        writer.add_scalar('Loss/Train_NC', train_metrics['nc'], epoch)

        writer.add_scalar('Loss/Val_Total', val_metrics['total'], epoch)
        writer.add_scalar('Loss/Val_NP', val_metrics['np'], epoch)
        writer.add_scalar('Loss/Val_HV', val_metrics['hv'], epoch)
        writer.add_scalar('Loss/Val_NC', val_metrics['nc'], epoch)

        writer.add_scalar('Metric/Train_NP_IoU', train_metrics['iou'], epoch)
        writer.add_scalar('Metric/Val_NP_IoU', val_metrics['iou'], epoch)

        writer.add_scalar('Optim/Backbone_LR', backbone_lr, epoch)
        writer.add_scalar('Optim/Decoder_LR', decoder_lr, epoch)

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
            logger.info(f"[Early Stop] val_loss 连续 {args.patience} 个 epoch 未改善")
            break

    writer.close()
    logger.info(f"训练结束，best_val_loss={best_val_loss:.4f}")

if __name__ == '__main__':
    main()