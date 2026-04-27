# train.py
import os
import logging
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm

from models.seg_model import HoverSegModel
from losses.seg_loss import seg_loss
from data.dataset import get_dataloader
from utils.metrics import compute_np_iou
from utils.logger import CSVLogger

#── 日志 ──────────────────────────────────────────────────────
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

# ── CSV 字段（与plot_curves.py 完全对齐）────────────────────
CSV_FIELDS = [
    'epoch', 'lr',
    'train_loss', 'train_loss_np', 'train_loss_hv', 'train_loss_nc',
    'val_loss','val_loss_np','val_loss_hv',   'val_loss_nc',
    'train_np_iou', 'val_iou', 'val_np_iou',
    'is_best',
]

# ── args──────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',   default='/home/lwy/dataset/PanNuke/processed')
    p.add_argument('--val_fold',    default='Fold2')
    p.add_argument('--epochs',      type=int,   default=100)
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--img_size',    type=int,   default=640)
    p.add_argument('--base_ch',     type=int,   default=64)
    p.add_argument('--num_classes', type=int,   default=5)
    p.add_argument('--save_dir',    default='./runs')
    p.add_argument('--resume',      default='')
    p.add_argument('--num_workers', type=int,   default=8)
    return p.parse_args()

# ── device helper ─────────────────────────────────────────────
def _to_device(hover_gts, device):
    out = {}
    for k, v in hover_gts.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v          # inst_map: list of Tensor，评估时才用
    return out

# ── train ─────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = dict(loss=0., loss_np=0., loss_hv=0., loss_nc=0., np_iou=0.)
    n = len(loader)

    pbar = tqdm(loader, desc='Train', leave=False,
                bar_format='{l_bar}{bar:20}{r_bar}')
    for imgs, bboxes, labels, hover_gts in pbar:
        imgs      = imgs.to(device)
        hover_gts = _to_device(hover_gts, device)

        out = model(imgs)
        loss, details = seg_loss(out, hover_gts)
        np_iou = compute_np_iou(out, hover_gts)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        total['loss']+= loss.item()
        total['loss_np'] += details['loss_np']
        total['loss_hv'] += details['loss_hv']
        total['loss_nc'] += details['loss_nc']
        total['np_iou']  += np_iou

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            np  =f"{details['loss_np']:.4f}",
            hv  =f"{details['loss_hv']:.4f}",
            nc  =f"{details['loss_nc']:.4f}",
            iou =f"{np_iou:.4f}",
        )

    return {k: v / n for k, v in total.items()}

# ── val ───────────────────────────────────────────────────────
@torch.no_grad()
def val_one_epoch(model, loader, device):
    model.eval()
    total = dict(loss=0., loss_np=0., loss_hv=0., loss_nc=0., np_iou=0.)
    n = len(loader)

    pbar = tqdm(loader, desc='Val', leave=False,
                bar_format='{l_bar}{bar:20}{r_bar}')
    for imgs, bboxes, labels, hover_gts in pbar:
        imgs      = imgs.to(device)
        hover_gts = _to_device(hover_gts, device)

        out    = model(imgs)
        loss, details = seg_loss(out, hover_gts)
        np_iou = compute_np_iou(out, hover_gts)

        total['loss']    += loss.item()
        total['loss_np'] += details['loss_np']
        total['loss_hv'] += details['loss_hv']
        total['loss_nc'] += details['loss_nc']
        total['np_iou']  += np_iou

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            nc  =f"{details['loss_nc']:.4f}",
            iou =f"{np_iou:.4f}",
        )

    return {k: v / n for k, v in total.items()}

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

    train_loader = get_dataloader(
        train_roots, batch_size=args.batch_size, shuffle=True,
        img_size=args.img_size, num_classes=args.num_classes,
        num_workers=args.num_workers,
    )
    val_loader = get_dataloader(
        val_root, batch_size=args.batch_size, shuffle=False,
        img_size=args.img_size, num_classes=args.num_classes,
        num_workers=args.num_workers,
    )

    model = HoverSegModel(
        base_ch=args.base_ch,
        num_classes=args.num_classes,
    ).to(device)

    # nc_head单独 2× lr
    nc_params= [p for n, p in model.named_parameters() if 'nc' in n]
    other_params = [p for n, p in model.named_parameters() if 'nc' not in n]
    optimizer = optim.AdamW([
        {'params': other_params, 'lr': args.lr},
        {'params': nc_params,    'lr': args.lr * 2},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch, best_val_loss = 0, float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        logger.info(f"[Resume] epoch={start_epoch}best_val_loss={best_val_loss:.4f}")

    for epoch in range(start_epoch, args.epochs):
        tr = train_one_epoch(model, train_loader, optimizer, device)
        vl = val_one_epoch(model, val_loader, device)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        is_best    = vl['loss'] < best_val_loss

        if is_best:
            best_val_loss = vl['loss']
            torch.save({
                'epoch':           epoch,
                'model_state':     model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_val_loss':   best_val_loss,
            }, os.path.join(save_dir, 'best.pt'))

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch':           epoch,
                'model_state':     model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_val_loss':   best_val_loss,
            }, os.path.join(save_dir, f'epoch_{epoch:03d}.pt'))

        msg = (f"[{epoch:03d}/{args.epochs}] lr={current_lr:.2e} "
               f"train loss={tr['loss']:.4f} np={tr['loss_np']:.4f} "
               f"hv={tr['loss_hv']:.4f} nc={tr['loss_nc']:.4f} | "
               f"val loss={vl['loss']:.4f} nc={vl['loss_nc']:.4f} "
               f"iou={vl['np_iou']:.4f}"
               f"{' ← best' if is_best else ''}")
        logger.info(msg)

        csv_log.log({
            'epoch':        epoch,
            'lr':round(current_lr, 8),
            'train_loss':   round(tr['loss'],    4),
            'train_loss_np':round(tr['loss_np'], 4),
            'train_loss_hv':round(tr['loss_hv'], 4),
            'train_loss_nc':round(tr['loss_nc'], 4),
            'val_loss':     round(vl['loss'],    4),
            'val_loss_np':  round(vl['loss_np'], 4),
            'val_loss_hv':  round(vl['loss_hv'], 4),
            'val_loss_nc':  round(vl['loss_nc'], 4),
            'train_np_iou': round(tr['np_iou'],  4),
            'val_iou':      round(vl['np_iou'],  4),
            'val_np_iou':   round(vl['np_iou'],  4),   # 同一列两个名字兼容
            'is_best':      int(is_best),
        })

    csv_log.close()
    logger.info(f"\n训练完成/最佳模型: {os.path.join(save_dir, 'best.pt')}")
    logger.info(f"画图: python plot_curves.py --log_dir {save_dir}")

if __name__ == '__main__':
    main()