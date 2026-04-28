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
    p.add_argument('--epochs',      type=int,   default=100)
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
    total = dict(loss=0., loss_np=0., loss_hv=0., loss_nc=0., np_iou=0.)
    n = len(loader)

    pbar = tqdm(loader, desc='Train', leave=False,
                bar_format='{l_bar}{bar:20}{r_bar}')
    for i, (imgs, bboxes, labels, hover_gts) in enumerate(pbar):
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
            focal  = f"{details['loss_focal']:.4f}",
            dice   = f"{details['loss_dice']:.4f}",
            hv  =f"{details['loss_hv']:.4f}",
            nc  =f"{details['loss_nc']:.4f}",
            iou =f"{np_iou:.4f}",
        )

        if i == 0 and epoch % 5 == 0:
            with torch.no_grad():
                fg= (hover_gts['hv_map'].abs() > 0.01)
                pred_fg = out['hv_map'][fg].abs().mean().item()
                gt_fg   = hover_gts['hv_map'][fg].abs().mean().item()
            print(f"\n[HV诊断 epoch={epoch}]前景 pred={pred_fg:.4f} gt={gt_fg:.4f}")

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

        out          = model(imgs)
        loss, details = seg_loss(out, hover_gts)
        np_iou= compute_np_iou(out, hover_gts)

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

    # ← NC 分支lr从 2× 降到 0.5×，避免分类头过拟合
    nc_params= [p for n, p in model.named_parameters() if 'nc' in n]
    other_params = [p for n, p in model.named_parameters() if 'nc' not in n]

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    def _warmup_cosine(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs          # 线性warmup
    # cosine decay
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