# train.py
import os
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm

from models.seg_model import HoverSegModel
from losses.seg_loss import seg_loss
from data.dataset import get_dataloader
from utils.metrics import compute_np_iou

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',default='/home/lwy/dataset/PanNuke/processed')
    p.add_argument('--val_fold',    default='Fold2')
    p.add_argument('--epochs',      type=int,   default=100)
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--img_size',    type=int,   default=640)
    p.add_argument('--base_ch',     type=int,   default=64)
    p.add_argument('--num_classes', type=int,   default=5)   # ← 新增
    p.add_argument('--save_dir',    default='./runs')
    p.add_argument('--resume',      default='')
    p.add_argument('--num_workers', type=int,   default=8)
    return p.parse_args()

def _to_device(hover_gts, device):
    """安全地把hover_gts 搬到 device，跳过 list 类型（inst_map）"""
    out = {}
    for k, v in hover_gts.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, list):
            # inst_map 是 list of Tensor，评估时才用，训练不上GPU
            out[k] = v
        else:
            out[k] = v
    return out

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = dict(loss=0., loss_np=0., loss_hv=0., loss_nc=0.)  # ← 加loss_nc
    n = len(loader)

    pbar = tqdm(loader, desc='Train', leave=False, bar_format='{l_bar}{bar:20}{r_bar}')
    for imgs, bboxes, labels, hover_gts in pbar:
        imgs = imgs.to(device)
        hover_gts = _to_device(hover_gts, device)   # ← 用安全版本

        out = model(imgs)
        loss, details = seg_loss(out, hover_gts)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        total['loss']+= loss.item()
        total['loss_np'] += details['loss_np']
        total['loss_hv'] += details['loss_hv']
        total['loss_nc'] += details['loss_nc']   # ← 追踪 nc loss

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'np':   f"{details['loss_np']:.4f}",
            'hv':   f"{details['loss_hv']:.4f}",
            'nc':   f"{details['loss_nc']:.4f}",  # ← 显示
        })

    return {k: v / n for k, v in total.items()}

@torch.no_grad()
def val_one_epoch(model, loader, device):
    model.eval()
    total = dict(loss=0., loss_np=0., loss_hv=0., loss_nc=0., np_iou=0.)
    n = len(loader)

    pbar = tqdm(loader, desc='Val', leave=False, bar_format='{l_bar}{bar:20}{r_bar}')
    for imgs, bboxes, labels, hover_gts in pbar:
        imgs = imgs.to(device)
        hover_gts = _to_device(hover_gts, device)

        out = model(imgs)
        loss, details = seg_loss(out, hover_gts)
        np_iou = compute_np_iou(out, hover_gts)

        total['loss']    += loss.item()
        total['loss_np'] += details['loss_np']
        total['loss_hv'] += details['loss_hv']
        total['loss_nc'] += details['loss_nc']
        total['np_iou']  += np_iou

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'nc':   f"{details['loss_nc']:.4f}",
            'iou':  f"{np_iou:.4f}",
        })

    return {k: v / n for k, v in total.items()}

def main():
    args= get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ALL_FOLDS   = ['Fold1', 'Fold2', 'Fold3']
    train_folds = [f for f in ALL_FOLDS if f != args.val_fold]
    train_roots = [os.path.join(args.data_root, f) for f in train_folds]
    val_root= os.path.join(args.data_root, args.val_fold)

    save_dir = os.path.join(args.save_dir, f"{'_'.join(train_folds)}_vs_{args.val_fold}")
    os.makedirs(save_dir, exist_ok=True)

    train_loader = get_dataloader(
        train_roots, batch_size=args.batch_size, shuffle=True,
        img_size=args.img_size, num_classes=args.num_classes,   # ← 传 num_classes
        num_workers=args.num_workers,
    )
    val_loader = get_dataloader(
        val_root, batch_size=args.batch_size, shuffle=False,
        img_size=args.img_size, num_classes=args.num_classes,
        num_workers=args.num_workers,
    )

    model = HoverSegModel(
        base_ch=args.base_ch,
        num_classes=args.num_classes,   # ← 传 num_classes
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch, best_val_loss = 0, float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        print(f"[Resume] epoch={start_epoch} best_val_loss={best_val_loss:.4f}")

    for epoch in range(start_epoch, args.epochs):
        tr = train_one_epoch(model, train_loader, optimizer, device)
        vl = val_one_epoch(model, val_loader, device)
        scheduler.step()

        is_best = vl['loss'] < best_val_loss
        if is_best:
            best_val_loss = vl['loss']
            torch.save({
                'epoch':epoch,
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

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train loss={tr['loss']:.4f} np={tr['loss_np']:.4f} "
            f"hv={tr['loss_hv']:.4f} nc={tr['loss_nc']:.4f} | "
            f"val loss={vl['loss']:.4f} nc={vl['loss_nc']:.4f} "
            f"iou={vl['np_iou']:.4f} {'← best' if is_best else ''}"
        )
    print(f"\n训练完成！最佳模型: {os.path.join(save_dir, 'best.pt')}")
    print(f"评估: python evaluate.py --ckpt {os.path.join(save_dir, 'best.pt')} --val_fold {args.val_fold}")

if __name__ == '__main__':
    main()