# plot_curves.py
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_train_val(ax, df, train_col, val_col=None, title='', ylabel=None):
    has_train = train_col in df.columns
    has_val = val_col is not None and val_col in df.columns

    if has_train:
        ax.plot(df['epoch'], df[train_col], label='train', linewidth=1.8)
    if has_val:
        val_df = df.dropna(subset=[val_col])
        ax.plot(val_df['epoch'], val_df[val_col], '--', label='val', linewidth=1.8)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel('epoch')
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

def add_best_vlines(axes, df):
    if 'is_best' not in df.columns:
        return
    best_epochs = df.loc[df['is_best'] == 1, 'epoch'].tolist()
    for ax in axes.flat:
        for ep in best_epochs:
            ax.axvline(ep, color='red', alpha=0.12, linewidth=0.9)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log_dir', required=True,
                   help='runs/Fold1_Fold3_vs_Fold2 等 save_dir')
    p.add_argument('--save', default='training_curves.png',
                   help='输出图像文件名或绝对路径')
    p.add_argument('--dpi', type=int, default=180)
    args = p.parse_args()

    csv_path = os.path.join(args.log_dir, 'log.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'找不到 {csv_path}')

    df = pd.read_csv(csv_path)
    if 'epoch' not in df.columns:
        raise ValueError('log.csv 中缺少 epoch 列')

    # 画图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(os.path.basename(os.path.normpath(args.log_dir)), fontsize=15, y=0.98)

    # 1) Total loss
    plot_train_val(
        axes[0, 0], df,
        'train_loss', 'val_loss',
        title='Total Loss',
        ylabel='loss'
    )

    # 2) NP loss
    plot_train_val(
        axes[0, 1], df,
        'train_loss_np', 'val_loss_np',
        title='NP Loss',
        ylabel='loss'
    )

    # 3) HV loss
    plot_train_val(
        axes[0, 2], df,
        'train_loss_hv', 'val_loss_hv',
        title='HV Loss',
        ylabel='loss'
    )

    # 4) NC loss
    plot_train_val(
        axes[1, 0], df,
        'train_loss_nc', 'val_loss_nc',
        title='NC Loss',
        ylabel='loss'
    )

    # 5) NP IoU
    plot_train_val(
        axes[1, 1], df,
        'train_np_iou', 'val_np_iou',
        title='NP IoU',
        ylabel='iou'
    )

    # 6) Learning Rate
    ax = axes[1, 2]
    if 'lr_backbone' in df.columns:
        ax.plot(df['epoch'], df['lr_backbone'], label='backbone_lr', linewidth=1.8)
    if 'lr_decoder' in df.columns:
        ax.plot(df['epoch'], df['lr_decoder'], label='decoder_lr', linewidth=1.8)
    if 'lr' in df.columns:
        ax.plot(df['epoch'], df['lr'], label='lr', linewidth=1.8)

    ax.set_title('Learning Rate', fontsize=11)
    ax.set_xlabel('epoch')
    ax.set_ylabel('lr')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # best epoch 红线
    add_best_vlines(axes, df)

    plt.tight_layout()
    out_path = args.save if os.path.isabs(args.save) else os.path.join(args.log_dir, args.save)
    plt.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    print(f"saved → {out_path}")

if __name__ == '__main__':
    main()