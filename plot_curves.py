# plot_curves.py
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log_dir', required=True,
                   help='runs/Fold1_Fold3_vs_Fold2 等save_dir')
    p.add_argument('--save', default='training_curves.png')
    args = p.parse_args()

    csv_path = os.path.join(args.log_dir, 'log.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 {csv_path}")

    df  = pd.read_csv(csv_path)
    val = df.dropna(subset=['val_loss'])

    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    fig.suptitle(os.path.basename(args.log_dir), fontsize=14, y=1.01)

    def plot(ax, col_train, col_val=None, title=''):
        if col_train in df.columns:
            ax.plot(df['epoch'], df[col_train], label='train')
        if col_val and col_val in val.columns:
            ax.plot(val['epoch'], val[col_val], '--', label='val')
        ax.set_title(title)
        ax.set_xlabel('epoch')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # 行0
    plot(axes[0, 0], 'train_loss','val_loss',    'Total Loss')
    plot(axes[0, 1], 'train_loss_np', 'val_loss_np', 'NP Loss')
    plot(axes[0, 2], 'train_loss_hv', 'val_loss_hv', 'HV Loss')
    plot(axes[0, 3], 'train_loss_nc', 'val_loss_nc', 'NC Loss')
    # 行1
    plot(axes[1, 0], 'train_np_iou','val_np_iou',  'NP IoU')
    plot(axes[1, 1], 'train_np_iou',  'val_iou',     'IoU (train_np / val)')

    # lr单独画
    axes[1, 2].plot(df['epoch'], df['lr'], color='orange')
    axes[1, 2].set_title('Learning Rate')
    axes[1, 2].set_xlabel('epoch')
    axes[1, 2].grid(alpha=0.3)

    # loss汇总（train vs val 对比）
    ax = axes[1, 3]
    ax.plot(df['epoch'],df['train_loss'], label='train_loss')
    ax.plot(val['epoch'], val['val_loss'],'--', label='val_loss')
    ax.set_title('Loss Overview')
    ax.set_xlabel('epoch')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # best epoch红线
    if 'is_best' in df.columns:
        best_epochs = df[df['is_best'] == 1]['epoch']
        for ax in axes.flat:
            for ep in best_epochs:
                ax.axvline(ep, color='red', alpha=0.15, linewidth=0.8)

    plt.tight_layout()
    out_path = args.save if os.path.isabs(args.save) else os.path.join(args.log_dir, args.save)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"saved → {out_path}")

if __name__ == '__main__':
    # 使用示例：
    # python plot_curves.py \
    #   --log_dir ./runs/Fold1_Fold2_vs_Fold3 \
    #   --save training_curves.png
    main()