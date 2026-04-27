# plot_curves.py
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log_dir', required=True)
    p.add_argument('--save',    default='training_curves.png')
    args = p.parse_args()

    df= pd.read_csv(os.path.join(args.log_dir, 'log.csv'))
    val = df.dropna(subset=['val_loss'])

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle(os.path.basename(args.log_dir), fontsize=14)

    def plot(ax, col_train, col_val=None, title=''):
        ax.plot(df['epoch'], df[col_train], label='train')
        if col_val and col_val in val.columns:
            ax.plot(val['epoch'], val[col_val], '--', label='val')
        ax.set_title(title)
        ax.set_xlabel('epoch')
        ax.legend()
        ax.grid(alpha=0.3)

    plot(axes[0,0], 'train_loss','val_loss',     'Total Loss')
    plot(axes[0,1], 'train_loss_box', 'val_loss_box', 'Box Loss')
    plot(axes[0,2], 'train_loss_cls', 'val_loss_cls', 'Cls Loss')
    plot(axes[0,3], 'train_loss_np',  'val_loss_np',  'NP Loss')
    plot(axes[1,0], 'train_loss_hv',  'val_loss_hv',  'HV Loss')
    plot(axes[1,1], 'train_iou',      'val_iou',      'IoU')
    plot(axes[1,2], 'train_np_iou',   'val_np_iou',   'NP IoU')

    # lr 单独画
    axes[1,3].plot(df['epoch'], df['lr'])
    axes[1,3].set_title('Learning Rate')
    axes[1,3].set_xlabel('epoch')
    axes[1,3].grid(alpha=0.3)

    # best epoch标记
    best_rows = df[df['is_best'] == 1]
    for ax in axes.flat:
        for ep in best_rows['epoch']:
            ax.axvline(ep, color='red', alpha=0.15, linewidth=0.8)

    plt.tight_layout()
    plt.savefig(args.save, dpi=150, bbox_inches='tight')
    print(f"saved → {args.save}")

if __name__ == '__main__':
    #使用实例python plot_curves.py \--log_dir ./runs/Fold1_Fold2_vs_Fold3 \--save    training_curves.png
    main()