#!/usr/bin/env bash
# train_3fold.sh
# 三折交叉验证：训练 → 评估 → 汇总
# 用法: bash train_3fold.sh [--epochs 100--batch_size 8 ...]
set -e

#── 可在此修改默认路径 ────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-/home/lwy/dataset/PanNuke/processed}"
SAVE_DIR="${SAVE_DIR:-./runs}"
SUMMARY_DIR="${SUMMARY_DIR:-./runs/summary}"

EXTRA_ARGS="$@"

VAL_FOLDS=("Fold1" "Fold2" "Fold3")

echo "=========================================="
echo "  HoverNet-YOLO  3-Fold Cross Validation"
echo "  data_root : ${DATA_ROOT}"
echo "  save_dir  : ${SAVE_DIR}"
echo "  Extra args: ${EXTRA_ARGS:-none}"
echo "=========================================="

mkdir -p "${SUMMARY_DIR}"
START_TIME=$(date +%s)

# ── 逐折训练 + 评估 ───────────────────────────────────────────
for i in "${!VAL_FOLDS[@]}"; do
    VAL=${VAL_FOLDS[$i]}
    FOLD_IDX=$((i + 1))

    echo ""
    echo "------------------------------------------"
    echo "  [${FOLD_IDX}/3] Training  val=${VAL}"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------"

    python train.py \
        --val_fold   "${VAL}" \
        --data_root  "${DATA_ROOT}" \
        --save_dir   "${SAVE_DIR}" \
        ${EXTRA_ARGS}

    echo ""
    echo "------------------------------------------"
    echo "  [${FOLD_IDX}/3] Evaluating  val=${VAL}"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------"

    #找到本折best.pt（train.py 保存在 runs/FoldX_FoldY_vs_FoldZ/best.pt）
    # 训练脚本把 val_fold 排除后，另外两折按字母序拼成目录名
    # 例: val=Fold2 → Fold1_Fold3_vs_Fold2
    ALL_FOLDS=("Fold1" "Fold2" "Fold3")
    TRAIN_FOLDS=()
    for f in "${ALL_FOLDS[@]}"; do
        [[ "$f" != "$VAL" ]] && TRAIN_FOLDS+=("$f")
    done
    TRAIN_STR=$(IFS='_'; echo "${TRAIN_FOLDS[*]}")
    CKPT="${SAVE_DIR}/${TRAIN_STR}_vs_${VAL}/best.pth"

    if [[ ! -f "${CKPT}" ]]; then
        echo "[ERROR] checkpoint not found: ${CKPT}"
        exit 1
    fi

    python evaluate.py \
        --ckpt          "${CKPT}" \
        --val_fold      "${VAL}" \
        --data_root     "${DATA_ROOT}" \
        --save_dir      "${SAVE_DIR}/${TRAIN_STR}_vs_${VAL}" \
        ${EXTRA_ARGS}

    echo "  [${FOLD_IDX}/3] done✓"
done

# ── 汇总 3折均值 ─────────────────────────────────────────────
echo ""
echo "------------------------------------------"
echo "  Aggregating 3-fold results..."
echo "------------------------------------------"

python -<<'PYEOF'
import os, json, sys
import numpy as np

save_dir    = os.environ.get('SAVE_DIR','./runs')
summary_dir = os.environ.get('SUMMARY_DIR', './runs/summary')
val_folds   = ['Fold1', 'Fold2', 'Fold3']
all_folds   = ['Fold1', 'Fold2', 'Fold3']

CLASS_NAMES = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
METRIC_KEYS = ['PQ', 'DQ', 'SQ', 'F1', 'Precision', 'Recall', 'cls_acc']

fold_metrics = []
for val in val_folds:
    train = [f for f in all_folds if f != val]
    train_str = '_'.join(train)
    json_path = os.path.join(save_dir, f'{train_str}_vs_{val}', f'metrics_{val}.json')
    if not os.path.exists(json_path):
        print(f'[WARN] not found: {json_path}', file=sys.stderr)
        continue
    with open(json_path) as f:
        fold_metrics.append((val, json.load(f)))

if not fold_metrics:
    print('[ERROR] no metrics found', file=sys.stderr)
    sys.exit(1)

#整体指标均值
summary = {}
for key in METRIC_KEYS:
    vals = [m[key] for _, m in fold_metrics]
    summary[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                    'per_fold': {v: float(m[key]) for v, m in fold_metrics}}

# per-class PQ 均值
per_cls_summary = {}
for cls in CLASS_NAMES:
    for metric in ['PQ', 'DQ', 'SQ']:
        vals = [m['per_class'][cls][metric] for _, m in fold_metrics]
        per_cls_summary.setdefault(cls, {})[metric] = {
            'mean': float(np.mean(vals)),
            'std':  float(np.std(vals)),
        }
summary['per_class'] = per_cls_summary

# 保存 json
os.makedirs(summary_dir, exist_ok=True)
json_out = os.path.join(summary_dir, '3fold_summary.json')
with open(json_out, 'w') as f:
    json.dump(summary, f, indent=2)

# 打印 + 保存 txt
sep = '=' * 65
lines = [sep,
         '  3-Fold Cross Validation Summary',
         sep]
for key in METRIC_KEYS:
    v = summary[key]
    fold_str = '  '.join(f"{fv}={v['per_fold'][fv]:.4f}" for fv in val_folds
                          if fv in v['per_fold'])
    lines.append(f"  {key:<12}: {v['mean']:.4f} ± {v['std']:.4f}({fold_str})")

lines += ['', 'Per-class PQ (mean ± std):', '-' * 65]
for cls in CLASS_NAMES:
    v = per_cls_summary[cls]
    lines.append(
        f"  {cls:<14}: "
        f"PQ={v['PQ']['mean']:.4f}±{v['PQ']['std']:.4f}  "
        f"DQ={v['DQ']['mean']:.4f}±{v['DQ']['std']:.4f}  "
        f"SQ={v['SQ']['mean']:.4f}±{v['SQ']['std']:.4f}"
    )
lines.append(sep)

txt_out = os.path.join(summary_dir, '3fold_summary.txt')
with open(txt_out, 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('\n'.join(lines))
print(f'\n[Saved] {json_out}')
print(f'[Saved] {txt_out}')
PYEOF

# ── 总耗时 ────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo "=========================================="
echo "  All3 folds finished"
echo "  Total time : ${HOURS}h ${MINS}m ${SECS}s"
echo "  Results: ${SAVE_DIR}/"
echo "  Summary    : ${SUMMARY_DIR}/3fold_summary.txt"
echo "=========================================="