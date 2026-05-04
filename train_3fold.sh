#!/usr/bin/env bash
set -e

DATA_ROOT="${DATA_ROOT:-/home/lwy/dataset/PanNuke/processed}"
SAVE_DIR="${SAVE_DIR:-./runs/second_try}"
SUMMARY_DIR="${SUMMARY_DIR:-./runs/second_try/summary}"
ENABLE_SEARCH="${ENABLE_SEARCH:-true}"

EXTRA_ARGS="$@"
VAL_FOLDS=("Fold1" "Fold2" "Fold3")

echo "=========================================="
echo "  HoverNet-YOLO  3-Fold Cross Validation"
echo "  data_root    : ${DATA_ROOT}"
echo "  save_dir     : ${SAVE_DIR}"
echo "  param_search : ${ENABLE_SEARCH}"
echo "  Extra args   : ${EXTRA_ARGS:-none}"
echo "=========================================="

mkdir -p "${SUMMARY_DIR}"
START_TIME=$(date +%s)

for i in "${!VAL_FOLDS[@]}"; do
    VAL=${VAL_FOLDS[$i]}
    FOLD_IDX=$((i + 1))

    # ── 训练 ──────────────────────────────────────────────
    echo ""
    echo "=========================================="
    echo "  [${FOLD_IDX}/3] Training  val=${VAL}"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    python train.py \
        --val_fold   "${VAL}" \
        --data_root  "${DATA_ROOT}" \
        --save_dir   "${SAVE_DIR}" \
        ${EXTRA_ARGS}
done