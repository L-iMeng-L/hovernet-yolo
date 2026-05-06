set -e

DATA_ROOT="${DATA_ROOT:-/home/lwy/dataset/PanNuke/processed}"
SAVE_DIR="${SAVE_DIR:-./runs/third_try}"
SUMMARY_DIR="${SUMMARY_DIR:-./runs/third_try/summary}"

VAL_FOLDS=("Fold1" "Fold2" "Fold3")

echo "=========================================="
echo "  HoverNet-YOLO  3-Fold Cross Validation"
echo "  data_root    : ${DATA_ROOT}"
echo "  save_dir     : ${SAVE_DIR}"
echo "=========================================="

mkdir -p "${SUMMARY_DIR}"
mkdir -p "${SAVE_DIR}"

START_TIME=$(date +%s)

for i in "${!VAL_FOLDS[@]}"; do
    VAL="${VAL_FOLDS[$i]}"
    FOLD_IDX=$((i + 1))
    FOLD_SAVE_DIR="${SAVE_DIR}/${VAL}"

    echo ""
    echo "=========================================="
    echo "  [${FOLD_IDX}/3] Training  val=${VAL}"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  fold_save_dir: ${FOLD_SAVE_DIR}"
    echo "=========================================="

    mkdir -p "${FOLD_SAVE_DIR}"

    python train.py \
        --val_fold "${VAL}" \
        --data_root "${DATA_ROOT}" \
        --save_dir "${FOLD_SAVE_DIR}" \
        "$@"

    echo "  [${FOLD_IDX}/3] Finished  val=${VAL}"
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "  3-Fold Cross Validation Finished"
echo "  Total time: ${DURATION} seconds"
echo "  Results saved in: ${SAVE_DIR}"
echo "=========================================="