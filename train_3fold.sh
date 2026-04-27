# train_3fold.sh
# 三折交叉验证：每次用2折训练，1折验证
# 用法: bash train_3fold.sh [--epochs 100 --batch_size 8 ...]

set -e

EXTRA_ARGS="$@"

# val_fold 依次轮换，train自动取另外两折
VAL_FOLDS=("Fold1" "Fold2" "Fold3")

echo "=========================================="
echo "HoverNet-YOLO  3-Fold Cross Validation"
echo "  train = the other two folds"
echo "  Extra args: ${EXTRA_ARGS:-none}"
echo "=========================================="

START_TIME=$(date +%s)

for i in "${!VAL_FOLDS[@]}"; do
    VAL=${VAL_FOLDS[$i]}
    FOLD_IDX=$((i + 1))

    echo ""
    echo "------------------------------------------"
    echo "  Fold ${FOLD_IDX}/3 | val=${VAL}  train=other two"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------"

    python train.py \
        --val_fold "$VAL" \
        $EXTRA_ARGS

    echo "[Fold ${FOLD_IDX}] done ✓"
done

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo "=========================================="
echo "  All 3 folds finished"
echo "  Total time: ${HOURS}h ${MINS}m ${SECS}s"
echo "  Results in: ./runs/"
echo "=========================================="