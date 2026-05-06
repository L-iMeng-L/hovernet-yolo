#!/bin/bash

DATA_ROOT="/home/lwy/dataset/PanNuke/processed"
CKPT_DIR="/home/lwy/hovernet-yolo/runs/"
RESULT_DIR="./3fold_results"

mkdir -p "$RESULT_DIR"

declare -A FOLD_CONFIG
FOLD_CONFIG["Fold1"]="Fold2_Fold3_vs_Fold1/best.pth"
FOLD_CONFIG["Fold2"]="Fold1_Fold3_vs_Fold2/best.pth"
FOLD_CONFIG["Fold3"]="Fold1_Fold2_vs_Fold3/best.pth"

echo "========================================="
echo "3-Fold Evaluation (Paper-style metrics)"
echo "========================================="

NP_THRESH=0.4
OVERALL_THRESH=0.7
KSIZE=19
MARKER_KSIZE=7
MIN_AREA=5
MATCH_IOU=0.5

for fold in Fold1 Fold2 Fold3; do
    ckpt="${CKPT_DIR}/${FOLD_CONFIG[$fold]}"
    echo -e "\n--- Evaluating $fold ---"
    python evaluate.py \
        --ckpt "$ckpt" \
        --val_fold "$fold" \
        --data_root "$DATA_ROOT" \
        --batch_size 64 \
        --num_workers 16 \
        --np_thresh "$NP_THRESH" \
        --overall_thresh "$OVERALL_THRESH" \
        --ksize "$KSIZE" \
        --marker_ksize "$MARKER_KSIZE" \
        --min_area "$MIN_AREA" \
        --match_iou "$MATCH_IOU" \
        --save_dir "$RESULT_DIR"
done

echo -e "\n[Stage 2] Aggregating Results..."
python aggregate_3fold.py \
    --result_dir "$RESULT_DIR" \
    --output "$RESULT_DIR/3fold_summary.txt"

echo -e "\nDone. Results saved to: $RESULT_DIR"