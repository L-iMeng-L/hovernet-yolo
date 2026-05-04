#!/bin/bash

# 配置
DATA_ROOT="/home/lwy/dataset/PanNuke/processed"
CKPT_DIR="/home/lwy/hovernet-yolo/runs/"
RESULT_DIR="./3fold_results"
SEARCH_DIR="./bayesian_search_results"

# 创建结果目录
mkdir -p $RESULT_DIR
mkdir -p $SEARCH_DIR

# 定义3折配置
declare -A FOLD_CONFIG
FOLD_CONFIG["Fold1"]="Fold2_Fold3_vs_Fold1/best.pth"
FOLD_CONFIG["Fold2"]="Fold1_Fold3_vs_Fold2/best.pth"
FOLD_CONFIG["Fold3"]="Fold1_Fold2_vs_Fold3/best.pth"

echo "========================================="
echo "3-Fold Bayesian Search + Evaluation"
echo "========================================="

# 阶段1: 贝叶斯搜索
echo -e "\n[Stage 1] Bayesian Search on 3 Folds..."
for fold in Fold1 Fold2 Fold3; do
    ckpt="${CKPT_DIR}/${FOLD_CONFIG[$fold]}"
    echo -e "\n--- Searching on $fold ---"
    python bayesian_search.py \
        --ckpt "$ckpt" \
        --val_fold "$fold" \
        --data_root "$DATA_ROOT" \
        --batch_size 64 \
        --num_workers 16 \
        --n_calls 100 \
        --sample_size 1000 \
        --save_dir "$SEARCH_DIR"
done

# 阶段2: 使用最优参数评估
echo -e "\n[Stage 2] Evaluation with Best Params..."
for fold in Fold1 Fold2 Fold3; do
    ckpt="${CKPT_DIR}/${FOLD_CONFIG[$fold]}"
    params_json="${SEARCH_DIR}/best_params_${fold}.json"
    
    echo -e "\n--- Evaluating $fold ---"
    python evaluate.py \
        --ckpt "$ckpt" \
        --val_fold "$fold" \
        --data_root "$DATA_ROOT" \
        --batch_size 64 \
        --num_workers 16 \
        --params_json "$params_json" \
        --save_dir "$RESULT_DIR"
done

# 阶段3: 汇总结果
echo -e "\n[Stage 3] Aggregating Results..."
python aggregate_3fold.py \
    --result_dir "$RESULT_DIR" \
    --output "$RESULT_DIR/3fold_summary.txt"

echo -e "\n========================================="
echo "All Done! Results saved to: $RESULT_DIR"
echo "========================================="