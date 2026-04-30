#!/usr/bin/env bash
set -e

DATA_ROOT="${DATA_ROOT:-/home/lwy/dataset/PanNuke/processed}"
SAVE_DIR="${SAVE_DIR:-./runs}"
SUMMARY_DIR="${SUMMARY_DIR:-./runs/summary}"
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

    # ── 准备路径 ──────────────────────────────────────────
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

    # ── 参数搜索 ──────────────────────────────────────────
    if [[ "${ENABLE_SEARCH}" == "true" ]]; then
        echo ""
        echo "=========================================="
        echo "  [${FOLD_IDX}/3] Param Search  val=${VAL}"
        echo "  $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="

        python bayesian_search.py \
            --ckpt        "${CKPT}" \
            --val_fold    "${VAL}" \
            --data_root   "${DATA_ROOT}" \
            --n_calls     300 \
            --sample_size 200 \
            --save_dir    "${SAVE_DIR}/${TRAIN_STR}_vs_${VAL}"

        # 读取最优参数
        BEST_JSON="${SAVE_DIR}/${TRAIN_STR}_vs_${VAL}/best_params_${VAL}.json"
        if [[ ! -f "${BEST_JSON}" ]]; then
            echo "[WARN] best_params not found, using defaults"
            NP_T=0.32; OV_T=0.4; MK_K=3; MIN_A=3
        else
            read NP_T OV_T _ MK_K MIN_A <<< $(python3 -c "
import json
with open('${BEST_JSON}') as f:
    p = json.load(f)['best_params']
print(p['np_thresh'], p['overall_thresh'], p['ksize'], p['marker_ksize'], p['min_area'])
")
        fi
    else
        NP_T=0.32; OV_T=0.4; MK_K=3; MIN_A=3
    fi

    # ── 评估 ──────────────────────────────────────────────
    echo ""
    echo "=========================================="
    echo "  [${FOLD_IDX}/3] Evaluating  val=${VAL}"
    echo "  Params: np=${NP_T} ov=${OV_T} mk=${MK_K} ma=${MIN_A}"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    python evaluate.py \
        --ckpt           "${CKPT}" \
        --val_fold       "${VAL}" \
        --data_root      "${DATA_ROOT}" \
        --save_dir       "${SAVE_DIR}/${TRAIN_STR}_vs_${VAL}" \
        --np_thresh      "${NP_T}" \
        --overall_thresh "${OV_T}" \
        --marker_ksize   "${MK_K}" \
        --min_area       "${MIN_A}" \
        ${EXTRA_ARGS}

    echo "  [${FOLD_IDX}/3] ✓ Done"
done

# ── 三折汇总 ──────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Summarizing 3-Fold Results"
echo "=========================================="

python3 << 'PYEOF'
import os, json, glob

save_dir = os.environ.get('SAVE_DIR', './runs')
summary_dir = os.environ.get('SUMMARY_DIR', './runs/summary')
os.makedirs(summary_dir, exist_ok=True)

results = []
for fold in ['Fold1', 'Fold2', 'Fold3']:
    pattern = f"{save_dir}/*_vs_{fold}/metrics_{fold}.json"
    files = glob.glob(pattern)
    if files:
        with open(files[0]) as f:
            data = json.load(f)
            results.append({'fold': fold, **data})

if not results:
    print("[WARN] No metrics found")
    exit(0)

keys = ['PQ', 'DQ', 'SQ', 'F1', 'Precision', 'Recall', 'cls_acc']
avg = {k: sum(r[k] for r in results) / len(results) for k in keys}

summary = {'per_fold': results, 'average': avg}

with open(f"{summary_dir}/3fold_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

with open(f"{summary_dir}/3fold_summary.txt", 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("  3-Fold Cross Validation Summary\n")
    f.write("=" * 60 + "\n\n")
    for r in results:
        f.write(f"{r['fold']}:\n")
        for k in keys:
            f.write(f"  {k:12s}: {r[k]:.4f}\n")
        f.write("\n")
    f.write("-" * 60 + "\n")
    f.write("Average:\n")
    for k in keys:
        f.write(f"  {k:12s}: {avg[k]:.4f}\n")
    f.write("=" * 60 + "\n")

print(f"\nSummary saved to {summary_dir}/")
for k in keys:
    print(f"  {k:12s}: {avg[k]:.4f}")
PYEOF

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo "=========================================="
echo "  All 3 folds finished"
echo "  Total time: ${HOURS}h ${MINS}m ${SECS}s"
echo "  Results   : ${SAVE_DIR}/"
echo "  Summary   : ${SUMMARY_DIR}/3fold_summary.txt"
echo "=========================================="