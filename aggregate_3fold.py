# aggregate_3fold.py
import os
import json
import argparse
import numpy as np

CLASS_NAMES = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']

def safe_mean_std(values):
    values = np.array(values, dtype=np.float64)
    return float(np.nanmean(values)), float(np.nanstd(values))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--result_dir', required=True)
    p.add_argument('--output', default='3fold_summary.txt')
    args = p.parse_args()

    folds = ['Fold1', 'Fold2', 'Fold3']
    all_metrics = {}

    # 读取每折结果
    for fold in folds:
        json_path = os.path.join(args.result_dir, f'metrics_{fold}.json')
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found, skipping...")
            continue
        with open(json_path) as f:
            all_metrics[fold] = json.load(f)

    if len(all_metrics) == 0:
        print("No metrics found!")
        return

    fold_names = list(all_metrics.keys())

    # 计算均值和标准差
    metrics_keys = ['PQ', 'DQ', 'SQ', 'F1', 'Precision', 'Recall', 'cls_acc', 'global_cls_acc']
    summary = {}

    for key in metrics_keys:
        values = [all_metrics[fold].get(key, np.nan) for fold in fold_names]
        mean, std = safe_mean_std(values)
        summary[key] = {
            'mean': mean,
            'std': std,
            'values': values
        }

    # Per-class汇总
    per_class_summary = {}
    for cls_name in CLASS_NAMES:
        per_class_summary[cls_name] = {}
        for metric in ['PQ', 'DQ', 'SQ', 'F1', 'Precision', 'Recall']:
            values = []
            for fold in fold_names:
                v = all_metrics[fold].get('per_class', {}).get(cls_name, {}).get(metric, np.nan)
                values.append(v)
            mean, std = safe_mean_std(values)
            per_class_summary[cls_name][metric] = {
                'mean': mean,
                'std': std,
                'values': values
            }

    # 打印和保存
    output_lines = []
    sep = '=' * 70

    output_lines.append(sep)
    output_lines.append('3-Fold Cross-Validation Summary')
    output_lines.append(sep)
    output_lines.append(f'\nFolds: {", ".join(fold_names)}')
    output_lines.append(f'\nOverall Metrics (Mean ± Std):')
    output_lines.append('-' * 70)

    for key in metrics_keys:
        mean = summary[key]['mean']
        std = summary[key]['std']
        values_str = ', '.join([
            'nan' if np.isnan(v) else f'{v:.4f}' for v in summary[key]['values']
        ])
        output_lines.append(f'  {key:<15}: {mean:.4f} ± {std:.4f}  [{values_str}]')

    output_lines.append(f'\nPer-Class Metrics (Mean ± Std):')
    output_lines.append('-' * 70)
    for cls_name in CLASS_NAMES:
        line = f'  {cls_name:<14}: '
        for metric in ['PQ', 'DQ', 'SQ', 'F1', 'Precision', 'Recall']:
            mean = per_class_summary[cls_name][metric]['mean']
            std = per_class_summary[cls_name][metric]['std']
            line += f'{metric}={mean:.4f}±{std:.4f}  '
        output_lines.append(line.strip())

    output_lines.append(sep)

    # 打印到终端
    for line in output_lines:
        print(line)

    # 保存到文件
    with open(args.output, 'w') as f:
        f.write('\n'.join(output_lines))

    print(f'\n[Saved] {args.output}')

    # 保存json
    json_output = args.output.replace('.txt', '.json')
    with open(json_output, 'w') as f:
        json.dump({
            'overall': summary,
            'per_class': per_class_summary,
            'fold_details': all_metrics
        }, f, indent=2)

    print(f'[Saved] {json_output}')

if __name__ == '__main__':
    main()