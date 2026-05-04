import os
import json
import argparse
import numpy as np

CLASS_NAMES = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']

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
    
    # 计算均值和标准差
    metrics_keys = ['PQ', 'DQ', 'SQ', 'F1', 'Precision', 'Recall', 'cls_acc']
    summary = {}
    
    for key in metrics_keys:
        values = [all_metrics[fold][key] for fold in all_metrics.keys()]
        summary[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
    
    # Per-class汇总
    per_class_summary = {}
    for cls_name in CLASS_NAMES:
        per_class_summary[cls_name] = {}
        for metric in ['PQ', 'DQ', 'SQ']:
            values = [all_metrics[fold]['per_class'][cls_name][metric] 
                     for fold in all_metrics.keys()]
            per_class_summary[cls_name][metric] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
    
    # 打印和保存
    output_lines = []
    sep = '=' * 70
    
    output_lines.append(sep)
    output_lines.append('3-Fold Cross-Validation Summary')
    output_lines.append(sep)
    output_lines.append(f'\nFolds: {", ".join(all_metrics.keys())}')
    output_lines.append(f'\nOverall Metrics (Mean ± Std):')
    output_lines.append('-' * 70)
    
    for key in metrics_keys:
        mean = summary[key]['mean']
        std = summary[key]['std']
        values_str = ', '.join([f'{v:.4f}' for v in summary[key]['values']])
        output_lines.append(f'  {key:<12}: {mean:.4f} ± {std:.4f}  [{values_str}]')
    
    output_lines.append(f'\nPer-Class PQ (Mean ± Std):')
    output_lines.append('-' * 70)
    for cls_name in CLASS_NAMES:
        pq_mean = per_class_summary[cls_name]['PQ']['mean']
        pq_std = per_class_summary[cls_name]['PQ']['std']
        dq_mean = per_class_summary[cls_name]['DQ']['mean']
        sq_mean = per_class_summary[cls_name]['SQ']['mean']
        output_lines.append(
            f'  {cls_name:<14}: PQ={pq_mean:.4f}±{pq_std:.4f}  '
            f'DQ={dq_mean:.4f}  SQ={sq_mean:.4f}'
        )
    
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