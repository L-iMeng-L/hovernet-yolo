# utils/metrics.py
import numpy as np
from scipy.optimize import linear_sum_assignment

def remap_label(pred):
    """重新映射实例ID为连续整数 [1, 2, 3, ...]"""
    pred_id = list(np.unique(pred))
    if 0 in pred_id:
        pred_id.remove(0)
    if len(pred_id) == 0:
        return pred
    
    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred

def get_fast_pq(true, pred, match_iou=0.5):
    """
    完全对齐HoverNet官方实现
    
    Returns:
        [dq, sq, pq]: 三个指标
        [paired_true, paired_pred, unpaired_true, unpaired_pred]: 配对信息（都是list）
    """
    assert match_iou >= 0.0, "Can't be negative"
    
    true = np.copy(true)
    pred = np.copy(pred)
    
    true = remap_label(true)
    pred = remap_label(pred)
    
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))
    
    # 构建mask缓存
    true_masks = [None]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)
    
    pred_masks = [None]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)
    
    # 计算pairwise IoU
    pairwise_iou = np.zeros([len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64)
    
    for true_id in true_id_list[1:]:
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:
                continue
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou
    
    # 匹配策略
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1
        paired_pred += 1
        # 转为list（统一返回类型）
        paired_true = list(paired_true)
        paired_pred = list(paired_pred)
    else:
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]
    
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    
    dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1.0e-6)
    sq = paired_iou.sum() / (tp + 1.0e-6)
    pq = dq * sq
    
    return [dq, sq, pq], [paired_true, paired_pred, unpaired_true, unpaired_pred]

def batch_binary_metrics(pred_inst_list, true_inst_list, match_iou=0.5, num_workers=None):
    """
    完全对齐官方run.py的二值PQ计算流程
    """
    pq_list = []
    dq_list = []
    sq_list = []
    
    TP_total = 0
    FP_total = 0
    FN_total = 0
    IoU_sum_total = 0.0
    
    for pred_inst, true_inst in zip(pred_inst_list, true_inst_list):
        pred_bin = remap_label(pred_inst)
        true_bin = remap_label(true_inst)
        
        # 空GT时append nan（你的方式更好）
        if len(np.unique(true_bin)) == 1:
            pq_list.append(np.nan)
            dq_list.append(np.nan)
            sq_list.append(np.nan)
            continue
        
        [dq, sq, pq], [paired_true, paired_pred, unpaired_true, unpaired_pred] = get_fast_pq(
            true_bin, pred_bin, match_iou
        )
        
        pq_list.append(pq)
        dq_list.append(dq)
        sq_list.append(sq)
        
        # 累积全局统计
        tp = len(paired_true)
        fp = len(unpaired_pred)
        fn = len(unpaired_true)
        
        TP_total += tp
        FP_total += fp
        FN_total += fn
        IoU_sum_total += sq * tp
    
    # Per-image平均
    PQb = np.nanmean(pq_list) if len(pq_list) > 0 else 0.0
    DQb = np.nanmean(dq_list) if len(dq_list) > 0 else 0.0
    SQb = np.nanmean(sq_list) if len(sq_list) > 0 else 0.0
    
    # F1-Detection（正确公式）
    Fd = 2 * TP_total / (2 * TP_total + FP_total + FN_total + 1e-10) if (TP_total + FP_total + FN_total) > 0 else 0.0
    
    Precision_b = TP_total / (TP_total + FP_total + 1e-10) if (TP_total + FP_total) > 0 else 0.0
    Recall_b = TP_total / (TP_total + FN_total + 1e-10) if (TP_total + FN_total) > 0 else 0.0
    
    return {
        'TPb': int(TP_total),
        'FPb': int(FP_total),
        'FNb': int(FN_total),
        'IoU_sum_b': float(IoU_sum_total),
        'PQb': float(PQb),
        'DQb': float(DQb),
        'SQb': float(SQb),
        'Fd': float(Fd),
        'Precision_b': float(Precision_b),
        'Recall_b': float(Recall_b),
    }

def batch_multiclass_metrics(pred_inst_list, true_inst_list,
                             pred_cls_list, true_cls_list,
                             num_classes, match_iou=0.5):
    """
    完全对齐官方run.py的多类PQ计算流程
    """
    per_class = {}
    
    for cls_id in range(num_classes):
        pq_list = []
        dq_list = []
        sq_list = []
        
        TP_total = 0
        FP_total = 0
        FN_total = 0
        IoU_sum_total = 0.0
        
        for pred_inst, true_inst, pred_cls, true_cls in zip(
            pred_inst_list, true_inst_list, pred_cls_list, true_cls_list
        ):
            # 提取该类别的实例
            pred_tmp = np.zeros_like(pred_inst, dtype=np.int32)
            true_tmp = np.zeros_like(true_inst, dtype=np.int32)
            
            for inst_id, c in pred_cls.items():
                if c == cls_id:
                    pred_tmp[pred_inst == inst_id] = inst_id
            
            for inst_id, c in true_cls.items():
                if c == cls_id:
                    true_tmp[true_inst == inst_id] = inst_id
            
            pred_tmp = remap_label(pred_tmp)
            true_tmp = remap_label(true_tmp)
            
            # 空GT时append nan
            if len(np.unique(true_tmp)) == 1:
                pq_list.append(np.nan)
                dq_list.append(np.nan)
                sq_list.append(np.nan)
                continue
            
            [dq, sq, pq], [paired_true, paired_pred, unpaired_true, unpaired_pred] = get_fast_pq(
                true_tmp, pred_tmp, match_iou
            )
            
            pq_list.append(pq)
            dq_list.append(dq)
            sq_list.append(sq)
            
            # 累积全局统计
            tp = len(paired_true)
            fp = len(unpaired_pred)
            fn = len(unpaired_true)
            
            TP_total += tp
            FP_total += fp
            FN_total += fn
            IoU_sum_total += sq * tp
        
        # Per-image平均（官方方法）
        PQ_class = np.nanmean(pq_list) if len(pq_list) > 0 else 0.0
        DQ_class = np.nanmean(dq_list) if len(dq_list) > 0 else 0.0
        SQ_class = np.nanmean(sq_list) if len(sq_list) > 0 else 0.0
        
        # 全局统计（补充）
        F1_class = 2 * TP_total / (2 * TP_total + FP_total + FN_total + 1e-10) if (TP_total + FP_total + FN_total) > 0 else 0.0
        Precision_class = TP_total / (TP_total + FP_total + 1e-10) if (TP_total + FP_total) > 0 else 0.0
        Recall_class = TP_total / (TP_total + FN_total + 1e-10) if (TP_total + FN_total) > 0 else 0.0
        
        per_class[cls_id] = {
            'TP': int(TP_total),
            'FP': int(FP_total),
            'FN': int(FN_total),
            'IoU_sum': float(IoU_sum_total),
            'PQ': float(PQ_class),
            'DQ': float(DQ_class),  # ← 改为per-image平均
            'SQ': float(SQ_class),  # ← 改为per-image平均
            'F1': float(F1_class),
            'Precision': float(Precision_class),
            'Recall': float(Recall_class),
        }
    
    # mPQ: 5个类别PQ的平均（官方定义）
    PQM = np.nanmean([per_class[c]['PQ'] for c in range(num_classes)])
    DQm = np.nanmean([per_class[c]['DQ'] for c in range(num_classes)])
    SQm = np.nanmean([per_class[c]['SQ'] for c in range(num_classes)])
    
    # 全局统计
    TPm = sum(per_class[c]['TP'] for c in range(num_classes))
    FPm = sum(per_class[c]['FP'] for c in range(num_classes))
    FNm = sum(per_class[c]['FN'] for c in range(num_classes))
    IoU_sum_m = sum(per_class[c]['IoU_sum'] for c in range(num_classes))
    
    F1m = 2 * TPm / (2 * TPm + FPm + FNm + 1e-10) if (TPm + FPm + FNm) > 0 else 0.0
    Precision_m = TPm / (TPm + FPm + 1e-10) if (TPm + FPm) > 0 else 0.0
    Recall_m = TPm / (TPm + FNm + 1e-10) if (TPm + FNm) > 0 else 0.0
    
    return {
        'TPm': int(TPm),
        'FPm': int(FPm),
        'FNm': int(FNm),
        'IoU_sum_m': float(IoU_sum_m),
        'PQM': float(PQM),
        'DQm': float(DQm),
        'SQm': float(SQm),
        'F1m': float(F1m),
        'Precision_m': float(Precision_m),
        'Recall_m': float(Recall_m),
        'per_class': per_class
    }

def match_instances(true_inst, pred_inst, match_iou=0.5):
    """用于混淆矩阵的实例匹配"""
    true_inst = remap_label(true_inst)
    pred_inst = remap_label(pred_inst)
    
    [dq, sq, pq], [paired_true, paired_pred, unpaired_true, unpaired_pred] = get_fast_pq(
        true_inst, pred_inst, match_iou
    )
    
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    iou_sum = sq * tp
    
    matched_pairs = [(int(t), int(p), 0.0) for t, p in zip(paired_true, paired_pred)]
    
    return tp, fp, fn, iou_sum, matched_pairs