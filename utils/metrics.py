# utils/metrics.py
import numpy as np
from scipy.optimize import linear_sum_assignment
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def _remap_label(inst):
    """将实例 id 重映射为连续整数，返回 remap 后 mask 和 old->new 映射"""
    ids = [i for i in np.unique(inst) if i != 0]
    out = np.zeros_like(inst, dtype=np.int32)
    old2new = {}
    for new_id, old_id in enumerate(ids, 1):
        out[inst == old_id] = new_id
        old2new[int(old_id)] = int(new_id)
    return out, old2new

def _build_cls_maps(cls_dict, old2new):
    """将原始 instance id -> class id 映射到 remap 后的 instance id"""
    if cls_dict is None:
        return None
    out = {}
    for old_id, cls_id in cls_dict.items():
        if old_id in old2new:
            out[old2new[old_id]] = int(cls_id)
    return out

def _compute_iou_matrix(true_inst, pred_inst):
    true_ids = [i for i in np.unique(true_inst) if i != 0]
    pred_ids = [i for i in np.unique(pred_inst) if i != 0]

    iou_mat = np.zeros((len(true_ids), len(pred_ids)), dtype=np.float64)

    true_masks = [true_inst == t for t in true_ids]
    pred_masks = [pred_inst == p for p in pred_ids]

    for i, tm in enumerate(true_masks):
        for j, pm in enumerate(pred_masks):
            inter = np.logical_and(tm, pm).sum()
            union = np.logical_or(tm, pm).sum()
            iou_mat[i, j] = inter / (union + 1e-8)

    return iou_mat, true_ids, pred_ids

def match_instances(true_inst, pred_inst, match_iou=0.5):
    """
    对单张图做实例匹配，不考虑类别。
    返回:
        tp, fp, fn, iou_sum, matched_pairs
    其中 matched_pairs: [(true_id, pred_id, iou), ...]
    """
    true_inst, _ = _remap_label(true_inst)
    pred_inst, _ = _remap_label(pred_inst)

    true_ids = [i for i in np.unique(true_inst) if i != 0]
    pred_ids = [i for i in np.unique(pred_inst) if i != 0]

    n_true = len(true_ids)
    n_pred = len(pred_ids)

    if n_true == 0 and n_pred == 0:
        return 0, 0, 0, 0.0, []
    if n_true == 0:
        return 0, n_pred, 0, 0.0, []
    if n_pred == 0:
        return 0, 0, n_true, 0.0, []

    iou_mat, true_ids, pred_ids = _compute_iou_matrix(true_inst, pred_inst)

    ri, ci = linear_sum_assignment(-iou_mat)
    valid = iou_mat[ri, ci] > match_iou
    ri, ci = ri[valid], ci[valid]

    matched_ious = iou_mat[ri, ci]
    tp = len(matched_ious)
    fp = n_pred - tp
    fn = n_true - tp
    iou_sum = float(matched_ious.sum())

    matched_pairs = [(int(true_ids[t]), int(pred_ids[p]), float(iou_mat[t, p]))
                     for t, p in zip(ri, ci)]

    return tp, fp, fn, iou_sum, matched_pairs

def match_instances_classwise(true_inst, pred_inst, true_cls=None, pred_cls=None,
                              class_id=0, match_iou=0.5):
    """
    对单张图、单个类别做严格匹配。
    返回:
        tp, fp, fn, iou_sum, matched_pairs
    matched_pairs: [(true_id, pred_id, iou), ...]，这里的 id 是过滤后的局部 id
    """
    true_inst, true_old2new = _remap_label(true_inst)
    pred_inst, pred_old2new = _remap_label(pred_inst)

    true_cls_r = _build_cls_maps(true_cls, true_old2new)
    pred_cls_r = _build_cls_maps(pred_cls, pred_old2new)

    true_ids = [i for i in np.unique(true_inst) if i != 0]
    pred_ids = [i for i in np.unique(pred_inst) if i != 0]

    true_keep = [i for i in true_ids if true_cls_r is not None and true_cls_r.get(i, -1) == class_id]
    pred_keep = [i for i in pred_ids if pred_cls_r is not None and pred_cls_r.get(i, -1) == class_id]

    n_true = len(true_keep)
    n_pred = len(pred_keep)

    if n_true == 0 and n_pred == 0:
        return 0, 0, 0, 0.0, []
    if n_true == 0:
        return 0, n_pred, 0, 0.0, []
    if n_pred == 0:
        return 0, 0, n_true, 0.0, []

    iou_mat = np.zeros((n_true, n_pred), dtype=np.float64)
    true_masks = [true_inst == t for t in true_keep]
    pred_masks = [pred_inst == p for p in pred_keep]

    for i, tm in enumerate(true_masks):
        for j, pm in enumerate(pred_masks):
            inter = np.logical_and(tm, pm).sum()
            union = np.logical_or(tm, pm).sum()
            iou_mat[i, j] = inter / (union + 1e-8)

    ri, ci = linear_sum_assignment(-iou_mat)
    valid = iou_mat[ri, ci] > match_iou
    ri, ci = ri[valid], ci[valid]

    matched_ious = iou_mat[ri, ci]
    tp = len(matched_ious)
    fp = n_pred - tp
    fn = n_true - tp
    iou_sum = float(matched_ious.sum())

    matched_pairs = [(int(true_keep[t]), int(pred_keep[p]), float(iou_mat[t, p]))
                     for t, p in zip(ri, ci)]

    return tp, fp, fn, iou_sum, matched_pairs

def _compute_single_sample(args):
    pred, true, p_cls, t_cls, match_iou = args
    return match_instances(true, pred, match_iou)

def batch_seg_metrics(pred_inst_list, true_inst_list,
                      pred_cls_list=None, true_cls_list=None,
                      match_iou=0.5, num_workers=None):
    """
    数据集级别整体指标：累计 TP/FP/FN/IoU_sum 后统一计算
    这里 cls_acc 返回“全局 matched pairs 上的分类准确率”
    """
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    args_list = []
    for i, (pred, true) in enumerate(zip(pred_inst_list, true_inst_list)):
        p_cls = pred_cls_list[i] if pred_cls_list is not None else None
        t_cls = true_cls_list[i] if true_cls_list is not None else None
        args_list.append((pred, true, p_cls, t_cls, match_iou))

    if num_workers > 1 and len(args_list) > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_compute_single_sample, args_list))
    else:
        results = [_compute_single_sample(args) for args in args_list]

    TP = sum(r[0] for r in results)
    FP = sum(r[1] for r in results)
    FN = sum(r[2] for r in results)
    IoU_sum = sum(r[3] for r in results)

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    dq = TP / (TP + 0.5 * FP + 0.5 * FN + 1e-8)
    sq = IoU_sum / (TP + 1e-8) if TP > 0 else 0.0
    pq = IoU_sum / (TP + 0.5 * FP + 0.5 * FN + 1e-8)

    return dict(
        TP=int(TP),
        FP=int(FP),
        FN=int(FN),
        IoU_sum=float(IoU_sum),
        PQ=float(pq),
        DQ=float(dq),
        SQ=float(sq),
        F1=float(f1),
        Precision=float(precision),
        Recall=float(recall),
    )