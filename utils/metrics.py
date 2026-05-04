# utils/metrics.py
import numpy as np
from scipy.optimize import linear_sum_assignment
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import torch

@torch.no_grad()
def compute_np_iou(hover_pred, hover_gt, thresh=0.5):
    pred = (hover_pred['np_map'] > thresh).float()
    gt = (hover_gt['np_map'] > thresh).float()
    inter = (pred * gt).sum()
    union = (pred + gt).clamp(0, 1).sum()
    return (inter / (union + 1e-8)).item()


def _remap_label(inst):
    """将实例 id 重映射为连续整数，返回 remap 后的 mask 和 old->new 映射"""
    ids = [i for i in np.unique(inst) if i != 0]
    out = np.zeros_like(inst, dtype=np.int32)
    old2new = {}
    for new_id, old_id in enumerate(ids, 1):
        out[inst == old_id] = new_id
        old2new[int(old_id)] = int(new_id)
    return out, old2new

def _match_instances(true_inst, pred_inst, match_iou=0.5):
    """
    对单张图进行实例匹配，返回匹配结果和统计量。

    返回:
        tp, fp, fn, iou_sum, matched_pairs
    """
    true_inst, true_old2new = _remap_label(true_inst)
    pred_inst, pred_old2new = _remap_label(pred_inst)

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

    iou_mat = np.zeros((n_true, n_pred), dtype=np.float64)

    true_masks = [true_inst == t for t in true_ids]
    pred_masks = [pred_inst == p for p in pred_ids]

    for i, tm in enumerate(true_masks):
        for j, pm in enumerate(pred_masks):
            inter = np.logical_and(tm, pm).sum()
            union = np.logical_or(tm, pm).sum()
            iou_mat[i, j] = inter / (union + 1e-8)

    # Hungarian matching
    ri, ci = linear_sum_assignment(-iou_mat)
    valid = iou_mat[ri, ci] > match_iou
    ri = ri[valid]
    ci = ci[valid]

    matched_ious = iou_mat[ri, ci]
    tp = len(matched_ious)
    fp = n_pred - tp
    fn = n_true - tp
    iou_sum = float(matched_ious.sum())

    matched_pairs = [(int(true_ids[t]), int(pred_ids[p]), float(iou_mat[t, p]))
                     for t, p in zip(ri, ci)]

    return tp, fp, fn, iou_sum, matched_pairs

def _build_cls_maps(cls_dict, old2new):
    """
    将原始 instance id -> class id 映射到 remap 后的 id 上
    """
    if cls_dict is None:
        return None
    out = {}
    for old_id, cls_id in cls_dict.items():
        if old_id in old2new:
            out[old2new[old_id]] = int(cls_id)
    return out

def get_fast_pq(true_inst, pred_inst, true_cls=None, pred_cls=None, match_iou=0.5):
    """
    计算单张图的 PQ / DQ / SQ / F1 / Precision / Recall / cls_acc

    返回:
        dict:
            tp, fp, fn, iou_sum,
            PQ, DQ, SQ, F1, Precision, Recall, cls_acc
    """
    true_inst_r, true_old2new = _remap_label(true_inst)
    pred_inst_r, pred_old2new = _remap_label(pred_inst)

    true_cls_r = _build_cls_maps(true_cls, true_old2new)
    pred_cls_r = _build_cls_maps(pred_cls, pred_old2new)

    true_ids = [i for i in np.unique(true_inst_r) if i != 0]
    pred_ids = [i for i in np.unique(pred_inst_r) if i != 0]

    n_true = len(true_ids)
    n_pred = len(pred_ids)

    if n_true == 0 and n_pred == 0:
        return dict(tp=0, fp=0, fn=0, iou_sum=0.0,
                    PQ=np.nan, DQ=np.nan, SQ=np.nan,
                    F1=np.nan, Precision=np.nan, Recall=np.nan,
                    cls_acc=np.nan)

    if n_true == 0:
        return dict(tp=0, fp=n_pred, fn=0, iou_sum=0.0,
                    PQ=0.0, DQ=0.0, SQ=0.0,
                    F1=0.0, Precision=0.0, Recall=0.0,
                    cls_acc=0.0)

    if n_pred == 0:
        return dict(tp=0, fp=0, fn=n_true, iou_sum=0.0,
                    PQ=0.0, DQ=0.0, SQ=0.0,
                    F1=0.0, Precision=0.0, Recall=0.0,
                    cls_acc=0.0)

    iou_mat = np.zeros((n_true, n_pred), dtype=np.float64)
    true_masks = [true_inst_r == t for t in true_ids]
    pred_masks = [pred_inst_r == p for p in pred_ids]

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

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-8)
    sq = iou_sum / (tp + 1e-8) if tp > 0 else 0.0
    pq = iou_sum / (tp + 0.5 * fp + 0.5 * fn + 1e-8)

    cls_acc = np.nan
    if true_cls_r is not None and pred_cls_r is not None and tp > 0:
        correct = 0
        for t_idx, p_idx in zip(ri, ci):
            t_id = int(true_ids[t_idx])
            p_id = int(pred_ids[p_idx])
            if true_cls_r.get(t_id, -1) == pred_cls_r.get(p_id, -2):
                correct += 1
        cls_acc = correct / tp

    return dict(
        tp=tp, fp=fp, fn=fn, iou_sum=iou_sum,
        PQ=pq, DQ=dq, SQ=sq,
        F1=f1, Precision=precision, Recall=recall,
        cls_acc=cls_acc
    )

def _compute_single_sample(args):
    pred, true, p_cls, t_cls, match_iou = args
    return get_fast_pq(true, pred, t_cls, p_cls, match_iou)

def batch_seg_metrics(pred_inst_list, true_inst_list,
                      pred_cls_list=None, true_cls_list=None,
                      match_iou=0.5, num_workers=None):
    """
    数据集级别指标：先累计 TP/FP/FN/IoU_sum，再统一计算 PQ/DQ/SQ/F1

    返回:
        dict(PQ, DQ, SQ, F1, Precision, Recall, cls_acc)
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

    TP = sum(r["tp"] for r in results)
    FP = sum(r["fp"] for r in results)
    FN = sum(r["fn"] for r in results)
    IoU_sum = sum(r["iou_sum"] for r in results)

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    dq = TP / (TP + 0.5 * FP + 0.5 * FN + 1e-8)
    sq = IoU_sum / (TP + 1e-8) if TP > 0 else 0.0
    pq = IoU_sum / (TP + 0.5 * FP + 0.5 * FN + 1e-8)

    cls_vals = [r["cls_acc"] for r in results if not np.isnan(r["cls_acc"])]
    cls_acc = float(np.mean(cls_vals)) if len(cls_vals) > 0 else np.nan

    return dict(
        PQ=float(pq),
        DQ=float(dq),
        SQ=float(sq),
        F1=float(f1),
        Precision=float(precision),
        Recall=float(recall),
        cls_acc=float(cls_acc) if not np.isnan(cls_acc) else np.nan
    )