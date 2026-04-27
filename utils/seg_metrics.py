# utils/seg_metrics.py
import numpy as np
from scipy.optimize import linear_sum_assignment

def _remap_label(inst):
    """重新编号实例id为连续整数，同时返回 旧id→新id 的映射"""
    ids = [i for i in np.unique(inst) if i != 0]
    out = np.zeros_like(inst)
    old2new = {}
    for new_id, old_id in enumerate(ids, 1):
        out[inst == old_id] = new_id
        old2new[old_id] = new_id
    return out, old2new

def get_fast_pq(true_inst, pred_inst, true_cls=None, pred_cls=None, match_iou=0.5):
    """
    true_inst, pred_inst : (H,W) int32
    true_cls, pred_cls   : dict {原始inst_id: class_id}，可为None

    返回: dictPQ, DQ, SQ, F1, Precision, Recall, cls_acc
    """
    true_inst, true_old2new = _remap_label(true_inst)
    pred_inst, pred_old2new = _remap_label(pred_inst)

    #── 把 cls_dict 的 key 同步更新为 remap 后的新 id ──────────
    def _remap_cls(cls_dict, old2new):
        if cls_dict is None:
            return None
        return {old2new[old]: c for old, c in cls_dict.items() if old in old2new}

    true_cls_r = _remap_cls(true_cls, true_old2new)
    pred_cls_r = _remap_cls(pred_cls, pred_old2new)

    true_ids = [i for i in np.unique(true_inst) if i != 0]
    pred_ids = [i for i in np.unique(pred_inst) if i != 0]

    n_true, n_pred = len(true_ids), len(pred_ids)

    # ── Bug2修复：空图不贡献分数，用NaN 标记，聚合时跳过 ──────
    if n_true == 0 and n_pred == 0:
        return dict(PQ=np.nan, DQ=np.nan, SQ=np.nan,F1=np.nan, Precision=np.nan, Recall=np.nan, cls_acc=np.nan)
    if n_true == 0 or n_pred == 0:
        return dict(PQ=0., DQ=0., SQ=0., F1=0., Precision=0., Recall=0., cls_acc=0.)

    # ── IoU 矩阵 ────────────────────────────────────────────────
    iou_mat = np.zeros((n_true, n_pred), dtype=np.float64)
    true_masks = [true_inst == t for t in true_ids]
    pred_masks = [pred_inst == p for p in pred_ids]
    for i, tm in enumerate(true_masks):
        for j, pm in enumerate(pred_masks):
            inter = (tm & pm).sum()
            union = (tm | pm).sum()
            iou_mat[i, j] = inter / (union + 1e-8)

    # ── 匈牙利匹配 ───────────────────────────────────────────────
    ri, ci = linear_sum_assignment(-iou_mat)
    valid = iou_mat[ri, ci] > match_iou
    ri, ci = ri[valid], ci[valid]
    paired_iou = iou_mat[ri, ci]

    tp = len(ri)
    fp = n_pred - tp
    fn = n_true - tp

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    sq        = paired_iou.mean() if tp > 0 else 0.0
    dq        = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-8)
    pq        = sq * dq

    # ── 分类准确率（Bug1 修复：用 remap 后的新 id 查字典）───────
    cls_acc = np.nan
    if true_cls_r is not None and pred_cls_r is not None and tp > 0:
        correct = 0
        for ti, pi in zip(ri, ci):
            t_id = true_ids[ti]   # 已是 remap 后的新 id
            p_id = pred_ids[pi]
            if true_cls_r.get(t_id, -1) == pred_cls_r.get(p_id, -2):
                correct += 1
        cls_acc = correct / tp

    return dict(PQ=pq, DQ=dq, SQ=sq, F1=f1,
                Precision=precision, Recall=recall, cls_acc=cls_acc)

def batch_seg_metrics(pred_inst_list, true_inst_list,
                      pred_cls_list=None, true_cls_list=None,
                      match_iou=0.5):
    keys = ['PQ', 'DQ', 'SQ', 'F1', 'Precision', 'Recall', 'cls_acc']
    accum = {k: [] for k in keys}

    for i, (pred, true) in enumerate(zip(pred_inst_list, true_inst_list)):
        p_cls = pred_cls_list[i] if pred_cls_list else None
        t_cls = true_cls_list[i] if true_cls_list else None
        res = get_fast_pq(true, pred, t_cls, p_cls, match_iou)
        for k in keys:
            accum[k].append(res[k])

    # nanmean：跳过空图的NaN，只统计有实例的图
    return {k: float(np.nanmean(v)) if any(~np.isnan(v) for v in [accum[k]])
            else 0.0 for k, v in accum.items()}