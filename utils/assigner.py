# utils/assigner.py
import math
import torch
import torch.nn.functional as F

#─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def make_anchor_points(feat_shapes, strides, device):
    """
    生成各尺度 anchor point（原图像素坐标，格点中心）

    Returns:
        all_pts: (total_A, 2)  [cx, cy]all_strides: (total_A,)anchor_per_scale: list[int]  每尺度的 anchor 数量
        pts_per_scale   : list of Tensor(H*W, 2) 每尺度单独保存，供 loss 用
    """
    all_pts, all_strides, anchor_per_scale, pts_per_scale = [], [], [], []
    for (H, W), s in zip(feat_shapes, strides):
        ys = torch.arange(H, device=device, dtype=torch.float32)
        xs = torch.arange(W, device=device, dtype=torch.float32)
        cy, cx = torch.meshgrid(ys, xs, indexing='ij')
        pts = torch.stack([(cx + 0.5) * s, (cy + 0.5) * s], dim=-1).reshape(-1, 2)
        all_pts.append(pts)
        all_strides.append(torch.full((pts.shape[0],), s, device=device, dtype=torch.float32))
        anchor_per_scale.append(pts.shape[0])
        pts_per_scale.append(pts)
    return torch.cat(all_pts,0), torch.cat(all_strides, 0), anchor_per_scale, pts_per_scale

def decode_pred_boxes(pred_ltrb, anchor_pts):
    """
    ltrb + anchor_pts →xyxy（原图像素坐标）

    pred_ltrb : (A, 4)
    anchor_pts: (A, 2)  [cx, cy]
    """
    cx, cy = anchor_pts[:, 0], anchor_pts[:, 1]
    return torch.stack([
        cx - pred_ltrb[:, 0],
        cy - pred_ltrb[:, 1],
        cx + pred_ltrb[:, 2],
        cy + pred_ltrb[:, 3],
    ], dim=-1)

def iou_matrix(boxes1, boxes2, eps=1e-7):
    """
    boxes1: (N, 4)xyxy
    boxes2: (M, 4) xyxy
    Returns: (N, M)
    """
    N, M = boxes1.shape[0], boxes2.shape[0]
    b1 = boxes1.unsqueeze(1).expand(N, M, 4)
    b2 = boxes2.unsqueeze(0).expand(N, M, 4)
    ix1 = torch.max(b1[..., 0], b2[..., 0])
    iy1 = torch.max(b1[..., 1], b2[..., 1])
    ix2 = torch.min(b1[..., 2], b2[..., 2])
    iy2 = torch.min(b1[..., 3], b2[..., 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    a1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    a2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = a1.unsqueeze(1) + a2.unsqueeze(0) - inter + eps
    return inter / union

# ─────────────────────────────────────────────
# Task-Aligned Assigner
# ─────────────────────────────────────────────

class TaskAlignedAssigner:
    def __init__(self, topk=10, alpha=0.5, beta=6.0, num_classes=2):
        self.topk= topk
        self.alpha       = alpha
        self.beta        = beta
        self.num_classes = num_classes

    @torch.no_grad()
    def assign(self, pred_cls_logits, pred_boxes, anchor_pts, gt_boxes, gt_labels, img_size):
        """
        pred_cls_logits: (A, nc)
        pred_boxes     : (A, 4)xyxy 已解码
        anchor_pts     : (A, 2)
        gt_boxes       : (G, 4)  xyxy
        gt_labels      : (G,)
        """
        device = pred_cls_logits.device
        A = anchor_pts.shape[0]
        G = gt_boxes.shape[0]

        if G == 0:
            return (
                torch.zeros(A, dtype=torch.bool, device=device),
                torch.zeros(A, 4, device=device),
                torch.full((A,), -1, dtype=torch.long, device=device),
                torch.zeros(A, self.num_classes, device=device),
            )

        # Step1: in-center mask(G, A)
        cx = anchor_pts[:, 0].unsqueeze(0)# (1, A)
        cy = anchor_pts[:, 1].unsqueeze(0)
        in_x = (cx > gt_boxes[:, 0:1]) & (cx < gt_boxes[:, 2:3])  # (G, A)
        in_y = (cy > gt_boxes[:, 1:2]) & (cy < gt_boxes[:, 3:4])
        in_gt = in_x & in_y

        # Step2: 对齐分数 (G, A)
        pred_cls_prob = pred_cls_logits.sigmoid()# (A, nc)
        cls_scores = pred_cls_prob[:, gt_labels.long()].T# (G, A)
        iou_mat= iou_matrix(pred_boxes, gt_boxes).T     # (G, A)
        align_score = (cls_scores ** self.alpha) * (iou_mat ** self.beta) * in_gt.float()

        # Step3: 每gt 选topk
        k = min(self.topk, A)
        _, topk_idx = align_score.topk(k, dim=1)   # (G, k)
        topk_mask = torch.zeros_like(align_score, dtype=torch.bool)
        topk_mask.scatter_(1, topk_idx, True)
        topk_mask &= in_gt

        # Step4: 冲突解决（一个 anchor 保留iou 最大的 gt）
        matched_cnt = topk_mask.sum(0)   # (A,)
        if (matched_cnt > 1).any():
            conflict = matched_cnt > 1
            best_gt= iou_mat[:, conflict].argmax(0)
            topk_mask[:, conflict] = False
            topk_mask[best_gt, conflict.nonzero(as_tuple=True)[0]] = True

        # Step5: 生成 targets
        fg_mask         = topk_mask.any(0)                # (A,)
        assigned_gt_idx = align_score[:, fg_mask].argmax(0)            # (#fg,)

        tgt_boxes= torch.zeros(A, 4, device=device)
        tgt_labels = torch.full((A,), -1, dtype=torch.long, device=device)
        tgt_scores = torch.zeros(A, self.num_classes, device=device)

        tgt_boxes[fg_mask]= gt_boxes[assigned_gt_idx]
        tgt_labels[fg_mask] = gt_labels[assigned_gt_idx]

        matched_iou = iou_mat[assigned_gt_idx, fg_mask.nonzero(as_tuple=True)[0]]
        one_hot     = F.one_hot(gt_labels[assigned_gt_idx], self.num_classes).float()
        tgt_scores[fg_mask] = matched_iou.unsqueeze(1) * one_hot

        return fg_mask, tgt_boxes, tgt_labels, tgt_scores

# ─────────────────────────────────────────────
# build_yolo_targets（train.py 调用入口）
# ─────────────────────────────────────────────

def build_yolo_targets(yolo_preds, gt_boxes_list, gt_labels_list,
                       img_size, device, num_classes=2,
                       strides=(8, 16, 32), topk=10, alpha=0.5, beta=6.0):
    """
    Returns:
        scale_results: list of dict per scale
                {'reg':(B,4,H,W) xyxy, 'cls':(B,nc,H,W), 'mask':(B,1,H,W)}
        pts_per_scale       : list of Tensor(H*W, 2)供 loss 解码用"""
    B = len(gt_boxes_list)
    assigner = TaskAlignedAssigner(topk=topk, alpha=alpha, beta=beta, num_classes=num_classes)

    feat_shapes = [(p['reg'].shape[2], p['reg'].shape[3]) for p in yolo_preds]
    all_pts, all_strides, anchor_per_scale, pts_per_scale = make_anchor_points(
        feat_shapes, list(strides), device
    )

    #拼接预测 (B, total_A, *)
    pred_regs = torch.cat(
        [p['reg'].permute(0, 2, 3, 1).reshape(B, -1, 4) for p in yolo_preds], dim=1
    )
    pred_clss = torch.cat(
        [p['cls'].permute(0, 2, 3, 1).reshape(B, -1, num_classes) for p in yolo_preds], dim=1
    )

    # 结果容器
    scale_results = [
        {
            'reg' : torch.zeros(B, 4, H, W, device=device),# xyxy target
            'cls' : torch.zeros(B, num_classes, H, W, device=device),
            'mask': torch.zeros(B, 1, H, W, device=device),
        }
        for H, W in feat_shapes
    ]

    scale_offsets = [0]
    for n in anchor_per_scale:
        scale_offsets.append(scale_offsets[-1] + n)

    for b in range(B):
        gt_boxes  = gt_boxes_list[b].to(device)
        gt_labels = gt_labels_list[b].to(device)

        pred_boxes_b = decode_pred_boxes(pred_regs[b], all_pts).clamp(0, img_size)

        fg_mask, tgt_boxes, _, tgt_scores = assigner.assign(
            pred_cls_logits=pred_clss[b],
            pred_boxes=pred_boxes_b,
            anchor_pts=all_pts,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            img_size=img_size,
        )

        for s_idx, (H, W) in enumerate(feat_shapes):
            s0, s1 = scale_offsets[s_idx], scale_offsets[s_idx + 1]

            fg_s= fg_mask[s0:s1]                    # (H*W,)
            boxes_s  = tgt_boxes[s0:s1].reshape(H, W, 4).permute(2,0,1) # (4,H,W)xyxy
            scores_s = tgt_scores[s0:s1].reshape(H, W, num_classes).permute(2,0,1)
            mask_s   = fg_s.reshape(H, W).unsqueeze(0).float()           # (1,H,W)

            scale_results[s_idx]['reg'][b]  = boxes_s
            scale_results[s_idx]['cls'][b]  = scores_s
            scale_results[s_idx]['mask'][b] = mask_s

    return scale_results, pts_per_scale