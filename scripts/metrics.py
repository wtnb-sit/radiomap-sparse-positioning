"""評価指標（train.py / eval.py 共用）

  ・per_image_iou : サンプルごとのクラス別IoU
  ・miou_obstacle : サンプルごとの mIoU と 障害物IoU（argmax 2値化）
  ・HD95, Boundary-F1 は eval.py 側で使う（境界抽出は本モジュール）

mIoU集計方針（training_eval_spec.md §9.3）：per-image で計算し、
観測点数ごとにサンプル平均（macro）する。
"""

import numpy as np
import torch


@torch.no_grad()
def miou_obstacle(logits, target, eps=1e-6):
    """logits(B,2,H,W), target(B,H,W) long → (mIoU[B], 障害物IoU[B])。

    argmax(閾値0.5)で2値化。mIoU=（障害物IoU+フリースペースIoU)/2。
    """
    pred = logits.argmax(dim=1)                      # (B,H,W)
    miou = torch.empty(pred.shape[0])
    obst_iou = torch.empty(pred.shape[0])
    for b in range(pred.shape[0]):
        p, t = pred[b], target[b]
        ious = []
        for c in (0, 1):                             # 0=free, 1=obst
            pc, tc = (p == c), (t == c)
            inter = (pc & tc).sum().float()
            union = (pc | tc).sum().float()
            ious.append(((inter + eps) / (union + eps)).item())
        miou[b] = 0.5 * (ious[0] + ious[1])
        obst_iou[b] = ious[1]
    return miou, obst_iou


def binary_boundary(mask_2d):
    """2値マップ（H,W bool/0-1）の境界（外周1画素）を返す。

    boundary = mask & ~erosion(mask)。scipy.ndimage.binary_erosion 使用。
    """
    from scipy.ndimage import binary_erosion
    m = np.asarray(mask_2d).astype(bool)
    if not m.any():
        return np.zeros_like(m, dtype=bool)
    return m & ~binary_erosion(m)


def _boundary_coords(binary_2d):
    return np.argwhere(binary_boundary(binary_2d))       # (N,2)


# 画像対角長（予測が空のときのHD95ペナルティ）：√(40²+50²)≈64.03
DIAG_PENALTY = float(np.hypot(40, 50))


def obstacle_f1(pred_bin, gt_bin, eps=1e-6):
    """障害物クラスの画素F1（Dice）= 2TP/(2TP+FP+FN)。"""
    p = np.asarray(pred_bin).astype(bool)
    g = np.asarray(gt_bin).astype(bool)
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()
    return float((2 * tp + eps) / (2 * tp + fp + fn + eps))


def _directed_dists(a_coords, b_coords):
    """a の各点から b の最近傍までの距離配列。b が空なら None。"""
    from scipy.spatial.distance import cdist
    if len(a_coords) == 0 or len(b_coords) == 0:
        return None
    return cdist(a_coords, b_coords).min(axis=1)


def hd95(pred_bin, gt_bin, penalty=DIAG_PENALTY):
    """双方向の境界距離をプールした95パーセンタイル（セル）。

    予測が空（境界なし）の場合はペナルティ値を返す。
    """
    pc = _boundary_coords(pred_bin)
    gc = _boundary_coords(gt_bin)
    if len(pc) == 0 or len(gc) == 0:
        return penalty
    d_pg = _directed_dists(pc, gc)
    d_gp = _directed_dists(gc, pc)
    pooled = np.concatenate([d_pg, d_gp])
    return float(np.percentile(pooled, 95))


def boundary_f1(pred_bin, gt_bin, tau):
    """許容半径 τ セルでの Boundary F1。予測境界が空なら0。"""
    pc = _boundary_coords(pred_bin)
    gc = _boundary_coords(gt_bin)
    if len(pc) == 0 or len(gc) == 0:
        return 0.0
    d_pg = _directed_dists(pc, gc)          # 予測→正解
    d_gp = _directed_dists(gc, pc)          # 正解→予測
    precision = np.mean(d_pg <= tau)        # τ以内にマッチした予測境界の割合
    recall = np.mean(d_gp <= tau)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def hd95_pooled_distances(pred_bin, gt_bin):
    """HD95分布可視化用：双方向距離を全て返す（プール済み配列 or None）。"""
    pc = _boundary_coords(pred_bin)
    gc = _boundary_coords(gt_bin)
    d_pg = _directed_dists(pc, gc)
    d_gp = _directed_dists(gc, pc)
    if d_pg is None or d_gp is None:
        return None
    return np.concatenate([d_pg, d_gp])
