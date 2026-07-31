"""Step1：P_0・n の回帰（方式B：全サンプル共通の固定値）

学習データ（train）の疎な観測点から、対数距離パスロスモデル
    RSS_obs ≈ P_0 - 10 n log10( max(d, d_0) / d_0 )
を最小二乗回帰し、全データ共通の固定値 P_0, n を決定する。

--- 仕様（research_design_final.md §4.5, preprocessing_final.md §2） ---
  ・使用データ：学習データの疎な観測点のみ（完全マップに依存しない）
  ・座標：測位誤差を含んだまま（poserr/ の観測点座標）
  ・障害物減衰：含めたまま回帰（LOS判定は推論時に不可能なため）
  ・距離 d：各観測点のグリッド中心から送信機 (TX_X,TX_Y)=(5,4)m までのユークリッド距離
  ・回帰は train セット（1001–5000）のみ

出力：data/processed/pathloss_params.json
  { "P0": ..., "n": ..., "d0": 0.2, "tx": [5.0,4.0],
    "n_points": ..., "r2": ..., "residual_std": ..., "conditions": [...] }

使い方:
  python fit_pathloss.py                     # 既定: 学習セット・obs1200(poserr)で回帰
  python fit_pathloss.py --conditions all     # 全12条件をプール
  python fit_pathloss.py --conditions 600,1200
"""

import argparse
import os
import glob
import re
import json
import numpy as np

from data_gen_lib import (
    MISSING_VALUE, TX_X, TX_Y, D0, grid_center_coords,
)
from splits import split_of

OBS_COUNTS = list(range(100, 1201, 100))


def _sample_id(path):
    m = re.search(r'(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else None


def collect_points(processed_root, conditions, split="train"):
    """指定条件・分割の観測点から (d[m], rss[dBm]) の配列を集める。

    poserr/obs<N>/sparse_<id>.npy を読み、有効セルの中心座標から距離を計算。
    """
    X, Y = grid_center_coords()                    # (40,50) 物理座標[m]
    dist_grid = np.sqrt((X - TX_X) ** 2 + (Y - TX_Y) ** 2)   # 各セルの送信機距離

    d_list, rss_list = [], []
    n_files = 0
    for num_obs in conditions:
        in_dir = os.path.join(processed_root, "poserr", f"obs{num_obs}")
        files = sorted(glob.glob(os.path.join(in_dir, "sparse_*.npy")))
        for f in files:
            sid = _sample_id(f)
            if split_of(sid) != split:
                continue
            a = np.load(f)
            valid = a > MISSING_VALUE + 1
            d_list.append(dist_grid[valid])
            rss_list.append(a[valid])
            n_files += 1
    if not d_list:
        raise RuntimeError("観測点が集まりませんでした（パス・条件・分割を確認）")
    d = np.concatenate(d_list)
    rss = np.concatenate(rss_list).astype(float)
    return d, rss, n_files


def fit_pathloss(d, rss, d0=D0):
    """RSS = P_0 - 10 n log10(max(d,d0)/d0) を最小二乗回帰。

    Returns: P0, n, r2, residual_std
    """
    x = np.log10(np.maximum(d, d0) / d0)           # 説明変数
    # y = a + b x   （a = P0, b = -10 n）
    b, a = np.polyfit(x, rss, 1)                    # 傾きb・切片a
    P0 = a
    n = -b / 10.0
    pred = a + b * x
    resid = rss - pred
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((rss - rss.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return P0, n, r2, float(resid.std())


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description="P_0・n の回帰（方式B）")
    ap.add_argument("--processed-root",
                    default=os.path.join(root, "data", "processed"))
    ap.add_argument("--conditions", default="1200",
                    help="回帰に使う観測点数条件。'all' か カンマ区切り（例 600,1200）")
    ap.add_argument("--split", default="train", help="使用する分割（既定 train）")
    ap.add_argument("--out", default=None, help="出力JSONパス")
    args = ap.parse_args()

    if args.conditions.strip().lower() == "all":
        conditions = OBS_COUNTS
    else:
        conditions = [int(c) for c in args.conditions.split(",")]

    print(f"条件={conditions}  分割={args.split}")
    d, rss, n_files = collect_points(args.processed_root, conditions, args.split)
    print(f"読み込みファイル数={n_files}  観測点総数={len(d):,}")

    P0, n, r2, resid_std = fit_pathloss(d, rss)
    print("\n=== 回帰結果（方式B：全データ共通の固定値） ===")
    print(f"  P_0 = {P0:.4f} dBm")
    print(f"  n   = {n:.4f}")
    print(f"  R^2 = {r2:.4f}")
    print(f"  残差std = {resid_std:.4f} dB")
    print(f"  距離範囲: {d.min():.3f} 〜 {d.max():.3f} m")

    params = {
        "P0": float(P0), "n": float(n), "d0": float(D0),
        "tx": [float(TX_X), float(TX_Y)],
        "n_points": int(len(d)), "n_files": int(n_files),
        "r2": float(r2), "residual_std": float(resid_std),
        "conditions": conditions, "split": args.split,
    }
    out = args.out or os.path.join(args.processed_root, "pathloss_params.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(params, fp, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] {out}")


if __name__ == "__main__":
    main()
