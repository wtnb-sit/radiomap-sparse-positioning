"""スクリプト1：障害物マップ生成（正解ラベル）

objectファイルから 40x50 の2値障害物マップを生成する。
  障害物 = 1、フリースペース = 0
グリッド中心が いずれかの障害物の [x_min,x_max]x[y_min,y_max] に
含まれる（境界含む）かで判定し、Y反転して RSSマップと同じ座標系にする。

観測点数に依存しないので1サンプルにつき1回だけ生成する。

使い方:
  # 1サンプル
  python obstacle_map.py --object ../object1001.object --out ../data/processed/labels/label_1001.npy
  # ディレクトリ内の全objectを一括処理
  python obstacle_map.py --all --raw-dir ../data/raw --out-dir ../data/processed/labels
"""

import argparse
import os
import re
import glob
import numpy as np

from data_gen_lib import (
    N_ROWS, N_COLS, OBSTACLE, FREESPACE,
    parse_obstacles, x_center_of_col, y_center_of_row_raw,
)


def build_obstacle_map(object_path):
    """objectファイルから 40x50 の障害物マップ（Y反転済み）を返す。"""
    obstacles = parse_obstacles(object_path)

    # 反転前の向き（row_raw が 1..40、row_raw=1 が y 最小）で埋める
    occ_raw = np.zeros((N_ROWS, N_COLS), dtype=np.uint8)

    # グリッド中心の座標を先に計算
    x_centers = np.array([x_center_of_col(c) for c in range(1, N_COLS + 1)])
    y_centers = np.array([y_center_of_row_raw(r) for r in range(1, N_ROWS + 1)])

    for (x_min, x_max, y_min, y_max) in obstacles:
        col_mask = (x_centers >= x_min) & (x_centers <= x_max)   # (50,)
        row_mask = (y_centers >= y_min) & (y_centers <= y_max)   # (40,)
        # 外積でブロック領域を1に
        occ_raw[np.ix_(row_mask, col_mask)] = OBSTACLE

    # Y反転（row = 41 - row_raw）: row_raw=1(下)→ 最終行40、row_raw=40(上)→ 最終行1
    # occ_raw の index0 = row_raw=1 = y最小 → flipud で index0 を y最大(=上)にする
    obstacle_map = np.flipud(occ_raw).astype(np.uint8)
    return obstacle_map, obstacles


def _sample_id_from_name(path):
    """object1001.object -> '1001' のように数字部分を取り出す。"""
    base = os.path.basename(path)
    m = re.search(r'(\d+)', base)
    return m.group(1) if m else base


def process_one(object_path, out_path):
    obstacle_map, obstacles = build_obstacle_map(object_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.save(out_path, obstacle_map)
    n_obst_cells = int(obstacle_map.sum())
    print(f"[OK] {os.path.basename(object_path)}: "
          f"障害物={len(obstacles)}個, 障害物セル={n_obst_cells}/{N_ROWS*N_COLS}, "
          f"-> {out_path}")
    return obstacle_map


def main():
    ap = argparse.ArgumentParser(description="障害物マップ生成（スクリプト1）")
    ap.add_argument("--object", help="単一objectファイルのパス")
    ap.add_argument("--out", help="単一出力 .npy パス")
    ap.add_argument("--all", action="store_true", help="ディレクトリ内の全objectを処理")
    ap.add_argument("--raw-dir", default="data/raw", help="objectファイル群のディレクトリ")
    ap.add_argument("--out-dir", default="data/processed/labels", help="ラベル出力ディレクトリ")
    args = ap.parse_args()

    if args.all:
        obj_files = sorted(glob.glob(os.path.join(args.raw_dir, "*.object")))
        if not obj_files:
            print(f"[WARN] {args.raw_dir} に .object ファイルがありません")
            return
        for obj in obj_files:
            sid = _sample_id_from_name(obj)
            out = os.path.join(args.out_dir, f"label_{sid}.npy")
            process_one(obj, out)
    else:
        if not args.object or not args.out:
            ap.error("--all を使わない場合は --object と --out が必要です")
        process_one(args.object, args.out)


if __name__ == "__main__":
    main()
