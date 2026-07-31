"""スクリプト2：欠損付与

完全なRSSマップ（rxPowerXXXX.txt）から、障害物マップ上の
フリースペースのみを観測点候補とし、サンプル固有シードで
シャッフルした先頭 num_obs 個を観測点として残し、それ以外を
欠損値 -250 に置換して保存する。

包含関係：同じ env_id ならシャッフル順が固定なので
  100個 ⊂ 200個 ⊂ ... ⊂ 1200個 が自動的に保証される。

使い方:
  # 1サンプル・1観測点数
  python inject_missing.py --rss ../rxPower1001.txt \
      --label ../data/processed/labels/label_1001.npy \
      --num-obs 100 --env-id 1001 \
      --out ../data/processed/obs100/sparse_1001.npy
"""

import argparse
import os
import numpy as np

from data_gen_lib import (
    N_ROWS, N_COLS, MISSING_VALUE, FREESPACE,
    load_rss_map,
)


def inject_missing(rss_map, obstacle_map, num_obs, env_id):
    """欠損付与済みRSSマップを返す。

    Args:
        rss_map:      (40,50) 完全なRSS行列（dBm）
        obstacle_map: (40,50) 障害物=1 / フリースペース=0
        num_obs:      残す観測点数
        env_id:       サンプル固有シード（包含関係を保証）

    Returns:
        sparse: (40,50) 観測点のみRSS値、他は -250
        n_selected: 実際に選ばれた観測点数
        n_free: フリースペース総数
    """
    # フリースペースの (row, col) インデックス一覧（決定的な順序）
    free_rc = np.argwhere(obstacle_map == FREESPACE)   # shape (n_free, 2)
    n_free = len(free_rc)

    # 安全チェック
    if n_free < num_obs:
        print(f"[WARN] env_id={env_id}: フリースペース数({n_free}) < "
              f"観測点数({num_obs})。全フリースペースを観測点にします。")

    # サンプル固有シードでシャッフル（包含関係のため順序を固定）
    rng = np.random.default_rng(seed=int(env_id))
    perm = rng.permutation(n_free)          # インデックスの並べ替え
    take = min(num_obs, n_free)
    selected = free_rc[perm[:take]]         # 先頭 take 個

    # 出力：まず全て欠損値、選ばれた観測点だけ実測値を入れる
    sparse = np.full((N_ROWS, N_COLS), MISSING_VALUE, dtype=np.float32)
    rows = selected[:, 0]
    cols = selected[:, 1]
    sparse[rows, cols] = rss_map[rows, cols].astype(np.float32)

    return sparse, take, n_free


def process_one(rss_path, label_path, num_obs, env_id, out_path):
    rss_map, filled = load_rss_map(rss_path)
    if filled != N_ROWS * N_COLS:
        print(f"[WARN] {os.path.basename(rss_path)}: "
              f"RSSグリッド充填数 {filled}/{N_ROWS*N_COLS}（欠けがある可能性）")
    obstacle_map = np.load(label_path)

    sparse, n_sel, n_free = inject_missing(rss_map, obstacle_map, num_obs, env_id)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.save(out_path, sparse)
    print(f"[OK] env_id={env_id} num_obs={num_obs}: "
          f"観測点={n_sel} (フリースペース={n_free}) -> {out_path}")
    return sparse


def main():
    ap = argparse.ArgumentParser(description="欠損付与（スクリプト2）")
    ap.add_argument("--rss", required=True, help="rxPowerXXXX.txt のパス")
    ap.add_argument("--label", required=True, help="障害物マップ .npy のパス")
    ap.add_argument("--num-obs", type=int, required=True, help="観測点数")
    ap.add_argument("--env-id", type=int, required=True, help="サンプルID（シード兼用）")
    ap.add_argument("--out", required=True, help="出力 .npy パス")
    args = ap.parse_args()

    process_one(args.rss, args.label, args.num_obs, args.env_id, args.out)


if __name__ == "__main__":
    main()
