"""スクリプト3：測位誤差付与

欠損付与済みの疎なRSSマップ（obs<N>/sparse_<id>.npy）に対し、
各観測点へ独立な測位誤差を与えて「間違った位置」へ付け替える。
残差計算・正規化は行わない（別スクリプト）。

--- 誤差モデル（preprocessing_final.md §4.4 / research_design_final.md §4.4） ---
  各観測点ごとに独立に  e_x, e_y ~ N(0, σ²)
  σ = 0.61 × sqrt(2/π) ≈ 0.487 [m]   （変位の大きさがレイリー分布・平均0.61m）
  Δc = round(e_x / R),  Δr = round(e_y / R),  R = 0.2 [m/cell]
  境界クリッピング（グリッド外に出たら端に留める）
  丸めは対称な round-half-away-from-zero（±0.5→±1）。零平均対称なので
  Y反転の符号規約は誤差分布に影響しない（row+Δr で適用）。

--- 方針（確定事項） ---
  ・オフライン固定生成（On-the-fly不採用）。1ファイル1回だけ生成して保存。
  ・シードは (env_id, num_obs) ごとに固定 → 再現可能・ファイル間で独立。
  ・衝突（2点が同一セルへ変位）は「後勝ち（上書き）」。
    → 決定的な走査順で書き込み、後の点が先の点を上書きする。
       衝突した分だけ有効観測点数はわずかに減る。
  ・障害物マップは使わない（推論時の正解非依存の原則。変位先が障害物セルでも許容）。

使い方:
  # 1ファイル
  python add_positioning_error.py --sparse ../data/processed/obs100/sparse_1001.npy \
      --num-obs 100 --env-id 1001 \
      --out ../data/processed/poserr/obs100/sparse_1001.npy
  # 全サンプル・全観測点数を一括
  python add_positioning_error.py --all
"""

import argparse
import os
import glob
import re
import numpy as np

from data_gen_lib import N_ROWS, N_COLS, MISSING_VALUE

# 測位誤差の標準偏差 [m]：変位の大きさが平均0.61mのレイリー分布になるよう設定
SIGMA = 0.61 * np.sqrt(2.0 / np.pi)   # ≈ 0.48671
RESOLUTION = 0.2                       # R [m/cell]

OBS_COUNTS = list(range(100, 1201, 100))   # 100..1200


def _round_away(v):
    """round-half-away-from-zero（±0.5→±1）。配列可。"""
    return np.sign(v) * np.floor(np.abs(v) + 0.5)


def add_positioning_error(sparse_map, num_obs, env_id):
    """疎なRSSマップに測位誤差を付与して返す。

    Args:
        sparse_map: (40,50) 欠損付与済みRSS（観測点のみ値、他は -250）
        num_obs:    観測点数（シード生成に使用）
        env_id:     サンプルID（シード生成に使用）

    Returns:
        out:        (40,50) 測位誤差付きの疎なRSS（他は -250）
        stats:      dict（元観測点数・出力有効点数・境界クリップ数・衝突数）
    """
    # 観測点（有効セル）の (row, col) を決定的な順序（row-major）で取得
    obs_rc = np.argwhere(sparse_map > MISSING_VALUE + 1)   # shape (n, 2)
    n = len(obs_rc)

    # (env_id, num_obs) ごとに固定したシードで誤差を生成（再現可能・ファイル間独立）
    rng = np.random.default_rng([int(env_id), int(num_obs)])
    e_x = rng.normal(0.0, SIGMA, size=n)
    e_y = rng.normal(0.0, SIGMA, size=n)

    # グリッド変位量（対称丸め）
    d_col = _round_away(e_x / RESOLUTION).astype(int)
    d_row = _round_away(e_y / RESOLUTION).astype(int)

    rows = obs_rc[:, 0]
    cols = obs_rc[:, 1]
    # 零平均対称ノイズのため row+Δr の符号規約でよい（分布に影響しない）
    new_rows = np.clip(rows + d_row, 0, N_ROWS - 1)
    new_cols = np.clip(cols + d_col, 0, N_COLS - 1)

    clipped = int(np.sum((rows + d_row != new_rows) | (cols + d_col != new_cols)))

    # 出力：まず全欠損、決定的な順序で書き込み（後勝ち＝後のインデックスが上書き）
    out = np.full((N_ROWS, N_COLS), MISSING_VALUE, dtype=np.float32)
    for i in range(n):
        out[new_rows[i], new_cols[i]] = sparse_map[rows[i], cols[i]]

    n_valid_out = int(np.sum(out > MISSING_VALUE + 1))
    collisions = n - n_valid_out   # 上書きで失われた点数

    stats = {"n_in": n, "n_out": n_valid_out,
             "clipped": clipped, "collisions": collisions}
    return out, stats


def _sample_id(path):
    m = re.search(r'(\d+)', os.path.basename(path))
    return m.group(1) if m else None


def process_one(sparse_path, num_obs, env_id, out_path):
    sparse_map = np.load(sparse_path)
    out, stats = add_positioning_error(sparse_map, num_obs, env_id)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.save(out_path, out)
    print(f"[OK] id={env_id} num_obs={num_obs}: "
          f"in={stats['n_in']} out={stats['n_out']} "
          f"(clip={stats['clipped']}, collide={stats['collisions']}) -> {out_path}")
    return out


def run_all(processed_root):
    """data/processed/obs<N>/sparse_<id>.npy 全てに適用し
    data/processed/poserr/obs<N>/sparse_<id>.npy へ保存する。

    併せて、各ファイルの実効観測点数を poserr/poserr_stats.csv に記録する。
    （評価時に「条件ごとの測位誤差後 実効点数」を集計できるようにするため。
      グラフ横軸は num_obs でよいが、補足統計として有用。）
    """
    total = 0
    tot_in = tot_out = 0
    csv_rows = ["num_obs,env_id,n_in,n_out,clipped,collisions"]
    per_cond = {}   # num_obs -> list of n_out（条件ごとの実効点数集計用）

    for num_obs in OBS_COUNTS:
        in_dir = os.path.join(processed_root, f"obs{num_obs}")
        out_dir = os.path.join(processed_root, "poserr", f"obs{num_obs}")
        files = sorted(glob.glob(os.path.join(in_dir, "sparse_*.npy")))
        if not files:
            print(f"[WARN] {in_dir} に sparse_*.npy がありません")
            continue
        os.makedirs(out_dir, exist_ok=True)
        per_cond[num_obs] = []
        for f in files:
            sid = _sample_id(f)
            sparse_map = np.load(f)
            out, stats = add_positioning_error(sparse_map, num_obs, int(sid))
            np.save(os.path.join(out_dir, f"sparse_{sid}.npy"), out)
            total += 1
            tot_in += stats["n_in"]
            tot_out += stats["n_out"]
            per_cond[num_obs].append(stats["n_out"])
            csv_rows.append(f"{num_obs},{sid},{stats['n_in']},"
                            f"{stats['n_out']},{stats['clipped']},{stats['collisions']}")
        arr = np.array(per_cond[num_obs])
        print(f"[OK] obs{num_obs}: {len(files)}ファイル  "
              f"実効点数 平均={arr.mean():.1f} 標準偏差={arr.std():.1f} "
              f"最小={arr.min()} 最大={arr.max()}")

    # 統計CSVを保存
    poserr_root = os.path.join(processed_root, "poserr")
    os.makedirs(poserr_root, exist_ok=True)
    csv_path = os.path.join(poserr_root, "poserr_stats.csv")
    with open(csv_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(csv_rows) + "\n")

    lost = tot_in - tot_out
    rate = (lost / tot_in * 100) if tot_in else 0.0
    print(f"\n完了。総ファイル数={total}, "
          f"総観測点 in={tot_in} out={tot_out} "
          f"(衝突消失={lost}, {rate:.3f}%)")
    print(f"実効点数の統計CSV -> {csv_path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description="測位誤差付与（スクリプト3）")
    ap.add_argument("--sparse", help="入力の疎なRSS .npy（obs<N>/sparse_<id>.npy）")
    ap.add_argument("--num-obs", type=int, help="観測点数（シード用）")
    ap.add_argument("--env-id", type=int, help="サンプルID（シード用）")
    ap.add_argument("--out", help="出力 .npy パス")
    ap.add_argument("--all", action="store_true", help="全サンプル・全観測点数を一括処理")
    ap.add_argument("--processed-root",
                    default=os.path.join(root, "data", "processed"))
    args = ap.parse_args()

    if args.all:
        run_all(args.processed_root)
    else:
        if not (args.sparse and args.num_obs and args.env_id and args.out):
            ap.error("--all を使わない場合は --sparse --num-obs --env-id --out が必要です")
        process_one(args.sparse, args.num_obs, args.env_id, args.out)


if __name__ == "__main__":
    main()
