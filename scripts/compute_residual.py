"""スクリプト5：残差計算（④）

測位誤差付きの疎なRSSマップ（poserr/obs<N>/sparse_<id>.npy）から、
理論値マップ（fspl_map.npy）を引いて残差マップを作る。

    Residual(p) = RSS_obs(p) - F_FSPL(p)   （観測点 M(p)=1）
                = -250                       （欠損 M(p)=0）

--- 仕様（research_design_final.md §4.6, preprocessing_final.md §④） ---
  ・入力RSSは測位誤差付き（poserr/）。理論値も「ズレた位置」の値を引く。
  ・残差は正規化前の物理量（dB）まで。正規化・マスク生成は別スクリプト。
  ・疎パターンは不変（有効セルの位置はそのまま、値だけ RSS→残差 に置換）。
  ・残差の物理的意味＝障害物による超過減衰（シャドウイング）。

出力：data/processed/residual/obs<N>/residual_<id>.npy  （(40,50) float32, 欠損=-250）

使い方:
  # 1ファイル
  python compute_residual.py --sparse ../data/processed/poserr/obs100/sparse_1001.npy \
      --fspl ../data/processed/fspl_map.npy \
      --out ../data/processed/residual/obs100/residual_1001.npy
  # 全サンプル・全観測点数を一括
  python compute_residual.py --all
"""

import argparse
import os
import glob
import re
import numpy as np

from data_gen_lib import N_ROWS, N_COLS, MISSING_VALUE

OBS_COUNTS = list(range(100, 1201, 100))


def compute_residual(sparse_map, fspl_map):
    """測位誤差付きRSSと理論値マップから残差マップを返す。

    Args:
        sparse_map: (40,50) 観測RSS（観測点のみ値、他は -250）
        fspl_map:   (40,50) 理論値マップ F_FSPL（全セルで密）

    Returns:
        residual: (40,50) float32（観測点=RSS-FSPL、欠損=-250）
        n_valid:  有効観測点数
    """
    valid = sparse_map > MISSING_VALUE + 1                      # 観測点マスク
    residual = np.full((N_ROWS, N_COLS), MISSING_VALUE, dtype=np.float32)
    residual[valid] = (sparse_map[valid] - fspl_map[valid]).astype(np.float32)
    return residual, int(valid.sum())


def _sample_id(path):
    m = re.search(r'(\d+)', os.path.basename(path))
    return m.group(1) if m else None


def process_one(sparse_path, fspl_path, out_path):
    sparse_map = np.load(sparse_path)
    fspl_map = np.load(fspl_path)
    residual, n_valid = compute_residual(sparse_map, fspl_map)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.save(out_path, residual)
    vals = residual[residual > MISSING_VALUE + 1]
    print(f"[OK] {os.path.basename(sparse_path)}: 観測点={n_valid} "
          f"残差[{vals.min():.2f},{vals.max():.2f}] 平均{vals.mean():.2f} -> {out_path}")
    return residual


def run_all(processed_root):
    """poserr/obs<N>/sparse_<id>.npy 全てに適用し
    residual/obs<N>/residual_<id>.npy へ保存する。"""
    fspl_path = os.path.join(processed_root, "fspl_map.npy")
    if not os.path.exists(fspl_path):
        raise FileNotFoundError(f"理論値マップがありません: {fspl_path}"
                                "（先に theoretical_map.py を実行）")
    fspl_map = np.load(fspl_path)

    total = 0
    tot_valid = 0
    gmin, gmax = np.inf, -np.inf
    for num_obs in OBS_COUNTS:
        in_dir = os.path.join(processed_root, "poserr", f"obs{num_obs}")
        out_dir = os.path.join(processed_root, "residual", f"obs{num_obs}")
        files = sorted(glob.glob(os.path.join(in_dir, "sparse_*.npy")))
        if not files:
            print(f"[WARN] {in_dir} に sparse_*.npy がありません")
            continue
        os.makedirs(out_dir, exist_ok=True)
        for f in files:
            sid = _sample_id(f)
            sparse_map = np.load(f)
            residual, n_valid = compute_residual(sparse_map, fspl_map)
            np.save(os.path.join(out_dir, f"residual_{sid}.npy"), residual)
            total += 1
            tot_valid += n_valid
            vals = residual[residual > MISSING_VALUE + 1]
            gmin = min(gmin, float(vals.min()))
            gmax = max(gmax, float(vals.max()))
        print(f"[OK] obs{num_obs}: {len(files)}ファイル処理")
    print(f"\n完了。総ファイル数={total}, 総観測点={tot_valid:,}, "
          f"残差の全体値域=[{gmin:.2f}, {gmax:.2f}] dB")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description="残差計算（スクリプト5）")
    ap.add_argument("--sparse", help="入力の測位誤差付きRSS .npy（poserr/obs<N>/...）")
    ap.add_argument("--fspl", default=None, help="理論値マップ .npy（既定: processed/fspl_map.npy）")
    ap.add_argument("--out", help="出力 .npy パス")
    ap.add_argument("--all", action="store_true", help="全サンプル・全観測点数を一括処理")
    ap.add_argument("--processed-root",
                    default=os.path.join(root, "data", "processed"))
    args = ap.parse_args()

    if args.all:
        run_all(args.processed_root)
    else:
        if not (args.sparse and args.out):
            ap.error("--all を使わない場合は --sparse と --out が必要です")
        fspl = args.fspl or os.path.join(args.processed_root, "fspl_map.npy")
        process_one(args.sparse, fspl, args.out)


if __name__ == "__main__":
    main()
