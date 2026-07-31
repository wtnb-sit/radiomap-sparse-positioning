"""スクリプト6：z-score正規化（⑤・サンプルごと/インスタンス正規化）

各残差マップを、そのマップ自身の有効観測点から計算した μ,σ で正規化する。

    Residual_norm(p) = (Residual(p) - μ) / σ    （有効セル M(p)=1）
                     = -250                      （欠損 M(p)=0）

    μ, σ ：そのサンプルの有効観測点（残差）の平均・標準偏差（母標準偏差 ddof=0）

--- 方針（本プロジェクトの決定） ---
  ・正規化は「サンプルごと（各学習データ単位）」に行う。
    → 各入力を平均0・標準偏差1に揃える標準的な入力正規化。
    → train/val/test に依存せず自己完結（推論時も同一処理で成立）。
  ・有効グリッドのみで μ,σ を計算（欠損は対象外）。疎パターンは不変。
  ・欠損セルは -250 のまま（センチネル維持）。
  ・σ=0（全有効値が同一）の異常時はゼロ除算回避のため σ=1 とする。
  ※ 資料(preprocessing_final.md §⑤)は「学習データ全体の μ,σ」と記載していたが、
     厳密な方式は未定義だったため、本実装ではサンプルごと正規化を採用する。

出力：data/processed/residual_norm/obs<N>/resnorm_<id>.npy  （(40,50) float32, 欠損=-250）
      data/processed/residual_norm/resnorm_stats.csv        （各ファイルの μ,σ,n_valid）

使い方:
  # 1ファイル
  python normalize_residual.py --residual ../data/processed/residual/obs100/residual_1001.npy \
      --out ../data/processed/residual_norm/obs100/resnorm_1001.npy
  # 全サンプル・全観測点数を一括
  python normalize_residual.py --all
"""

import argparse
import os
import glob
import re
import numpy as np

from data_gen_lib import N_ROWS, N_COLS, MISSING_VALUE

OBS_COUNTS = list(range(100, 1201, 100))
EPS = 1e-8


def normalize_residual(residual_map):
    """残差マップをサンプルごとz-score正規化して返す。

    Returns:
        out:     (40,50) float32（有効セル=正規化値、欠損=-250）
        mu, sigma, n_valid
    """
    valid = residual_map > MISSING_VALUE + 1
    vals = residual_map[valid].astype(np.float64)
    n_valid = int(valid.sum())

    mu = float(vals.mean()) if n_valid > 0 else 0.0
    sigma = float(vals.std(ddof=0)) if n_valid > 0 else 1.0
    denom = sigma if sigma > EPS else 1.0        # ゼロ除算回避

    out = np.full((N_ROWS, N_COLS), MISSING_VALUE, dtype=np.float32)
    out[valid] = ((vals - mu) / denom).astype(np.float32)
    return out, mu, sigma, n_valid


def _sample_id(path):
    m = re.search(r'(\d+)', os.path.basename(path))
    return m.group(1) if m else None


def process_one(residual_path, out_path):
    residual_map = np.load(residual_path)
    out, mu, sigma, n_valid = normalize_residual(residual_map)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.save(out_path, out)
    v = out[out > MISSING_VALUE + 1]
    print(f"[OK] {os.path.basename(residual_path)}: n={n_valid} "
          f"mu={mu:.3f} sigma={sigma:.3f} -> 正規化後[mean={v.mean():.3f}, std={v.std():.3f}]"
          f" -> {out_path}")
    return out


def run_all(processed_root):
    """residual/obs<N>/residual_<id>.npy 全てに適用し
    residual_norm/obs<N>/resnorm_<id>.npy へ保存する。μ,σをCSV記録。"""
    total = 0
    csv_rows = ["num_obs,env_id,mu,sigma,n_valid"]
    for num_obs in OBS_COUNTS:
        in_dir = os.path.join(processed_root, "residual", f"obs{num_obs}")
        out_dir = os.path.join(processed_root, "residual_norm", f"obs{num_obs}")
        files = sorted(glob.glob(os.path.join(in_dir, "residual_*.npy")))
        if not files:
            print(f"[WARN] {in_dir} に residual_*.npy がありません")
            continue
        os.makedirs(out_dir, exist_ok=True)
        for f in files:
            sid = _sample_id(f)
            residual_map = np.load(f)
            out, mu, sigma, n_valid = normalize_residual(residual_map)
            np.save(os.path.join(out_dir, f"resnorm_{sid}.npy"), out)
            csv_rows.append(f"{num_obs},{sid},{mu:.6f},{sigma:.6f},{n_valid}")
            total += 1
        print(f"[OK] obs{num_obs}: {len(files)}ファイル処理")

    norm_root = os.path.join(processed_root, "residual_norm")
    os.makedirs(norm_root, exist_ok=True)
    csv_path = os.path.join(norm_root, "resnorm_stats.csv")
    with open(csv_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(csv_rows) + "\n")
    print(f"\n完了。総ファイル数={total}")
    print(f"μ,σ 統計CSV -> {csv_path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description="z-score正規化（スクリプト6・サンプルごと）")
    ap.add_argument("--residual", help="入力の残差 .npy（residual/obs<N>/...）")
    ap.add_argument("--out", help="出力 .npy パス")
    ap.add_argument("--all", action="store_true", help="全サンプル・全観測点数を一括処理")
    ap.add_argument("--processed-root",
                    default=os.path.join(root, "data", "processed"))
    args = ap.parse_args()

    if args.all:
        run_all(args.processed_root)
    else:
        if not (args.residual and args.out):
            ap.error("--all を使わない場合は --residual と --out が必要です")
        process_one(args.residual, args.out)


if __name__ == "__main__":
    main()
