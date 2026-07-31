"""npyファイルの中身を確認する簡易ビューア。

使い方:
  # 数値サマリ + 行列を表示
  python inspect_npy.py ../data/processed/labels/label_1001.npy

  # 画像(PNG)として保存して目で見る
  python inspect_npy.py ../data/processed/obs100/sparse_1001.npy --png

  # CSVに書き出して Excel などで開く
  python inspect_npy.py ../data/processed/labels/label_1001.npy --csv out.csv
"""

import argparse
import os
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="npyビューア")
    ap.add_argument("path", help=".npy ファイルのパス")
    ap.add_argument("--png", action="store_true", help="同名の.pngとして画像保存")
    ap.add_argument("--csv", help="CSVに書き出すパス")
    ap.add_argument("--full", action="store_true", help="行列を省略せず全表示")
    args = ap.parse_args()

    a = np.load(args.path)
    print(f"file : {args.path}")
    print(f"shape: {a.shape}")
    print(f"dtype: {a.dtype}")
    print(f"min={a.min()}  max={a.max()}  mean={a.mean():.4f}")
    uniq = np.unique(a)
    if uniq.size <= 15:
        print(f"unique values: {uniq.tolist()}")
    # 欠損(-250)以外の有効値数（sparseマップ向け）
    valid = int((a > -249).sum())
    print(f"valid (>-249) cells: {valid} / {a.size}")

    if args.full:
        with np.printoptions(threshold=np.inf, linewidth=250):
            print(a)
    else:
        print("--- matrix (省略表示、全表示は --full) ---")
        print(a)

    if args.csv:
        fmt = "%d" if np.issubdtype(a.dtype, np.integer) else "%.4f"
        np.savetxt(args.csv, a, delimiter=",", fmt=fmt)
        print(f"[SAVED] {args.csv}")

    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out_png = os.path.splitext(args.path)[0] + "_view.png"
        plt.figure(figsize=(8, 6))
        # sparseは欠損をマスクして見やすく
        if a.min() <= -249:
            m = np.ma.masked_where(a <= -249, a)
            plt.imshow(m, cmap="viridis", aspect="auto")
        else:
            plt.imshow(a, cmap="gray_r", aspect="auto")
        plt.colorbar()
        plt.title(os.path.basename(args.path))
        plt.tight_layout()
        plt.savefig(out_png, dpi=110)
        print(f"[SAVED] {out_png}")


if __name__ == "__main__":
    main()
