"""make_clean_residual.py：測位誤差なし(clean)の正規化残差を生成

条件①はモデル入力が「正規化残差」だが、既存の residual_norm/ は poserr/（測位誤差付き）
から作られている。shape誤差が測位誤差由来かを切り分ける実験のため、
**測位誤差を付与していない obs<N>/sparse_<id>.npy から同じ手順で残差を作る**。

再利用（新規実装しない・再回帰しない）:
  compute_residual.compute_residual(sparse, fspl)  … RSS - F_FSPL（既存 fspl_map.npy）
  normalize_residual.normalize_residual(residual)  … サンプルごとz-score(ddof=0)

入力 : data/processed/obs<N>/sparse_<id>.npy      （欠損付与のみ・測位誤差なし）
出力 : data/processed/clean_residual_norm/obs<N>/resnorm_<id>.npy
       （既存 residual_norm/ とは別パス。既存ディレクトリは一切変更しない）

使い方:
  python make_clean_residual.py --obs 100 600
"""

import argparse
import glob
import os
import re
import numpy as np

from data_gen_lib import MISSING_VALUE
from compute_residual import compute_residual
from normalize_residual import normalize_residual

CLEAN_DIR = "clean_residual_norm"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description="clean(測位誤差なし)正規化残差の生成")
    ap.add_argument("--processed-root", default=os.path.join(root, "data", "processed"))
    ap.add_argument("--obs", type=int, nargs="*", default=[100, 600])
    args = ap.parse_args()

    fspl = np.load(os.path.join(args.processed_root, "fspl_map.npy"))
    print(f"FSPL: {fspl.shape} min={fspl.min():.3f} max={fspl.max():.3f}（既存を再利用・再回帰なし）")

    for n in args.obs:
        in_dir = os.path.join(args.processed_root, f"obs{n}")
        out_dir = os.path.join(args.processed_root, CLEAN_DIR, f"obs{n}")
        os.makedirs(out_dir, exist_ok=True)
        files = sorted(glob.glob(os.path.join(in_dir, "sparse_*.npy")))
        if not files:
            print(f"[WARN] {in_dir} が空"); continue
        cnt = 0
        for f in files:
            sid = int(re.search(r'(\d+)', os.path.basename(f)).group(1))
            sparse = np.load(f)
            res, n_valid = compute_residual(sparse, fspl)
            out, mu, sigma, nv = normalize_residual(res)
            np.save(os.path.join(out_dir, f"resnorm_{sid}.npy"), out)
            cnt += 1
        print(f"[OK] obs{n}: {cnt}件 -> {out_dir}")

    # ---- サニティ：clean と poserr の観測点座標が違うこと・点数が同等であること ----
    print("\n=== サニティ（1サンプルで clean vs poserr を比較）===")
    for n in args.obs:
        sid = 5501  # test 先頭
        cl = np.load(os.path.join(args.processed_root, CLEAN_DIR, f"obs{n}", f"resnorm_{sid}.npy"))
        po = np.load(os.path.join(args.processed_root, "residual_norm", f"obs{n}", f"resnorm_{sid}.npy"))
        mc = (cl > MISSING_VALUE + 1); mp = (po > MISSING_VALUE + 1)
        same = np.array_equal(mc, mp)
        rc_c = set(map(tuple, np.argwhere(mc))); rc_p = set(map(tuple, np.argwhere(mp)))
        print(f"  obs{n} id{sid}: clean有効点={mc.sum()} poserr有効点={mp.sum()} "
              f"（指定{n}点。poserrは衝突で減る）")
        print(f"    観測点座標が同一か: {same}（False＝poserr側がズレている＝期待どおり）")
        print(f"    clean のみに在る点={len(rc_c - rc_p)}  poserr のみに在る点={len(rc_p - rc_c)}")


if __name__ == "__main__":
    main()
