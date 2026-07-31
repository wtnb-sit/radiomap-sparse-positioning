"""スクリプト4：理論値マップ F_FSPL の生成（③）

回帰で得た固定 P_0, n を用いて、送信機 (TX_X,TX_Y)=(5,4)m からの
対数距離パスロスに基づく 40x50 の理論値マップを計算する。

    F_FSPL(p) = P_0 - 10 n log10( max(||p - t||, d_0) / d_0 )

送信機が固定のため全データ共通の1枚。障害物には依存しない。
残差計算・正規化は別スクリプト（本スクリプトは理論値マップのみ）。

出力：data/processed/fspl_map.npy  （shape (40,50), float32）

使い方:
  python theoretical_map.py                    # pathloss_params.json を読んで生成
  python theoretical_map.py --p0 -24.14 --n 1.85   # 値を直接指定
"""

import argparse
import os
import json
import numpy as np

from data_gen_lib import TX_X, TX_Y, D0, grid_center_coords


def build_fspl_map(P0, n, d0=D0, tx=(TX_X, TX_Y)):
    """40x50 の理論値マップ（F_FSPL, dBm）を返す。"""
    X, Y = grid_center_coords()                          # (40,50) 物理座標[m]
    d = np.sqrt((X - tx[0]) ** 2 + (Y - tx[1]) ** 2)     # 送信機距離[m]
    fspl = P0 - 10.0 * n * np.log10(np.maximum(d, d0) / d0)
    return fspl.astype(np.float32), d


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description="理論値マップ生成（スクリプト4）")
    ap.add_argument("--processed-root",
                    default=os.path.join(root, "data", "processed"))
    ap.add_argument("--params", default=None,
                    help="pathloss_params.json のパス（既定: processed/pathloss_params.json）")
    ap.add_argument("--p0", type=float, default=None, help="P_0 を直接指定")
    ap.add_argument("--n", type=float, default=None, help="n を直接指定")
    ap.add_argument("--out", default=None, help="出力 .npy パス")
    args = ap.parse_args()

    if args.p0 is not None and args.n is not None:
        P0, n, d0, tx = args.p0, args.n, D0, (TX_X, TX_Y)
        print(f"直接指定: P0={P0}, n={n}")
    else:
        params_path = args.params or os.path.join(args.processed_root,
                                                   "pathloss_params.json")
        with open(params_path, "r", encoding="utf-8") as fp:
            p = json.load(fp)
        P0, n, d0 = p["P0"], p["n"], p.get("d0", D0)
        tx = tuple(p.get("tx", (TX_X, TX_Y)))
        print(f"パラメータ読込 {params_path}: P0={P0:.4f}, n={n:.4f}, d0={d0}, tx={tx}")

    fspl, d = build_fspl_map(P0, n, d0, tx)
    out = args.out or os.path.join(args.processed_root, "fspl_map.npy")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.save(out, fspl)
    print(f"理論値マップ shape={fspl.shape} 値域=[{fspl.min():.2f}, {fspl.max():.2f}] dBm")
    print(f"  送信機セル(最大値)位置: {np.unravel_index(np.argmax(fspl), fspl.shape)}")
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
