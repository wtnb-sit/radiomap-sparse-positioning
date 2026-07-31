"""事前解析：d_critical（臨界深度）の分布と、マスク埋め尽くしブロックの N,k 決定

マスク埋め尽くしブロック（PConvで疎な観測を密化する第1ブロック）が
全欠損セルに情報を届けるのに必要な受容野を、最悪ケース（観測点100個）で見積もる。

--- 手順（research_design_final.md §8.1） ---
  Step 1: 観測点100個（測位誤差後・学習データ）の各サンプルの欠損マスクを生成
  Step 2: 距離変換で臨界深度を計算
            DT(p) = min_{q∈V} ||p-q||           （最近傍観測点までの距離）
            d_critical = max_{p∈{M=0}} DT(p)     （最も遠い欠損セルの距離）
            ※ scipy.ndimage.distance_transform_edt
  Step 3: データセット全体の d_critical 分布を取得
  Step 4: r_eff ≥ d_critical を満たす最小 N,k を決定
            r_eff = (k-1)/2 × N   （ストライド1）、k ∈ {5,7,9}

単位はグリッドセル。入力は測位誤差後の疎パターン（poserr/obs100）。
評価は学習セット（1001–5000）。

使い方:
  python analyze_d_critical.py               # train obs100 全体で分布とN,k
  python analyze_d_critical.py --sample 3000 # 1サンプルの距離変換を可視化検証
"""

import argparse
import os
import glob
import re
import math
import numpy as np
from scipy.ndimage import distance_transform_edt

from data_gen_lib import MISSING_VALUE
from splits import split_of

K_CANDIDATES = [5, 7, 9]


def d_critical_of(sparse_map):
    """疎マップ（-250=欠損）から d_critical（セル）を計算して返す。

    DT(p) = 各欠損セルから最近傍の観測点までのユークリッド距離。
    d_critical = その最大値。
    """
    valid = sparse_map > MISSING_VALUE + 1          # 観測点=True
    if not valid.any():
        return float("nan"), None
    missing = ~valid                                 # 欠損=True
    # distance_transform_edt: True セルから最近傍 False(=観測点) までの距離
    dt = distance_transform_edt(missing)
    return float(dt.max()), dt


def _sample_id(path):
    m = re.search(r'(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else None


def min_layers(d_target, k):
    """r_eff=(k-1)/2·N ≥ d_target を満たす最小の層数 N。"""
    per_layer = (k - 1) / 2.0
    return int(math.ceil(d_target / per_layer))


def analyze_all(processed_root, num_obs=100, split="train"):
    in_dir = os.path.join(processed_root, "poserr", f"obs{num_obs}")
    files = sorted(glob.glob(os.path.join(in_dir, "sparse_*.npy")))
    dcs, ids = [], []
    for f in files:
        sid = _sample_id(f)
        if split_of(sid) != split:
            continue
        dc, _ = d_critical_of(np.load(f))
        dcs.append(dc)
        ids.append(sid)
    dcs = np.array(dcs)
    print(f"対象: {split} obs{num_obs}  サンプル数={len(dcs)}")
    print("\n=== d_critical 分布（セル） ===")
    for label, v in [
        ("最大", dcs.max()), ("99.9%", np.percentile(dcs, 99.9)),
        ("99%", np.percentile(dcs, 99)), ("95%", np.percentile(dcs, 95)),
        ("平均", dcs.mean()), ("中央", np.median(dcs)),
        ("最小", dcs.min()), ("標準偏差", dcs.std()),
    ]:
        print(f"  {label:>6}: {v:.3f}")

    print("\n=== N,k の決定（r_eff=(k-1)/2*N >= d_critical） ===")
    targets = {"最大(100%)": dcs.max(), "99%": np.percentile(dcs, 99),
               "95%": np.percentile(dcs, 95)}
    header = "カバレッジ  d_crit  " + "  ".join(f"k={k}(N,r_eff)" for k in K_CANDIDATES)
    print(header)
    for name, d_t in targets.items():
        row = f"{name:>10} {d_t:6.2f} "
        for k in K_CANDIDATES:
            N = min_layers(d_t, k)
            r_eff = (k - 1) / 2.0 * N
            row += f"  N={N},r_eff={r_eff:.0f}"
        print(row)

    # CSV & ヒストグラム
    csv_path = os.path.join(processed_root, "d_critical_stats.csv")
    with open(csv_path, "w", encoding="utf-8") as fp:
        fp.write("env_id,d_critical\n")
        for sid, dc in zip(ids, dcs):
            fp.write(f"{sid},{dc:.4f}\n")
    print(f"\n[SAVED] {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        root = os.path.dirname(processed_root.rstrip(os.sep))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(dcs, bins=40, color="steelblue", edgecolor="white")
        ax.axvline(dcs.max(), color="red", ls="--", label=f"max={dcs.max():.2f}")
        ax.axvline(np.percentile(dcs, 99), color="orange", ls="--",
                   label=f"99%={np.percentile(dcs,99):.2f}")
        ax.set_title(f"d_critical distribution ({split} obs{num_obs}, n={len(dcs)})")
        ax.set_xlabel("d_critical [cells]"); ax.set_ylabel("count")
        ax.legend()
        out_png = os.path.join(root, "d_critical_hist.png")
        fig.tight_layout(); fig.savefig(out_png, dpi=110)
        print(f"[SAVED] {out_png}")
    except Exception as e:
        print(f"[WARN] ヒストグラム描画スキップ: {e}")


def verify_sample(processed_root, sid, num_obs=100):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = os.path.join(processed_root, "poserr", f"obs{num_obs}", f"sparse_{sid}.npy")
    sparse = np.load(f)
    dc, dt = d_critical_of(sparse)
    valid = sparse > MISSING_VALUE + 1
    # d_critical を与えるセル位置
    p = np.unravel_index(np.argmax(dt), dt.shape)
    print(f"id={sid} obs{num_obs}: 観測点={int(valid.sum())}  d_critical={dc:.3f} セル  "
          f"最遠欠損セル(row,col)={p}")

    extent = [0.0, 10.0, 0.0, 8.0]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    ax.imshow(valid, cmap="Greys", extent=extent, aspect="auto", vmin=0, vmax=1)
    ax.set_title(f"(1) Observation mask (obs{num_obs}, n={int(valid.sum())})")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    ax = axes[1]
    im = ax.imshow(dt, cmap="viridis", extent=extent, aspect="auto")
    # 最遠欠損セルを物理座標でマーク
    px = (p[1] + 0.5) * 0.2
    py = (40 - p[0] - 0.5) * 0.2
    ax.plot(px, py, "r*", markersize=18, label=f"d_critical={dc:.2f} cells")
    ax.set_title("(2) Distance transform (dist to nearest observation) [cells]")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"Sample {sid}: d_critical verification", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    root = os.path.dirname(processed_root.rstrip(os.sep))
    out_png = os.path.join(root, f"verify_dcritical_{sid}.png")
    fig.savefig(out_png, dpi=110)
    print(f"[SAVED] {out_png}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description="d_critical 事前解析")
    ap.add_argument("--processed-root",
                    default=os.path.join(root, "data", "processed"))
    ap.add_argument("--num-obs", type=int, default=100, help="最悪ケース観測点数（既定100）")
    ap.add_argument("--split", default="train")
    ap.add_argument("--sample", type=int, default=None,
                    help="指定すると1サンプルの距離変換を可視化検証")
    args = ap.parse_args()

    if args.sample is not None:
        verify_sample(args.processed_root, args.sample, args.num_obs)
    else:
        analyze_all(args.processed_root, args.num_obs, args.split)


if __name__ == "__main__":
    main()
