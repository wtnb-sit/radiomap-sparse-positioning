"""理論値マップの検証。

  (1) 理論値マップ F_FSPL（40x50, 距離減衰のみ・全データ共通）
  (2) ある1サンプルの完全RSS（参考: 通常は使わないが検証用）
  (3) 残差 = 完全RSS - F_FSPL（距離減衰が除かれ、障害物が浮き出る）
      → 障害物輪郭を重ねて、残差の落ち込みが障害物と一致するか確認
  (4) 回帰の散布図（RSS vs log距離）と回帰直線

出力：verify_fspl.png
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_gen_lib import load_rss_map, TX_X, TX_Y, D0, grid_center_coords, MISSING_VALUE
from obstacle_map import build_obstacle_map
from theoretical_map import build_fspl_map


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    processed = os.path.join(root, "data", "processed")
    sid = 3000

    with open(os.path.join(processed, "pathloss_params.json"), encoding="utf-8") as fp:
        p = json.load(fp)
    P0, n = p["P0"], p["n"]

    fspl, dist = build_fspl_map(P0, n)
    rss_full, _ = load_rss_map(os.path.join(root, "data", "raw", "rxPower",
                                            f"rxPower{sid}.txt"))
    obstacle_map, _ = build_obstacle_map(os.path.join(root, "data", "raw", "object",
                                                      f"object{sid}.object"))
    residual = rss_full - fspl

    print(f"P0={P0:.3f}, n={n:.3f}, R2={p['r2']:.3f}")
    print(f"FSPL 値域 [{fspl.min():.2f}, {fspl.max():.2f}]")
    print(f"残差 値域 [{residual.min():.2f}, {residual.max():.2f}] 平均 {residual.mean():.2f}")
    # 障害物セルとフリースペースセルで残差平均を比較
    obst = obstacle_map == 1
    print(f"残差平均: 障害物セル={residual[obst].mean():.2f} dB, "
          f"フリースペース={residual[~obst].mean():.2f} dB "
          f"（障害物側が低いはず）")

    extent = [0.0, 10.0, 0.0, 8.0]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # (1) 理論値マップ
    ax = axes[0, 0]
    im = ax.imshow(fspl, cmap="viridis", extent=extent, aspect="auto")
    ax.plot(TX_X, TX_Y, "r*", markersize=16, label="Tx (5,4)")
    ax.legend(loc="upper right")
    ax.set_title("(1) Theoretical map F_FSPL [dBm] (distance decay only)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (2) 完全RSS
    ax = axes[0, 1]
    im = ax.imshow(rss_full, cmap="viridis", extent=extent, aspect="auto")
    ax.set_title(f"(2) Full RSS (sample {sid}) [dBm]")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (3) 残差 + 障害物輪郭
    ax = axes[1, 0]
    im = ax.imshow(residual, cmap="RdBu", extent=extent, aspect="auto",
                   vmin=-np.abs(residual).max(), vmax=np.abs(residual).max())
    ax.contour(obstacle_map, levels=[0.5], colors="black", linewidths=2.0,
               extent=extent, origin="upper")
    ax.set_title("(3) Residual = RSS - F_FSPL (obstacle outline in black)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (4) 回帰散布図（サブサンプル）
    ax = axes[1, 1]
    X, Y = grid_center_coords()
    d_all = np.sqrt((X - TX_X) ** 2 + (Y - TX_Y) ** 2).ravel()
    rss_all = rss_full.ravel()
    xlog = np.log10(np.maximum(d_all, D0) / D0)
    ax.scatter(xlog, rss_all, s=6, alpha=0.35, label=f"sample {sid} (2000 pts)")
    xs = np.linspace(xlog.min(), xlog.max(), 50)
    ax.plot(xs, P0 - 10 * n * xs, "r-", linewidth=2,
            label=f"fit: P0={P0:.1f}, n={n:.2f}")
    ax.set_title("(4) Path-loss fit: RSS vs log10(d/d0)")
    ax.set_xlabel("log10(max(d,d0)/d0)"); ax.set_ylabel("RSS [dBm]")
    ax.legend()

    fig.suptitle("Theoretical map (F_FSPL) verification", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = os.path.join(root, "verify_fspl.png")
    fig.savefig(out_png, dpi=110)
    print(f"[SAVED] {out_png}")


if __name__ == "__main__":
    main()
