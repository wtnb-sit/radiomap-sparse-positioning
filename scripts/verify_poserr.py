"""測位誤差付与の1サンプル検証。

  ・欠損付与後(誤差なし)と 測位誤差付与後 を並べて可視化
  ・観測点が少しずつ「ズレて」いることを目視確認
  ・変位量の統計がレイリー分布（平均0.61m）と整合するか数値確認

出力：verify_poserr_<id>_obs<N>.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_gen_lib import MISSING_VALUE, N_ROWS, N_COLS
from add_positioning_error import add_positioning_error, SIGMA, RESOLUTION, _round_away


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    sid, num_obs = 3000, 600

    sparse_path = os.path.join(root, "data", "processed",
                               f"obs{num_obs}", f"sparse_{sid}.npy")
    sparse = np.load(sparse_path)
    out, stats = add_positioning_error(sparse, num_obs, sid)
    print(f"id={sid} obs{num_obs}: {stats}")

    # --- 変位量の統計チェック（レイリー平均0.61mと整合するか） ---
    n = stats["n_in"]
    rng = np.random.default_rng([sid, num_obs])
    e_x = rng.normal(0.0, SIGMA, size=n)
    e_y = rng.normal(0.0, SIGMA, size=n)
    mag = np.sqrt(e_x**2 + e_y**2)          # 連続変位の大きさ [m]
    print(f"[CHECK] 変位量(連続)の平均: {mag.mean():.3f} m (理論 0.61m)")
    print(f"[CHECK] σ(各軸): 実測 x={e_x.std():.3f} y={e_y.std():.3f} (設定 {SIGMA:.3f})")
    d_col = _round_away(e_x / RESOLUTION).astype(int)
    d_row = _round_away(e_y / RESOLUTION).astype(int)
    print(f"[CHECK] グリッド変位 |Δ| 最大: 行={np.abs(d_row).max()} 列={np.abs(d_col).max()} セル")
    print(f"[CHECK] 変位0セル(動かない点)の割合: "
          f"{np.mean((d_row==0)&(d_col==0))*100:.1f}%")

    extent = [0.0, 10.0, 0.0, 8.0]
    before = np.ma.masked_where(sparse <= MISSING_VALUE + 1, sparse)
    after = np.ma.masked_where(out <= MISSING_VALUE + 1, out)
    vmin = min(sparse[sparse > MISSING_VALUE + 1].min(),
               out[out > MISSING_VALUE + 1].min())
    vmax = max(sparse.max(), out.max())

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, data, title in [
        (axes[0], before, f"Before (missing only, obs={num_obs})"),
        (axes[1], after, f"After positioning error (obs={num_obs})"),
    ]:
        im = ax.imshow(data, cmap="viridis", extent=extent, aspect="auto",
                       vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"Sample {sid}: positioning-error injection "
                 f"(in={stats['n_in']} -> out={stats['n_out']}, "
                 f"collide={stats['collisions']})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = os.path.join(root, f"verify_poserr_{sid}_obs{num_obs}.png")
    fig.savefig(out_png, dpi=110)
    print(f"[SAVED] {out_png}")


if __name__ == "__main__":
    main()
