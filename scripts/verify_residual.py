"""残差計算の1サンプル検証。

  (1) 入力：測位誤差付き 疎なRSS（poserr, obs600）
  (2) 出力：疎な残差マップ（RSS - F_FSPL）
  (3) 疎パターンが不変か（有効セル位置の一致）を数値確認
  (4) 残差ヒストグラム（0付近に集中するか）

出力：verify_residual_<id>_obs<N>.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_gen_lib import MISSING_VALUE
from compute_residual import compute_residual


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    processed = os.path.join(root, "data", "processed")
    sid, num_obs = 3000, 600

    sparse = np.load(os.path.join(processed, "poserr", f"obs{num_obs}",
                                  f"sparse_{sid}.npy"))
    fspl = np.load(os.path.join(processed, "fspl_map.npy"))
    residual, n_valid = compute_residual(sparse, fspl)

    # --- 疎パターン不変の検証 ---
    m_in = sparse > MISSING_VALUE + 1
    m_out = residual > MISSING_VALUE + 1
    same_pattern = np.array_equal(m_in, m_out)
    print(f"id={sid} obs{num_obs}: 観測点={n_valid}")
    print(f"[CHECK] 疎パターン不変(入力と出力の有効セル一致): {same_pattern}")

    # --- 値の妥当性 ---
    rin = sparse[m_in]
    rout = residual[m_out]
    print(f"[CHECK] 入力RSS  値域[{rin.min():.2f},{rin.max():.2f}] 平均{rin.mean():.2f} 標準偏差{rin.std():.2f}")
    print(f"[CHECK] 残差     値域[{rout.min():.2f},{rout.max():.2f}] 平均{rout.mean():.2f} 標準偏差{rout.std():.2f}")
    print(f"[CHECK] 残差が0付近に集中(|平均|が入力より小): "
          f"{abs(rout.mean()) < abs(rin.mean())}")

    extent = [0.0, 10.0, 0.0, 8.0]
    before = np.ma.masked_where(~m_in, sparse)
    after = np.ma.masked_where(~m_out, residual)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    ax = axes[0]
    im = ax.imshow(before, cmap="viridis", extent=extent, aspect="auto")
    ax.set_title(f"(1) Input: sparse RSS (poserr, obs={num_obs}) [dBm]")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    vmax = np.abs(rout).max()
    im = ax.imshow(after, cmap="RdBu", extent=extent, aspect="auto",
                   vmin=-vmax, vmax=vmax)
    ax.set_title("(2) Output: residual = RSS - F_FSPL [dB]")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2]
    ax.hist(rin, bins=40, alpha=0.5, label=f"input RSS (mean {rin.mean():.1f})")
    ax.hist(rout, bins=40, alpha=0.5, label=f"residual (mean {rout.mean():.1f})")
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_title("(3) Distribution: raw RSS vs residual")
    ax.set_xlabel("value [dB]"); ax.set_ylabel("count")
    ax.legend()

    fig.suptitle(f"Sample {sid}: residual computation (obs={num_obs}, "
                 f"n_valid={n_valid})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = os.path.join(root, f"verify_residual_{sid}_obs{num_obs}.png")
    fig.savefig(out_png, dpi=110)
    print(f"[SAVED] {out_png}")


if __name__ == "__main__":
    main()
