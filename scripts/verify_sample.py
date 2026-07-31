"""1サンプル動作確認：RSSマップと障害物マップが同じ座標系で重なるか可視化。

出力：verify_1001.png
  (1) 完全なRSSマップ（40x50, dBm）
  (2) 障害物マップ（正解ラベル、障害物=1）
  (3) RSSマップに障害物輪郭を重ねたオーバーレイ
  (4) 欠損付与後（num_obs=100）のRSSマップ + 障害物輪郭

同じ座標変換・Y反転を通っているので、障害物の輪郭が
RSSマップの対応領域にぴったり重なることを目視で検証できる。
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_gen_lib import load_rss_map, MISSING_VALUE
from obstacle_map import build_obstacle_map
from inject_missing import inject_missing


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    rss_path = os.path.join(root, "rxPower1001.txt")
    object_path = os.path.join(root, "object1001.object")

    # --- 生成 ---
    rss_map, filled = load_rss_map(rss_path)
    obstacle_map, obstacles = build_obstacle_map(object_path)
    sparse, n_sel, n_free = inject_missing(rss_map, obstacle_map,
                                           num_obs=100, env_id=1001)

    print(f"RSS充填: {filled}/2000")
    print(f"障害物: {len(obstacles)}個  -> {obstacles}")
    print(f"障害物セル数: {int(obstacle_map.sum())}")
    # 障害物セルの (row,col) 範囲を表示（0始まり）
    rc = np.argwhere(obstacle_map == 1)
    if len(rc):
        print(f"障害物セル row範囲(0始): {rc[:,0].min()}..{rc[:,0].max()}, "
              f"col範囲(0始): {rc[:,1].min()}..{rc[:,1].max()}")
    print(f"欠損付与後 観測点: {n_sel}, フリースペース総数: {n_free}")

    # 表示範囲（extent）：物理座標 x:0..10, y:0..8 で、imshow origin='upper'
    # row0 が y最大側（上）に来るように extent=[x_left, x_right, y_bottom, y_top]
    extent = [0.0, 10.0, 0.0, 8.0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) 完全RSSマップ
    ax = axes[0, 0]
    im = ax.imshow(rss_map, cmap="viridis", extent=extent, aspect="auto")
    ax.set_title("(1) Full RSS map [dBm]")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (2) 障害物マップ
    ax = axes[0, 1]
    im = ax.imshow(obstacle_map, cmap="gray_r", extent=extent, aspect="auto",
                   vmin=0, vmax=1)
    ax.set_title("(2) Obstacle map (label: obstacle=1)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (3) RSS + 障害物輪郭オーバーレイ
    ax = axes[1, 0]
    im = ax.imshow(rss_map, cmap="viridis", extent=extent, aspect="auto")
    # contour は行列座標→物理座標に合わせるため、同じextentのグリッドを作る
    ax.contour(obstacle_map, levels=[0.5], colors="red", linewidths=2.0,
               extent=extent, origin="upper")
    ax.set_title("(3) Overlay: RSS + obstacle outline (red)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (4) 欠損付与後 + 障害物輪郭
    ax = axes[1, 1]
    masked = np.ma.masked_where(sparse <= MISSING_VALUE + 1, sparse)
    ax.imshow(obstacle_map, cmap="Greys", extent=extent, aspect="auto",
              vmin=0, vmax=3, alpha=0.3)   # 背景に障害物を薄く
    im = ax.imshow(masked, cmap="viridis", extent=extent, aspect="auto")
    ax.contour(obstacle_map, levels=[0.5], colors="red", linewidths=2.0,
               extent=extent, origin="upper")
    ax.set_title(f"(4) Sparse RSS (num_obs=100) + obstacle outline")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Sample 1001: coordinate-system alignment check", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = os.path.join(root, "verify_1001.png")
    fig.savefig(out_png, dpi=110)
    print(f"[SAVED] {out_png}")

    # --- 数値サニティチェック ---
    # 観測点はすべてフリースペース上にあるはず
    obs_rc = np.argwhere(sparse > MISSING_VALUE + 1)
    on_obstacle = sum(obstacle_map[r, c] == 1 for r, c in obs_rc)
    print(f"[CHECK] 観測点が障害物上にある数: {on_obstacle} (0であるべき)")

    # 包含関係チェック：num_obs=100 ⊂ num_obs=200
    sparse100, _, _ = inject_missing(rss_map, obstacle_map, 100, 1001)
    sparse200, _, _ = inject_missing(rss_map, obstacle_map, 200, 1001)
    set100 = set(map(tuple, np.argwhere(sparse100 > MISSING_VALUE + 1)))
    set200 = set(map(tuple, np.argwhere(sparse200 > MISSING_VALUE + 1)))
    print(f"[CHECK] 包含関係 100⊂200: {set100.issubset(set200)} "
          f"(|100|={len(set100)}, |200|={len(set200)})")


if __name__ == "__main__":
    main()
