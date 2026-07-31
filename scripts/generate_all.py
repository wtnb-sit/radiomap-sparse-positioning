"""全サンプル一括処理ドライバ

各サンプルについて：
  1) 障害物マップ（正解ラベル）を1回だけ生成      -> data/processed/labels/label_<id>.npy
  2) 観測点数 100..1200（100刻み・12条件）で欠損付与 -> data/processed/obs<N>/sparse_<id>.npy

rxPowerXXXX.txt と objectXXXX.object のペアを探索する。
探索先は --raw-dir（既定 data/raw）。無ければプロジェクトルート直下も探す。
env_id（シード兼用）はファイル名の数字部分（例：1001）を使う。

使い方:
  python generate_all.py
  python generate_all.py --raw-dir ../data/raw --out-root ../data/processed
"""

import argparse
import os
import re
import glob
import numpy as np

from data_gen_lib import load_rss_map, N_ROWS, N_COLS
from obstacle_map import build_obstacle_map
from inject_missing import inject_missing

OBS_COUNTS = list(range(100, 1201, 100))   # 100,200,...,1200 (12条件)


def _sample_id(path):
    m = re.search(r'(\d+)', os.path.basename(path))
    return m.group(1) if m else None


def _find_object_for(sid, raw_dir, root):
    """sid に対応する objectファイルを、想定しうる場所から探して返す（無ければ None）。"""
    candidates = [
        os.path.join(raw_dir, "object", f"object{sid}.object"),  # raw/object/ サブフォルダ
        os.path.join(raw_dir, f"object{sid}.object"),            # raw/ 直下（フラット）
        os.path.join(root, f"object{sid}.object"),               # プロジェクト直下
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def find_pairs(raw_dir, root):
    """(rss_path, object_path, sample_id) のリストを返す。

    rxPowerファイルは以下のいずれの配置でも探索する:
      - data/raw/rxPower/rxPowerXXXX.txt  (rxPower サブフォルダ)
      - data/raw/rxPowerXXXX.txt          (raw 直下・フラット)
      - <project>/rxPowerXXXX.txt         (プロジェクト直下)
    objectファイルは _find_object_for が同様に複数箇所を探す。
    """
    rss_dirs = [
        os.path.join(raw_dir, "rxPower"),  # raw/rxPower/ サブフォルダ
        raw_dir,                           # raw/ 直下
        root,                              # プロジェクト直下
    ]
    pairs = {}
    for d in rss_dirs:
        for rss in glob.glob(os.path.join(d, "rxPower*.txt")):
            sid = _sample_id(rss)
            if sid is None or sid in pairs:
                continue
            obj = _find_object_for(sid, raw_dir, root)
            if obj:
                pairs[sid] = (rss, obj, sid)
            else:
                print(f"[WARN] id={sid}: 対応するobjectファイルが見つかりません（スキップ）")
    return [pairs[k] for k in sorted(pairs.keys())]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description="全サンプル一括生成ドライバ")
    ap.add_argument("--raw-dir", default=os.path.join(root, "data", "raw"))
    ap.add_argument("--out-root", default=os.path.join(root, "data", "processed"))
    args = ap.parse_args()

    pairs = find_pairs(args.raw_dir, root)
    if not pairs:
        print(f"[WARN] rxPower*.txt / object*.object のペアが見つかりません "
              f"(探索: {args.raw_dir}, {root})")
        return

    labels_dir = os.path.join(args.out_root, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    print(f"サンプル数: {len(pairs)}  観測点条件: {OBS_COUNTS}")
    warn_samples = []

    for rss_path, obj_path, sid in pairs:
        env_id = int(sid)

        # --- スクリプト1：障害物マップ（1回だけ） ---
        obstacle_map, obstacles = build_obstacle_map(obj_path)
        n_free = int((obstacle_map == 0).sum())
        label_path = os.path.join(labels_dir, f"label_{sid}.npy")
        np.save(label_path, obstacle_map)

        # --- スクリプト2：各観測点数で欠損付与 ---
        rss_map, filled = load_rss_map(rss_path)
        if filled != N_ROWS * N_COLS:
            print(f"  [WARN] {sid}: RSS充填 {filled}/{N_ROWS*N_COLS}")

        for num_obs in OBS_COUNTS:
            if n_free < num_obs:
                warn_samples.append((sid, num_obs, n_free))
            obs_dir = os.path.join(args.out_root, f"obs{num_obs}")
            os.makedirs(obs_dir, exist_ok=True)
            sparse, n_sel, _ = inject_missing(rss_map, obstacle_map,
                                              num_obs, env_id)
            np.save(os.path.join(obs_dir, f"sparse_{sid}.npy"), sparse)

        print(f"[OK] id={sid}: 障害物{len(obstacles)}個 / "
              f"障害物セル{int(obstacle_map.sum())} / フリースペース{n_free} "
              f"-> label + {len(OBS_COUNTS)}条件のsparse")

    if warn_samples:
        print("\n[安全チェック警告] フリースペース数 < 観測点数 の組:")
        for sid, num_obs, n_free in warn_samples:
            print(f"  id={sid}: num_obs={num_obs} > freespace={n_free}")
    else:
        print("\n[安全チェック] 全サンプルでフリースペース数 >= 全観測点数条件 OK")

    print("完了。")


if __name__ == "__main__":
    main()
