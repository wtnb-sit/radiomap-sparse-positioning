"""Step 4：PyTorch Dataset（提案手法・条件3：正規化済み残差 + マスク）

各サンプルについて以下を返す:
    input : (1,40,50) float32  正規化済み残差（欠損セルは0埋め、マスクで無効化）
    mask  : (1,40,50) float32  観測点=1 / 欠損=0
    target: (40,50)   long     正解ラベル（フリースペース=0 / 障害物=1）

--- 仕様（research_design_final.md §5, §7） ---
  ・入力は residual_norm/obs<N>/resnorm_<id>.npy（サンプルごとz-score済み）
  ・マスクは読み込み時に導出（mask_util.make_mask）
  ・PConv入力は欠損を0埋め（マスクで無効化されるため値は不問だが0が安全）
  ・分割は splits.py（train 1001-5000 / val 5001-5500 / test 5501-6000）
  ・観測点数条件 num_obs を指定（100〜1200）

生RSS入力（条件1/2）は将来 input_type='raw' で追加予定（本実装では未対応）。
"""

import os
import glob
import re
import numpy as np
import torch
from torch.utils.data import Dataset

from data_gen_lib import MISSING_VALUE, N_ROWS, N_COLS
from mask_util import make_mask
from splits import split_of, ids_of


class RadioMapDataset(Dataset):
    """入力（残差 or 生RSS）+ マスク + ラベル を返す Dataset。

    input_type:
      'residual'（条件①③）: residual_norm/obs<N>/resnorm_<id>.npy（正規化済み）
      'raw'    （条件②）  : poserr/obs<N>/sparse_<id>.npy（測位誤差付き生RSS）を
                            有効観測点のμ,σでサンプルごとz-score正規化して使用
    いずれも欠損は0埋め、マスクは疎パターンから導出。
    """

    _CONFIG = {
        "residual": ("residual_norm", "resnorm_", False),  # (dir, prefix, need_norm)
        "raw":      ("poserr",        "sparse_",  True),
        # 測位誤差なし(clean)の正規化残差。make_clean_residual.py が生成。
        # shape誤差が測位誤差由来かを切り分ける実験用（既存2キーの挙動は不変）。
        "clean_residual": ("clean_residual_norm", "resnorm_", False),
    }

    def __init__(self, processed_root, split, num_obs, input_type="residual"):
        if input_type not in self._CONFIG:
            raise ValueError(f"input_type は {list(self._CONFIG)} のいずれか")
        subdir, prefix, need_norm = self._CONFIG[input_type]
        self.split = split
        self.num_obs = num_obs
        self.input_type = input_type
        self.prefix = prefix
        self.need_norm = need_norm
        self.in_dir = os.path.join(processed_root, subdir, f"obs{num_obs}")
        self.label_dir = os.path.join(processed_root, "labels")

        # 分割に含まれ、入力・ラベル両方が存在するIDだけを採用
        want = set(ids_of(split))
        self.ids = []
        for f in sorted(glob.glob(os.path.join(self.in_dir, f"{prefix}*.npy"))):
            sid = int(re.search(r'(\d+)', os.path.basename(f)).group(1))
            if sid in want and os.path.exists(
                    os.path.join(self.label_dir, f"label_{sid}.npy")):
                self.ids.append(sid)
        if not self.ids:
            raise RuntimeError(
                f"サンプルが見つかりません（{self.in_dir}, split={split}）")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        arr = np.load(os.path.join(self.in_dir, f"{self.prefix}{sid}.npy"))
        label = np.load(os.path.join(self.label_dir, f"label_{sid}.npy"))

        mask = make_mask(arr)                           # (40,50) uint8
        vals = arr[mask == 1]
        if self.need_norm:
            # 生RSSをサンプルごとz-score（有効観測点の統計量、ddof=0）
            mu = float(vals.mean())
            sigma = float(vals.std(ddof=0))
            sigma = sigma if sigma > 1e-8 else 1.0
            x = np.zeros_like(arr, dtype=np.float32)
            x[mask == 1] = ((vals - mu) / sigma).astype(np.float32)
        else:
            # residual は正規化済み。欠損は0埋め。
            x = np.where(mask == 1, arr, 0.0).astype(np.float32)

        x_t = torch.from_numpy(x)[None]                 # (1,40,50)
        mask_t = torch.from_numpy(mask.astype(np.float32))[None]
        target_t = torch.from_numpy(label.astype(np.int64))  # (40,50) long
        return x_t, mask_t, target_t


def sqrt_inverse_freq_weights(processed_root, split="train", ids=None):
    """sqrt逆頻度によるクラス重み [w_free, w_obst] を返す（合計=2に正規化）。

    重み w_c ∝ 1/sqrt(freq_c)。障害物が少数(約1.1%)のため少数側を重くする。
    完全逆頻度(1/freq)より緩やかで、Dice併用時の過検出を抑える。
    """
    label_dir = os.path.join(processed_root, "labels")
    if ids is None:
        ids = ids_of(split)
    n_obst = 0
    n_total = 0
    for sid in ids:
        p = os.path.join(label_dir, f"label_{sid}.npy")
        if not os.path.exists(p):
            continue
        a = np.load(p)
        n_obst += int(a.sum())
        n_total += a.size
    p_obst = n_obst / n_total
    p_free = 1.0 - p_obst
    w_free = 1.0 / np.sqrt(p_free)
    w_obst = 1.0 / np.sqrt(p_obst)
    s = w_free + w_obst
    return np.array([2 * w_free / s, 2 * w_obst / s], dtype=np.float32)


if __name__ == "__main__":
    # 簡易動作確認
    import argparse
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap.add_argument("--processed-root", default=os.path.join(root, "data", "processed"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--num-obs", type=int, default=100)
    args = ap.parse_args()

    ds = RadioMapDataset(args.processed_root, args.split, args.num_obs)
    print(f"split={args.split} obs{args.num_obs}: サンプル数={len(ds)}")
    x, m, y = ds[0]
    print(f"input {tuple(x.shape)} {x.dtype}  mask {tuple(m.shape)}  target {tuple(y.shape)} {y.dtype}")
    print(f"input 値域[{x.min():.3f},{x.max():.3f}]  mask和={int(m.sum())}  "
          f"target クラス{sorted(torch.unique(y).tolist())} 障害物画素={int(y.sum())}")
    w = sqrt_inverse_freq_weights(args.processed_root, args.split)
    print(f"クラス重み(sqrt逆頻度) [free,obst]={w}  比={w[1]/w[0]:.2f}:1")
