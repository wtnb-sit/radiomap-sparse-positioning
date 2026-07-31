"""⑥ マスク生成（その場で導出する方式）

マスク M は「観測点=1 / 欠損=0」の2値マップ。

    M(p) = 1  （観測点が存在：値 > -250）
         = 0  （欠損：値 == -250）

--- 方針（本プロジェクトの決定） ---
  ・マスクは事前生成せず、dataset.py 等で読み込み時に都度導出する（方針B）。
  ・疎パターンは全段（poserr / residual / residual_norm）で不変なので、
    正規化済み残差マップ（-250センチネル）から常に復元できる。
  ・用途：PConvのマスク付き畳み込み、ボトルネックのフェイルセーフ（マスクConcat）。

使い方（dataset.py 側の想定）:
    import numpy as np
    from mask_util import make_mask
    resnorm = np.load("data/processed/residual_norm/obs600/resnorm_1001.npy")
    mask = make_mask(resnorm)                 # (40,50) uint8, 観測点=1 欠損=0
    x = np.where(mask == 1, resnorm, 0.0)     # PConv入力は欠損を0埋め（マスクで無効化）
"""

import numpy as np

from data_gen_lib import MISSING_VALUE


def make_mask(sparse_map, dtype=np.uint8):
    """疎マップ（-250=欠損）から2値マスクを導出して返す。

    Args:
        sparse_map: (H,W) 配列。欠損セルは MISSING_VALUE(-250)。
                    residual_norm / residual / poserr のいずれでも可（疎パターン同一）。
        dtype:      出力の型（既定 uint8）。

    Returns:
        mask: (H,W) 観測点=1 / 欠損=0
    """
    return (sparse_map > MISSING_VALUE + 1).astype(dtype)


def apply_mask_fill(sparse_map, fill=0.0):
    """欠損セルを fill（既定0）で埋めた入力と、マスクの組を返す。

    PConv入力の想定：欠損は0埋めし、マスクで有効領域を示す。

    Returns:
        x:    (H,W) float32（観測点=元の値、欠損=fill）
        mask: (H,W) uint8（観測点=1、欠損=0）
    """
    mask = make_mask(sparse_map)
    x = np.where(mask == 1, sparse_map, fill).astype(np.float32)
    return x, mask
