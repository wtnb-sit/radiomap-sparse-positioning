"""共通ライブラリ：座標変換・objectパース・RSS読み込み

claude_code_instruction.md / data_generation_spec.md の仕様に準拠。

グリッド：40行 x 50列（解像度 0.2 m/cell）
  列インデックス col (1..50) : col = round(x * 5)
  行インデックス row (1..40) : row = 41 - round(y * 5)   ※Y反転
  四捨五入は「0.5を常に切り上げ」(MATLAB round準拠) = floor(v*5 + 0.5)

  グリッド中心座標（障害物判定用）：
    x_center = col     * 0.2 - 0.1
    y_center = row_raw * 0.2 - 0.1   (row_raw = 反転前の行 = round(y*5))
"""

import re
import numpy as np

# ---- グリッド定数 --------------------------------------------------------
N_ROWS = 40
N_COLS = 50
RESOLUTION = 0.2          # m/cell
MISSING_VALUE = -250.0    # 欠損値
OBSTACLE = 1              # 障害物ラベル
FREESPACE = 0             # フリースペースラベル

# ---- 送信機・パスロス関連定数 --------------------------------------------
# 送信機位置（物理座標[m]）：部屋中心 = グリッド(行20,列25)近傍。全データ共通・固定。
TX_X = 5.0
TX_Y = 4.0
D0 = 0.2                  # 基準距離[m]（= 1セル、log発散回避）


def grid_center_coords():
    """40x50 の各グリッド中心の物理座標 (X, Y)[m] を返す（RSS/ラベルと同じ向き）。

    出力の行 index0 = 上端（y最大, row_raw=40, y=7.9）、
             行 index39 = 下端（y最小, row_raw=1,  y=0.1）。
    列 index0 = 左端（x=0.1）、列 index49 = 右端（x=9.9）。

    Returns: X, Y  （ともに shape (40,50)）
    """
    cols = np.arange(1, N_COLS + 1)              # 1..50
    x = cols * RESOLUTION - 0.1                  # 各列の中心X（0.1..9.9）
    rows = np.arange(1, N_ROWS + 1)              # 最終行 1..40（Y反転後）
    row_raw = 41 - rows                          # 反転前 40..1
    y = row_raw * RESOLUTION - 0.1               # 各行の中心Y（7.9..0.1）
    X, Y = np.meshgrid(x, y)                     # (40,50)
    return X, Y


# ---- 座標変換 ------------------------------------------------------------
def col_from_x(x):
    """列インデックス col (1..50) = round_half_up(x * 5)。"""
    return int(np.floor(float(x) * 5.0 + 0.5))


def row_raw_from_y(y):
    """反転前の行インデックス row_raw (1..40) = round_half_up(y * 5)。"""
    return int(np.floor(float(y) * 5.0 + 0.5))


def xy_to_rowcol(x, y):
    """(x, y)[m] を (row, col)（ともに1始まり、Y反転済み）に変換して返す。"""
    col = col_from_x(x)
    row_raw = row_raw_from_y(y)
    row = 41 - row_raw          # Y反転
    return row, col


def x_center_of_col(col):
    """列 col (1..50) のグリッド中心 X 座標 [m]。"""
    return col * RESOLUTION - 0.1


def y_center_of_row_raw(row_raw):
    """行 row_raw (1..40, 反転前) のグリッド中心 Y 座標 [m]。"""
    return row_raw * RESOLUTION - 0.1


# ---- objectファイルのパース ---------------------------------------------
# 「小数 小数 小数」（スペース区切りで浮動小数が3つだけ）の行 = 頂点座標行
_VERTEX_RE = re.compile(
    r'^\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$'
)
# 障害物ブロックの境界。begin_<structure_group> や begin_<sub_structure> とは
# 区別するため、"begin_<structure>" の直後が空白か行末であることを要求する。
_STRUCTURE_RE = re.compile(r'^\s*begin_<structure>\s*$')


def parse_obstacles(object_path):
    """objectファイルから障害物ごとの (x_min, x_max, y_min, y_max) を返す。

    begin_<structure> ... で1つの直方体障害物。各structure内の頂点座標行
    （小数3つの行）から X, Y の min/max を計算する。CRLF改行に対応。

    Returns: list[tuple(x_min, x_max, y_min, y_max)]
    """
    with open(object_path, 'r', encoding='utf-8', errors='replace') as f:
        # CRLF(\r\n) / CR(\r) / LF(\n) いずれでも行分割できるよう splitlines を使う
        lines = f.read().splitlines()

    obstacles = []
    current_xs = None
    current_ys = None
    in_structure = False

    def flush():
        nonlocal current_xs, current_ys
        if current_xs:  # 頂点が1つ以上あれば確定
            obstacles.append((min(current_xs), max(current_xs),
                              min(current_ys), max(current_ys)))
        current_xs, current_ys = None, None

    for line in lines:
        if _STRUCTURE_RE.match(line):
            # 直前のstructureを確定してから新規開始
            flush()
            in_structure = True
            current_xs, current_ys = [], []
            continue

        if not in_structure:
            continue

        m = _VERTEX_RE.match(line)
        if m:
            x = float(m.group(1))
            y = float(m.group(2))
            # z = float(m.group(3))  # 2次元判定では未使用
            current_xs.append(x)
            current_ys.append(y)

    # 最後のstructureを確定
    flush()
    return obstacles


# ---- RSSファイルの読み込み ----------------------------------------------
def load_rss_map(rss_path):
    """rxPowerXXXX.txt を読み込み、40x50 のRSS行列（dBm）に変換して返す。

    ヘッダ3行をスキップし、各データ行の X(2列目), Y(3列目), Power(6列目) を使う。
    座標変換・Y反転を適用してグリッドに配置する。

    Returns: np.ndarray shape (40, 50), dtype float
    """
    rss = np.full((N_ROWS, N_COLS), np.nan, dtype=float)

    with open(rss_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()

    filled = 0
    for line in lines:
        parts = line.split()
        # データ行は「番号 X Y Z Distance Power Phase」= 7要素。1列目が整数。
        if len(parts) < 6:
            continue
        try:
            idx = int(parts[0])            # 番号（整数）でヘッダを弾く
        except ValueError:
            continue
        try:
            x = float(parts[1])
            y = float(parts[2])
            power = float(parts[5])
        except ValueError:
            continue

        row, col = xy_to_rowcol(x, y)
        if 1 <= row <= N_ROWS and 1 <= col <= N_COLS:
            rss[row - 1, col - 1] = power
            filled += 1

    return rss, filled
