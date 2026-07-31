"""train/val/test 分割の定義（連番ID範囲）。

  train : 1001–5000 (4000サンプル)
  val   : 5001–5500 (500サンプル)
  test  : 5501–6000 (500サンプル)

回帰(P_0,n)は train のみ、評価は test で行う。他スクリプトはここを参照する。
"""

TRAIN_RANGE = (1001, 5000)
VAL_RANGE = (5001, 5500)
TEST_RANGE = (5501, 6000)


def split_of(env_id):
    """env_id が属する分割名 'train'/'val'/'test' を返す（範囲外は None）。"""
    i = int(env_id)
    if TRAIN_RANGE[0] <= i <= TRAIN_RANGE[1]:
        return "train"
    if VAL_RANGE[0] <= i <= VAL_RANGE[1]:
        return "val"
    if TEST_RANGE[0] <= i <= TEST_RANGE[1]:
        return "test"
    return None


def ids_of(split):
    """分割名からID範囲(range)を返す。"""
    rng = {"train": TRAIN_RANGE, "val": VAL_RANGE, "test": TEST_RANGE}[split]
    return range(rng[0], rng[1] + 1)
