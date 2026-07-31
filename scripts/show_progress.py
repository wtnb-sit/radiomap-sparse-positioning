"""show_progress.py：実験の進捗確認（読み取り専用・学習中に実行してよい）

`run_experiments.py > log.txt 2>&1` のようにリダイレクトすると、Python の
標準出力バッファリングでログがすぐファイルに現れない（GPUは動いていても
無反応に見える）。一方 train.py は train_log.csv を**毎エポック
開く→追記→閉じる**ため、こちらはエポック単位で即ディスクに反映される。

本スクリプトは runs/ を走査して 69本（Step A 33 + Step B 36）の進捗表を出す。
ファイルを読むだけなので、学習中に別の Anaconda Prompt から安全に実行できる。

状態の意味:
  評価済み  test_metrics.json あり（学習＋評価完了）
  学習完了  完走したが評価は未（done.json、または上限到達/早期終了成立）
  学習中    train_log.csv が伸びている途中
  未着手    ディレクトリなし

使い方（別の Anaconda Prompt から）:
  cd scripts
  python show_progress.py
  python show_progress.py --watch 60      # 60秒ごとに更新（Ctrl+Cで終了）
"""

import argparse
import csv
import json
import os
import time
import unicodedata

OBS_LIST = list(range(100, 1201, 100))


def disp_width(s):
    """全角を2桁として数えた表示幅（表の桁揃え用）。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad(s, n):
    return s + " " * max(0, n - disp_width(s))


def expected_runs(conditions=(1, 2, 3)):
    """run_experiments.py が実行する順に run 名を並べる。
    Step A は各条件ごとに探索（1条件11本）、Step B は 1条件12本。
    conditions を絞れば、条件③のみの再実験など部分実行の進捗も正しく数えられる。"""
    names = []
    for c in conditions:
        names += [(f"search_cond{c}_lr{lr:g}", "A") for lr in (1e-3, 1e-4)]
        names += [(f"search_cond{c}_a{round(0.1 * i, 1)}", "A") for i in range(1, 10)]
    names += [(f"cond{c}_obs{o}", "B") for c in conditions for o in OBS_LIST]
    return names


def read_log(run_dir):
    """train_log.csv から進捗を読む（無ければ None）。"""
    p = os.path.join(run_dir, "train_log.csv")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as fp:
            rows = [r for r in csv.DictReader(fp) if r.get("epoch")]
        eps = [(int(r["epoch"]), float(r["val_miou"])) for r in rows]
    except (OSError, KeyError, ValueError):
        return None
    if not eps:
        return None
    best_epoch, best_miou = max(eps, key=lambda t: t[1])
    return {"last": eps[-1][0], "best_epoch": best_epoch,
            "best_miou": best_miou, "mtime": os.path.getmtime(p)}


def read_json(run_dir, name):
    p = os.path.join(run_dir, name)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as fp:
            return json.load(fp)
    except (OSError, json.JSONDecodeError):
        return None


def ago(ts):
    s = max(0, time.time() - ts)
    if s < 90:
        return f"{int(s)}秒前"
    if s < 5400:
        return f"{int(s / 60)}分前"
    return f"{s / 3600:.1f}時間前"


def state_of(run_dir, log, epochs, early_patience):
    """(状態, 完了か) を返す。完了判定は run_experiments.py と同じ規則。"""
    if read_json(run_dir, "test_metrics.json") is not None:
        return "評価済み", True
    done = read_json(run_dir, "done.json")
    if done and done.get("completed") and not done.get("limit"):
        return "学習完了", True
    if log is None:
        return "未着手", False
    if log["last"] >= epochs or (log["last"] - log["best_epoch"]) >= early_patience:
        return "学習完了", True
    return "学習中", False


def report(runs_dir, epochs, early_patience, conditions=(1, 2, 3)):
    sr = read_json(runs_dir, "search_result.json")
    if sr and "per_condition" in sr:
        parts = []
        for c in sorted(sr["per_condition"], key=int):
            d = sr["per_condition"][c]
            parts.append(f"条件{c}(lr={float(d['best_lr']):g},α={d['best_alpha']})")
        scale = sr.get("offset_lr_scale")
        tag = f"  offset_lr_scale={scale:g}" if scale is not None else ""
        print("Step A 決定済み: " + "  ".join(parts) + tag
              + f"  （{os.path.join(runs_dir, 'search_result.json')}）")

    rows, n_done, newest = [], 0, None
    for name, step in expected_runs(conditions):
        d = os.path.join(runs_dir, name)
        log = read_log(d)
        st, done = state_of(d, log, epochs, early_patience)
        if done:
            n_done += 1
        if st == "未着手":
            continue
        tm = read_json(d, "test_metrics.json")
        rows.append({
            "name": name, "step": step, "state": st,
            "epoch": "" if log is None else f"{log['last']}/{epochs}",
            "best": "" if log is None else f"{log['best_miou']:.4f}@{log['best_epoch']}",
            "test": "" if tm is None else f"{tm.get('mIoU', float('nan')):.4f}",
            "upd": "" if log is None else ago(log["mtime"]),
        })
        # 「実行中」として報告するのは学習中のrunのうち最も新しく更新されたもの
        if log and st == "学習中" and (newest is None or log["mtime"] >= newest[1]["mtime"]):
            newest = (name, log)

    if not rows:
        print(f"[WARN] 進捗が見つかりません（{runs_dir}）。まだ1エポックも"
              f"終わっていないか、--runs-dir が違う可能性があります。")
        return

    w = max(len(r["name"]) for r in rows)
    print("\n" + pad("run", w) + "  step  " + pad("状態", 10)
          + pad("epoch", 10) + pad("best val_mIoU", 15)
          + pad("test mIoU", 11) + "更新")
    print("-" * (w + 54))
    for r in rows:
        print(pad(r["name"], w) + f"  {r['step']:^4}  " + pad(r["state"], 10)
              + pad(r["epoch"], 10) + pad(r["best"], 15)
              + pad(r["test"], 11) + r["upd"])

    total = len(expected_runs(conditions))
    nc = len(conditions)
    print(f"\n完了 {n_done}/{total} 本（Step A {nc*11} + Step B {nc*12}, 条件{list(conditions)}）")
    if newest:
        name, log = newest
        stall = log["last"] - log["best_epoch"]
        print(f"実行中: {name}  epoch {log['last']}/{epochs}  "
              f"best={log['best_miou']:.4f}@{log['best_epoch']}  "
              f"停滞 {stall}/{early_patience}  （最終更新 {ago(log['mtime'])}）")
        if time.time() - log["mtime"] > 1800:
            print("  ※30分以上更新がありません。1エポックにこれ以上かかっているか、"
                  "停止している可能性があります（nvidia-smi でGPU稼働を確認）。")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description="実験の進捗確認（読み取り専用）")
    ap.add_argument("--runs-dir", default=os.path.join(root, "runs"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--early-patience", type=int, default=20)
    ap.add_argument("--conditions", default="1,2,3",
                    help="進捗を数える条件（例 3 で条件③のみの再実験用）")
    ap.add_argument("--watch", type=int, default=0, help="N秒ごとに更新")
    args = ap.parse_args()
    conditions = tuple(int(c) for c in args.conditions.split(","))

    while True:
        if args.watch:
            print("\n" + "=" * 70)
            print(time.strftime("%Y-%m-%d %H:%M:%S"))
        report(args.runs_dir, args.epochs, args.early_patience, conditions)
        if not args.watch:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
