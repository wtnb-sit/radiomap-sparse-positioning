"""run_experiments.py：実験ドライバ（train.py/eval.py を順に呼ぶ軽いドライバ）

training_eval_spec.md §5,§7 の流れ:
  Step A（ハイパラ探索・**各条件ごと**・obs100）
    条件①②③それぞれについて:
      Phase1: α=0.5固定で lr∈{1e-3,1e-4} → 検証mIoU高い方を採用
      Phase2: Phase1のlrで α=0.1..0.9 → 検証mIoU最高のαを採用
    → 各条件が自分の (lr_c, α_c) を持つ（3条件×11本=33本）
  Step B（本評価）
    条件①②③ × 観測点数100..1200（36本）を、**各条件の (lr_c, α_c)** で学習し test 評価

各学習は train.py をサブプロセスで実行（単位実行型＋軽いドライバ）。
検証mIoUは done.json / best.pt から読む。--dry-run で実行計画のみ表示。

【公平性の設計（2026-07-26 変更）】ハイパラは条件ごとに個別探索する。
以前は条件③のみで決めた (lr,α) を全条件に流用していたが、それだと提案手法③に
有利なバイアス（③向けに調整した設定で①②を戦わせる）が入り、「自手法に有利な
調整」との批判を受け得た。各条件を自分の最適点で比較することで DCN の効果(①vs③)・
残差の効果(②vs③)をアーキテクチャ/入力の違いに純粋に帰属させる。

47本は十数〜数十時間かかるため、途中で落ちてもやり直しにならないよう
**再開（レジューム）** に既定で対応する（training_eval_spec.md §9.8）:
  ・完了済みの run はスキップ（Step B は test_metrics.json、学習は完了判定で判断）
  ・Step A の決定（lr,α）は runs/search_result.json に保存し、再実行時は再利用
  ・--continue-on-error で1本の失敗で全体を止めず、最後に失敗一覧を報告

学習の完了判定は done.json（train.py が正常終了時のみ書く）を正とし、
それが無い旧runは train_log.csv から判定（エポック上限到達 or 早期終了成立）。
→ 途中で落ちた未収束の best.pt を「完了」と誤認して評価してしまうのを防ぐ。

使い方:
  python run_experiments.py                 # 探索→本評価まで通し（完了分はスキップ）
  python run_experiments.py --dry-run       # 残り作業の確認（何も学習しない）
  python run_experiments.py --skip-search --lr 1e-3 --alpha 0.5   # 探索を飛ばし全条件に同値
  python run_experiments.py --continue-on-error   # 落ちても続行し最後に報告
  python run_experiments.py --no-resume     # 完了分も無視して全部やり直す
"""

import argparse
import csv
import json
import os
import sys
import subprocess
import torch

OBS_LIST = list(range(100, 1201, 100))
SEARCH_RESULT = "search_result.json"


# ---------------------------------------------------------------- 完了判定

def _read_done(out_dir):
    """train.py の正常終了マーカー done.json を読む（無効・不在なら None）。"""
    p = os.path.join(out_dir, "done.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as fp:
            d = json.load(fp)
    except (json.JSONDecodeError, OSError):
        return None
    if not d.get("completed"):
        return None
    if d.get("limit"):        # --limit 付きの動作確認runは本番の完了扱いにしない
        return None
    return d


def _log_says_completed(out_dir, epochs, early_patience):
    """done.json が無い旧runの完了判定：train_log.csv から
    「エポック上限到達」または「早期終了成立」を確認する。"""
    p = os.path.join(out_dir, "train_log.csv")
    if not os.path.exists(p):
        return False
    try:
        with open(p, encoding="utf-8") as fp:
            rows = [r for r in csv.DictReader(fp) if r.get("epoch")]
    except OSError:
        return False
    if not rows:
        return False
    try:
        last = int(rows[-1]["epoch"])
        best_epoch = int(max(rows, key=lambda r: float(r["val_miou"]))["epoch"])
    except (KeyError, ValueError):
        return False
    return last >= epochs or (last - best_epoch) >= early_patience


def train_completed(out_dir, epochs, early_patience):
    """学習が完走済みか。途中で落ちた best.pt を完了と誤認しないための判定。"""
    if _read_done(out_dir) is not None:
        return True
    return _log_says_completed(out_dir, epochs, early_patience)


def _val_miou(out_dir):
    d = _read_done(out_dir)
    if d is not None and "best_val_miou" in d:
        return float(d["best_val_miou"])
    ck = torch.load(os.path.join(out_dir, "best.pt"),
                    map_location="cpu", weights_only=False)
    return float(ck["val_miou"])


# ---------------------------------------------------------------- 実行

def _run(cmd, ctx, label):
    """サブプロセス実行。--continue-on-error なら失敗を記録して False を返す。"""
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        if not ctx["continue_on_error"]:
            raise
        print(f"[FAIL] {label}: 終了コード {e.returncode} → 続行します")
        ctx["failures"].append(label)
        return False


def run_train(cond, obs, lr, alpha, out_dir, ctx):
    """学習1本。戻り値 (ok, val_miou)。完了済みならスキップして mIoU だけ返す。"""
    label = f"train cond{cond} obs{obs} lr={lr:g} α={alpha}"
    if ctx["resume"] and train_completed(out_dir, ctx["epochs"], ctx["early_patience"]):
        m = None if ctx["dry"] else _val_miou(out_dir)
        print(f"[skip:train] 完了済み {out_dir}"
              + ("" if m is None else f"  val_mIoU={m:.4f}"))
        ctx["skipped"] += 1
        return True, m

    cmd = [ctx["python"], os.path.join(ctx["script_dir"], "train.py"),
           "--condition", str(cond), "--num-obs", str(obs),
           "--lr", str(lr), "--alpha", str(alpha),
           "--out-dir", out_dir] + ctx["common"]
    print("[train]", " ".join(cmd))
    if ctx["dry"]:
        ctx["planned"] += 1
        return True, None
    if not _run(cmd, ctx, label):
        return False, None
    ctx["executed"] += 1
    return True, _val_miou(out_dir)


def run_eval(ckpt, obs, out_json, ctx):
    cmd = [ctx["python"], os.path.join(ctx["script_dir"], "eval.py"),
           "--ckpt", ckpt, "--num-obs", str(obs), "--out", out_json]
    print("[eval] ", " ".join(cmd))
    if ctx["dry"]:
        return True
    return _run(cmd, ctx, f"eval {ckpt}")


def _pick_best(scores, what):
    ok = {k: v for k, v in scores.items() if v is not None}
    if not ok:
        raise SystemExit(f"[ERROR] {what} の探索が全滅しました。ログを確認してください。")
    if len(ok) < len(scores):
        print(f"[WARN] {what}: 失敗した候補を除外して選択します（{len(ok)}/{len(scores)}）")
    return max(ok, key=ok.get)


# ---------------------------------------------------------------- main

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description="実験ドライバ")
    ap.add_argument("--runs-dir", default=os.path.join(root, "runs"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--search-obs", type=int, default=100)
    ap.add_argument("--conditions", default="1,2,3")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--early-patience", type=int, default=20,
                    help="train.py に渡す早期終了patience（完了判定にも使う）")
    ap.add_argument("--offset-lr-scale", type=float, default=0.1,
                    help="DCNオフセット層の学習率倍率（train.pyへ渡す）。既定0.1＝従来")
    ap.add_argument("--skip-search", action="store_true", help="探索を飛ばし --lr --alpha を使う")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-resume", action="store_true",
                    help="完了済みrunもスキップせず全てやり直す")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="1本失敗しても中断せず続行し、最後に失敗一覧を報告")
    args = ap.parse_args()

    os.makedirs(args.runs_dir, exist_ok=True)
    conditions = [int(c) for c in args.conditions.split(",")]
    ctx = {
        "python": args.python, "script_dir": here, "dry": args.dry_run,
        "resume": not args.no_resume, "continue_on_error": args.continue_on_error,
        "epochs": args.epochs, "early_patience": args.early_patience,
        "common": ["--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                   "--seed", str(args.seed), "--num-workers", str(args.num_workers),
                   "--early-patience", str(args.early_patience),
                   "--offset-lr-scale", str(args.offset_lr_scale)],
        "failures": [], "executed": 0, "skipped": 0, "planned": 0,
    }
    if ctx["resume"]:
        print("再開モード：完了済みのrunはスキップします（--no-resume で無効化）")

    # ---- Step A：ハイパラ探索（各条件ごと・search-obs） ----
    # hp[cond] = (lr_c, alpha_c)。各条件を自分の最適点で戦わせる（公平性・上記docstring）。
    search_path = os.path.join(args.runs_dir, SEARCH_RESULT)
    hp = {}

    def _search_condition(cond):
        """条件 cond について Phase1(lr)→Phase2(α) を探索し (best_lr, best_alpha,
        lr_scores, a_scores) を返す。"""
        print(f"--- Step A 条件{cond} Phase1: 学習率の決定"
              f"（α=0.5, obs{args.search_obs}） ---")
        lr_scores = {}
        for lr in (1e-3, 1e-4):
            od = os.path.join(args.runs_dir, f"search_cond{cond}_lr{lr:g}")
            lr_scores[lr] = run_train(cond, args.search_obs, lr, 0.5, od, ctx)[1]
        best_lr = 1e-3 if args.dry_run else _pick_best(lr_scores, f"条件{cond}の学習率")
        print(f"  → 条件{cond} 採用学習率: {best_lr}  ({lr_scores})")

        print(f"--- Step A 条件{cond} Phase2: αの探索"
              f"（0.1..0.9, obs{args.search_obs}） ---")
        a_scores = {}
        for a in [round(0.1 * i, 1) for i in range(1, 10)]:
            od = os.path.join(args.runs_dir, f"search_cond{cond}_a{a}")
            a_scores[a] = run_train(cond, args.search_obs, best_lr, a, od, ctx)[1]
        best_alpha = 0.5 if args.dry_run else _pick_best(a_scores, f"条件{cond}のα")
        print(f"  → 条件{cond} 採用α: {best_alpha}  ({a_scores})")
        return best_lr, best_alpha, lr_scores, a_scores

    # 再利用の可否：search_result.json が要求条件をすべて含むときのみ再利用
    reuse = None
    if ctx["resume"] and os.path.exists(search_path):
        with open(search_path, encoding="utf-8") as fp:
            reuse = json.load(fp)
        pc = reuse.get("per_condition", {})
        if not all(str(c) in pc for c in conditions):
            reuse = None  # 一部条件が未探索 → 再探索（完了済みの個別runはスキップされる）

    if args.skip_search:
        assert args.lr and args.alpha, "--skip-search 時は --lr と --alpha が必要"
        for c in conditions:
            hp[c] = (args.lr, args.alpha)
        print(f"探索スキップ: 全条件に lr={args.lr}, α={args.alpha} を適用")
    elif reuse is not None:
        pc = reuse["per_condition"]
        for c in conditions:
            hp[c] = (float(pc[str(c)]["best_lr"]), float(pc[str(c)]["best_alpha"]))
        print(f"[skip:StepA] {search_path} を再利用: "
              + ", ".join(f"条件{c}(lr={hp[c][0]:g},α={hp[c][1]})" for c in conditions))
    else:
        print(f"=== Step A: ハイパラ探索（各条件ごと, obs{args.search_obs}） ===")
        per_cond = {}
        for c in conditions:
            best_lr, best_alpha, lr_scores, a_scores = _search_condition(c)
            hp[c] = (best_lr, best_alpha)
            per_cond[str(c)] = {
                "best_lr": best_lr, "best_alpha": best_alpha,
                "lr_scores": {f"{k:g}": v for k, v in lr_scores.items()},
                "alpha_scores": {str(k): v for k, v in a_scores.items()}}
        if not args.dry_run:
            with open(search_path, "w", encoding="utf-8") as fp:
                json.dump({"per_condition": per_cond,
                           "search_obs": args.search_obs, "seed": args.seed,
                           "offset_lr_scale": args.offset_lr_scale},
                          fp, ensure_ascii=False, indent=2)
            print(f"[SAVED] {search_path}")

    # ---- Step B：本評価（条件×観測点数 = 個別学習36本・各条件の (lr_c,α_c)） ----
    print(f"=== Step B: 本評価 conditions={conditions} × obs{OBS_LIST} ===")
    for cond in conditions:
        lr_c, alpha_c = hp[cond]
        print(f"--- 条件{cond}（lr={lr_c:g}, α={alpha_c}） ---")
        for obs in OBS_LIST:
            od = os.path.join(args.runs_dir, f"cond{cond}_obs{obs}")
            out_json = os.path.join(od, "test_metrics.json")
            if ctx["resume"] and os.path.exists(out_json):
                print(f"[skip] 評価済み cond{cond}_obs{obs}")
                ctx["skipped"] += 1
                continue
            ok, _ = run_train(cond, obs, lr_c, alpha_c, od, ctx)
            if not ok:
                print(f"[skip:eval] 学習が失敗したため cond{cond}_obs{obs} の評価をスキップ")
                continue
            run_eval(os.path.join(od, "best.pt"), obs, out_json, ctx)

    # ---- サマリ ----
    if args.dry_run:
        print(f"\n[dry-run] 実行予定 {ctx['planned']}本 / スキップ（完了済み） {ctx['skipped']}本")
        return
    print(f"\n完了。実行 {ctx['executed']}本 / スキップ {ctx['skipped']}本 "
          f"/ 失敗 {len(ctx['failures'])}本")
    if ctx["failures"]:
        print("失敗した処理:")
        for f in ctx["failures"]:
            print(f"  - {f}")
        print("再実行すれば完了分はスキップされ、失敗分だけやり直せます。")
        sys.exit(1)
    print("集約は aggregate_results.py で行う。")


if __name__ == "__main__":
    main()
