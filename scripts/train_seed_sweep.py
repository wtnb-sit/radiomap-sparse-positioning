"""train_seed_sweep.py：測位誤差あり/なし × 学習シード3値 で条件①を学習

目的：誤差バジェットの shape 費目が (A)測位誤差由来のぼかし＝不可避 か
(B/C)それ以外＝一部削減可能 かを切り分ける。測位誤差なし(clean)入力で学習して
shape が崩壊するかを見る。同時に学習シードを3値振り、条件間差が
「学習の再現性の揺れ」に埋もれていないかを検証する。

【変える変数は2つだけ】input_mode ∈ {poserr, clean} と seed。
  ハイパラは条件①の既存確定値（runs/search_result.json の per_condition["1"]）を固定使用。
  clean 用の再探索は禁止（シードの揺れの測定が壊れるため）。
  データ生成シード（欠損・位置誤差の実現値）は再生成しない。既存の決定論的生成物を使う。
  モデル構造・k=9/N=4・損失・早期終了・スケジューラは一切変更しない。

既存 train.py は変更せず、そこから set_seed/build_model/make_optimizer/validate を
import して学習ループを同一構成で回す（--condition では clean 入力を選べないため別経路）。
出力は runs_seedsweep/ 配下に完全分離（既存 runs/ runs_offx*/ は読み取りのみ）。

使い方:
  python train_seed_sweep.py --num-obs 600 --input-mode clean --seed 0
  python train_seed_sweep.py --sweep            # 12本（2 input_mode × obs{100,600} × seed{0,1,2}）
  python train_seed_sweep.py --sweep --epochs 2 --limit 32   # 動作確認
"""

import argparse
import csv
import hashlib
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import RadioMapDataset, sqrt_inverse_freq_weights
from model import seg_loss
from metrics import miou_obstacle
# 既存 train.py の実装をそのまま使う（学習挙動を一致させる）
from train import set_seed, build_model, make_optimizer, validate

INPUT_TYPE = {"poserr": "residual", "clean": "clean_residual"}


def weight_hash(model):
    """学習前重みのハッシュ（3シードで初期重みが異なることの確認用）。"""
    h = hashlib.md5()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:12]


def load_cond1_hp(runs_dir):
    """条件①の確定ハイパラ (lr, alpha) を既存 search_result.json から読む。"""
    p = os.path.join(runs_dir, "search_result.json")
    with open(p, encoding="utf-8") as fp:
        d = json.load(fp)
    c = d["per_condition"]["1"]
    return float(c["best_lr"]), float(c["best_alpha"])


def train_one(num_obs, input_mode, seed, lr, alpha, out_dir, args):
    """1本の学習＋test評価。既存 train.py と同一の最適化・停止条件。"""
    if os.path.exists(os.path.join(out_dir, "done.json")):
        try:
            with open(os.path.join(out_dir, "done.json"), encoding="utf-8") as fp:
                if json.load(fp).get("completed"):
                    print(f"[skip] 完了済み {out_dir}")
                    return
        except (OSError, json.JSONDecodeError):
            pass

    os.makedirs(out_dir, exist_ok=True)
    input_type = INPUT_TYPE[input_mode]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)                        # torch/numpy/random/cuda すべて
    kind = "ablation"                     # 条件①=DCNなし
    print(f"\n=== 条件1 obs{num_obs} input={input_mode}({input_type}) seed={seed} "
          f"lr={lr:g} α={alpha} device={device} ===")

    train_ds = RadioMapDataset(args.processed_root, "train", num_obs, input_type)
    val_ds = RadioMapDataset(args.processed_root, "val", num_obs, input_type)
    if args.limit > 0:
        train_ds = Subset(train_ds, range(min(args.limit, len(train_ds))))
        val_ds = Subset(val_ds, range(min(args.limit, len(val_ds))))
    g = torch.Generator(); g.manual_seed(seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, generator=g, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    model = build_model(kind).to(device)
    print(f"初期重みhash={weight_hash(model)}（seedごとに異なるはず）")
    weights = torch.tensor(sqrt_inverse_freq_weights(args.processed_root, "train"),
                           device=device)
    opt = make_optimizer(model, lr)       # 条件①はオフセット層なし
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=args.sched_patience)

    log_path = os.path.join(out_dir, "train_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as fp:
        csv.writer(fp).writerow(["epoch", "train_loss", "val_miou", "val_loss", "lr"])

    best_miou, best_epoch, since = -1.0, -1, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr = []
        for x, m, y in train_dl:
            x, m, y = x.to(device), m.to(device), y.to(device)
            opt.zero_grad()
            loss, _, _ = seg_loss(model(x, m), y, weights, alpha)
            loss.backward(); opt.step(); tr.append(loss.item())
        tr_loss = float(np.mean(tr))
        val_miou, val_loss = validate(model, val_dl, device, weights, alpha)
        sched.step(val_miou)
        cur_lr = opt.param_groups[0]["lr"]
        with open(log_path, "a", newline="", encoding="utf-8") as fp:
            csv.writer(fp).writerow([epoch, f"{tr_loss:.5f}", f"{val_miou:.5f}",
                                     f"{val_loss:.5f}", f"{cur_lr:.2e}"])
        improved = val_miou > best_miou
        if improved:
            best_miou, best_epoch, since = val_miou, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_miou": val_miou, "condition": 1, "num_obs": num_obs,
                        "kind": kind, "input_type": input_type, "alpha": alpha,
                        "seed": seed, "input_mode": input_mode},
                       os.path.join(out_dir, "best.pt"))
        else:
            since += 1
        print(f"[{epoch:3d}] train_loss={tr_loss:.4f} val_mIoU={val_miou:.4f} "
              f"lr={cur_lr:.1e} {'*best' if improved else f'({since}/{args.early_patience})'}")
        if since >= args.early_patience:
            print(f"早期終了（{args.early_patience}エポック改善なし）")
            break

    print(f"完了。ベスト epoch={best_epoch} val_mIoU={best_miou:.4f}")
    with open(os.path.join(out_dir, "done.json"), "w", encoding="utf-8") as fp:
        json.dump({"completed": True, "condition": 1, "num_obs": num_obs,
                   "input_mode": input_mode, "input_type": input_type,
                   "data_dir": RadioMapDataset._CONFIG[input_type][0],
                   "lr": lr, "alpha": alpha, "seed": seed, "epochs": args.epochs,
                   "early_patience": args.early_patience, "last_epoch": epoch,
                   "best_epoch": best_epoch, "best_val_miou": best_miou,
                   "limit": args.limit}, fp, ensure_ascii=False, indent=2)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description="測位誤差有無×シード3値のスイープ（条件①）")
    ap.add_argument("--runs-dir", default=os.path.join(root, "runs_seedsweep"))
    ap.add_argument("--hp-runs-dir", default=os.path.join(root, "runs"),
                    help="条件①の確定ハイパラを読む既存runs（読み取りのみ）")
    ap.add_argument("--processed-root", default=os.path.join(root, "data", "processed"))
    ap.add_argument("--condition", type=int, default=1, choices=[1])
    ap.add_argument("--num-obs", type=int, default=600)
    ap.add_argument("--input-mode", choices=["poserr", "clean"], default="clean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true", help="12本を通す")
    ap.add_argument("--obs-list", type=int, nargs="*", default=[100, 600])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--sched-patience", type=int, default=10)
    ap.add_argument("--early-patience", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    lr, alpha = load_cond1_hp(args.hp_runs_dir)
    print(f"[サニティ] 条件①の確定ハイパラを既存 search_result.json から読込: "
          f"lr={lr:g}, α={alpha}（再探索していない）")
    os.makedirs(args.runs_dir, exist_ok=True)

    jobs = []
    if args.sweep:
        for mode in ("poserr", "clean"):
            for obs in args.obs_list:
                for s in args.seeds:
                    jobs.append((obs, mode, s))
    else:
        jobs.append((args.num_obs, args.input_mode, args.seed))
    print(f"実行予定 {len(jobs)}本")

    for obs, mode, s in jobs:
        od = os.path.join(args.runs_dir, f"cond1_{mode}_obs{obs}_seed{s}")
        train_one(obs, mode, s, lr, alpha, od, args)


if __name__ == "__main__":
    main()
