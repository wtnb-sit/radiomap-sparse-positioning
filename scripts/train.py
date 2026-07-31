"""train.py：1回分の学習（単一の 条件×観測点数）

training_eval_spec.md に準拠:
  ・Optimizer=Adam、既存層=base_lr / DCNオフセット層=base_lr×offset_lr_scale（既定0.1）
  ・ReduceLROnPlateau(検証mIoU, mode='max', patience=10, factor=0.5)
  ・早期終了(検証mIoU, patience=20)
  ・ベストモデル選定は検証mIoU（損失ではない）
  ・損失=重み付きCE(sqrt逆頻度) + α·Dice（40×50全域）
  ・シード固定で再現可能

条件マッピング:
  1: MaskedCNNUNet(DCNなし) + 残差
  2: DCNUNet(DCNあり)       + 生RSS(raw)
  3: DCNUNet(DCNあり)       + 残差（提案手法）

使い方（1回分）:
  python train.py --condition 3 --num-obs 100 --lr 1e-3 --alpha 0.5 \
      --seed 42 --out-dir ../runs/cond3_obs100
"""

import argparse
import os
import csv
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import RadioMapDataset, sqrt_inverse_freq_weights
from model import DCNUNet, seg_loss
from model_ablation import MaskedCNNUNet
from metrics import miou_obstacle

# 条件 -> (モデル種別, 入力種別)
COND = {
    1: ("ablation", "residual"),   # マスク付きCNN + 通常Conv + 残差
    2: ("dcn",      "raw"),        # マスク付きCNN + DCN + 生RSS
    3: ("dcn",      "residual"),   # 提案手法
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(kind):
    return DCNUNet() if kind == "dcn" else MaskedCNNUNet()


def make_optimizer(model, base_lr, offset_lr_scale=0.1):
    """既存層=base_lr、DCNオフセット層=base_lr×offset_lr_scale のグループ分け。

    offset_lr_scale の既定は 0.1（従来動作）。オフセットが小さくしか育たず
    位置補正が機能しなかった件（results_analysis.md §3）の検証のため、1.0 等に
    上げて再実験できるようにしている。
    """
    offset = list(model.offset_parameters())
    if offset:
        offset_ids = {id(p) for p in offset}
        base = [p for p in model.parameters() if id(p) not in offset_ids]
        groups = [{"params": base, "lr": base_lr},
                  {"params": offset, "lr": base_lr * offset_lr_scale}]
    else:
        groups = [{"params": model.parameters(), "lr": base_lr}]
    return torch.optim.Adam(groups, lr=base_lr)


@torch.no_grad()
def validate(model, loader, device, weights, alpha):
    model.eval()
    mious, losses = [], []
    for x, m, y in loader:
        x, m, y = x.to(device), m.to(device), y.to(device)
        logits = model(x, m)
        loss, _, _ = seg_loss(logits, y, weights, alpha)
        losses.append(loss.item())
        miou, _ = miou_obstacle(logits.cpu(), y.cpu())
        mious.append(miou)
    return torch.cat(mious).mean().item(), float(np.mean(losses))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description="学習（1回分）")
    ap.add_argument("--condition", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--num-obs", type=int, required=True)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--sched-patience", type=int, default=10)
    ap.add_argument("--early-patience", type=int, default=20)
    ap.add_argument("--offset-lr-scale", type=float, default=0.1,
                    help="DCNオフセット層の学習率倍率（base_lr×これ）。既定0.1＝従来動作")
    ap.add_argument("--processed-root", default=os.path.join(root, "data", "processed"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="動作確認用：train/valのサンプル数を上限で切る（0=無制限）")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kind, input_type = COND[args.condition]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"条件{args.condition} ({kind}, {input_type})  obs{args.num_obs}  "
          f"lr={args.lr} α={args.alpha}  device={device}")

    # データ
    train_ds = RadioMapDataset(args.processed_root, "train", args.num_obs, input_type)
    val_ds = RadioMapDataset(args.processed_root, "val", args.num_obs, input_type)
    if args.limit > 0:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, range(min(args.limit, len(train_ds))))
        val_ds = Subset(val_ds, range(min(args.limit, len(val_ds))))
    g = torch.Generator(); g.manual_seed(args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, generator=g, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    # モデル・損失・最適化
    model = build_model(kind).to(device)
    weights = torch.tensor(sqrt_inverse_freq_weights(args.processed_root, "train"),
                           device=device)
    opt = make_optimizer(model, args.lr, args.offset_lr_scale)
    print(f"オフセット学習率倍率 = {args.offset_lr_scale}（オフセット層lr={args.lr * args.offset_lr_scale:g}）")
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=args.sched_patience)

    log_path = os.path.join(args.out_dir, "train_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as fp:
        csv.writer(fp).writerow(["epoch", "train_loss", "val_miou", "val_loss", "lr"])

    best_miou, best_epoch, since_improve = -1.0, -1, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_losses = []
        for x, m, y in train_dl:
            x, m, y = x.to(device), m.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x, m)
            loss, _, _ = seg_loss(logits, y, weights, args.alpha)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())
        tr_loss = float(np.mean(tr_losses))

        val_miou, val_loss = validate(model, val_dl, device, weights, args.alpha)
        sched.step(val_miou)
        cur_lr = opt.param_groups[0]["lr"]
        with open(log_path, "a", newline="", encoding="utf-8") as fp:
            csv.writer(fp).writerow([epoch, f"{tr_loss:.5f}", f"{val_miou:.5f}",
                                     f"{val_loss:.5f}", f"{cur_lr:.2e}"])

        improved = val_miou > best_miou
        if improved:
            best_miou, best_epoch, since_improve = val_miou, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_miou": val_miou, "condition": args.condition,
                        "num_obs": args.num_obs, "kind": kind,
                        "input_type": input_type, "alpha": args.alpha,
                        "offset_lr_scale": args.offset_lr_scale},
                       os.path.join(args.out_dir, "best.pt"))
        else:
            since_improve += 1
        print(f"[{epoch:3d}] train_loss={tr_loss:.4f} val_mIoU={val_miou:.4f} "
              f"lr={cur_lr:.1e} {'*best' if improved else f'({since_improve}/{args.early_patience})'}")

        if since_improve >= args.early_patience:
            print(f"早期終了（{args.early_patience}エポック改善なし）")
            break

    print(f"完了。ベスト: epoch={best_epoch} val_mIoU={best_miou:.4f} "
          f"-> {os.path.join(args.out_dir, 'best.pt')}")

    # 正常終了マーカー（run_experiments.py の再開判定が「完走したrun」を識別するために使う）。
    # 途中で落ちた場合は書かれないので、未収束の best.pt を完了と誤認しない。
    done = {"completed": True, "condition": args.condition, "num_obs": args.num_obs,
            "kind": kind, "input_type": input_type, "lr": args.lr, "alpha": args.alpha,
            "offset_lr_scale": args.offset_lr_scale,
            "seed": args.seed, "epochs": args.epochs,
            "early_patience": args.early_patience, "last_epoch": epoch,
            "best_epoch": best_epoch, "best_val_miou": best_miou,
            "stopped": "early_stop" if since_improve >= args.early_patience else "max_epochs",
            "limit": args.limit}
    with open(os.path.join(args.out_dir, "done.json"), "w", encoding="utf-8") as fp:
        json.dump(done, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
