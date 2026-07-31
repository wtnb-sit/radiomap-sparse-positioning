"""eval.py：1モデルの評価（testセット）

training_eval_spec.md §4 の指標を計算する:
  ・mIoU（主指標）＋ 障害物IoU
  ・F1（障害物Dice）
  ・HD95（双方向境界距離の95%点、予測空はペナルティ√(40²+50²)≈64）
  ・Boundary-F1（τ=1,3,5）
per-image で計算し平均（macro）。結果を JSON 保存。

補助機能（§4.4）：--hd95-dist で検証セットのHD95距離分布ヒストグラムを描く。

使い方:
  python eval.py --ckpt ../runs/cond3_obs100/best.pt --num-obs 100 \
      --out ../runs/cond3_obs100/test_metrics.json
  python eval.py --ckpt ... --num-obs 100 --hd95-dist   # 距離分布(val)
"""

import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import RadioMapDataset
from model import DCNUNet
from model_ablation import MaskedCNNUNet
from metrics import (miou_obstacle, obstacle_f1, hd95, boundary_f1,
                     hd95_pooled_distances, DIAG_PENALTY)

TAUS = [1, 3, 5]


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DCNUNet() if ck["kind"] == "dcn" else MaskedCNNUNet()
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, ck


@torch.no_grad()
def evaluate(model, loader, device):
    mious, oious, f1s, hds = [], [], [], []
    bf1 = {t: [] for t in TAUS}
    for x, m, y in loader:
        x, m = x.to(device), m.to(device)
        logits = model(x, m)
        miou, oiou = miou_obstacle(logits.cpu(), y)
        mious.append(miou); oious.append(oiou)
        pred = logits.argmax(dim=1).cpu().numpy()       # (B,H,W)
        gt = y.numpy()
        for b in range(pred.shape[0]):
            pb, gb = (pred[b] == 1), (gt[b] == 1)
            f1s.append(obstacle_f1(pb, gb))
            hds.append(hd95(pb, gb))
            for t in TAUS:
                bf1[t].append(boundary_f1(pb, gb, t))
    res = {
        "mIoU": float(torch.cat(mious).mean()),
        "obstacle_IoU": float(torch.cat(oious).mean()),
        "F1": float(np.mean(f1s)),
        "HD95": float(np.mean(hds)),
        "mIoU_std": float(torch.cat(mious).std()),
    }
    for t in TAUS:
        res[f"BoundaryF1@{t}"] = float(np.mean(bf1[t]))
    return res


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description="モデル評価（test）")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num-obs", type=int, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--processed-root", default=os.path.join(root, "data", "processed"))
    ap.add_argument("--out", default=None, help="結果JSONの保存先")
    ap.add_argument("--limit", type=int, default=0, help="動作確認用の上限")
    ap.add_argument("--hd95-dist", action="store_true",
                    help="検証セットでHD95距離分布ヒストグラムを描く")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(args.ckpt, device)
    input_type = ck["input_type"]
    print(f"ckpt: 条件{ck['condition']} ({ck['kind']},{input_type}) "
          f"obs{ck['num_obs']} val_mIoU={ck['val_miou']:.4f}  device={device}")

    if args.hd95_dist:
        ds = RadioMapDataset(args.processed_root, "val", args.num_obs, input_type)
        if args.limit > 0:
            ds = Subset(ds, range(min(args.limit, len(ds))))
        dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers)
        dists = []
        with torch.no_grad():
            for x, m, y in dl:
                pred = model(x.to(device), m.to(device)).argmax(dim=1).cpu().numpy()
                gt = y.numpy()
                for b in range(pred.shape[0]):
                    d = hd95_pooled_distances(pred[b] == 1, gt[b] == 1)
                    if d is not None:
                        dists.append(d)
        dists = np.concatenate(dists) if dists else np.array([])
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axth = plt.subplots(figsize=(9, 5))
        axth.hist(dists, bins=50, color="steelblue", edgecolor="white")
        for q, col in [(95, "red"), (90, "orange"), (99, "green")]:
            v = np.percentile(dists, q)
            axth.axvline(v, color=col, ls="--", label=f"{q}%={v:.2f}")
        axth.set_title(f"HD95 boundary-distance distribution (val obs{args.num_obs})")
        axth.set_xlabel("distance [cells]"); axth.set_ylabel("count"); axth.legend()
        out_png = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                               f"hd95_dist_obs{args.num_obs}.png")
        fig.tight_layout(); fig.savefig(out_png, dpi=110)
        print(f"[SAVED] {out_png}  （95%={np.percentile(dists,95):.2f} セル）")
        return

    ds = RadioMapDataset(args.processed_root, args.split, args.num_obs, input_type)
    if args.limit > 0:
        ds = Subset(ds, range(min(args.limit, len(ds))))
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"{args.split}={len(ds)} で評価中...")
    res = evaluate(model, dl, device)
    res.update({"condition": ck["condition"], "num_obs": ck["num_obs"],
                "split": args.split, "n": len(ds)})
    print("=== 結果 ===")
    for k, v in res.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                                   "test_metrics.json")
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False, indent=2)
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
