"""compare_conditions_paired.py：2条件の per-image ペア統計比較

2条件（例②vs③）は同じテスト画像で評価しているので、**画像ごとに対応づけた
ペア比較**をすれば、画像難易度のばらつきを相殺して小さな差の有意性・大きさを
厳密に判定できる。summary.csv の平均だけでは分からない点を補う。

各 num_obs について:
  ・両条件の per-image mIoU（および障害物IoU）を同一IDで対応づけ
  ・差 Δ = 条件B − 条件A（既定 A=②, B=③）
  ・Wilcoxon 符号順位検定（H0: 差の中央値=0）の p 値
  ・平均Δ・中央値Δ・勝率（B>A の割合）
  ・対応のある効果量 d_z = 平均Δ / 差の標準偏差（大きさの目安）

【注意】観測点数条件は包含関係（obs100⊂obs200…）で互いに独立でないため、
obs をまたいだ「12/12で一貫」を独立試行の符号検定として扱うのは不適切。
本スクリプトは各 obs 内で n=500 のペア検定を行う（画像は独立標本）。

使い方（runs と data がある機で・scripts フォルダ）:
  python compare_conditions_paired.py                       # ②vs③・全obs
  python compare_conditions_paired.py --cond-a 1 --cond-b 3 # ①vs③（DCNの効果）
  python compare_conditions_paired.py --obs 100 600 1200
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import wilcoxon

from dataset import RadioMapDataset
from model import DCNUNet
from model_ablation import MaskedCNNUNet
from metrics import miou_obstacle


def per_image_scores(ckpt_path, processed_root, num_obs, batch, device):
    """(ids, mIoU[N], 障害物IoU[N]) を返す。"""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DCNUNet() if ck["kind"] == "dcn" else MaskedCNNUNet()
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    ds = RadioMapDataset(processed_root, "test", num_obs, ck["input_type"])
    dl = DataLoader(ds, batch_size=batch, shuffle=False)
    mi, oi = [], []
    with torch.no_grad():
        for x, m, y in dl:
            logits = model(x.to(device), m.to(device)).cpu()
            a, b = miou_obstacle(logits, y)
            mi.append(a); oi.append(b)
    return ds.ids, torch.cat(mi).numpy(), torch.cat(oi).numpy()


def paired_stats(a, b):
    """A,B の per-image スコア配列 → 統計量 dict（Δ=B−A）。"""
    d = b - a
    nz = d[d != 0]
    try:
        p = float(wilcoxon(a, b).pvalue) if len(nz) > 0 else 1.0
    except ValueError:
        p = 1.0
    sd = float(d.std(ddof=1))
    return {
        "mean_d": float(d.mean()), "median_d": float(np.median(d)),
        "win": float((d > 0).mean()), "tie": float((d == 0).mean()),
        "p": p, "dz": float(d.mean() / sd) if sd > 1e-12 else 0.0,
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description="2条件の per-image ペア比較")
    ap.add_argument("--runs-dir", default=os.path.join(root, "runs"),
                    help="両条件の runs ディレクトリ（--runs-dir-a/-b 未指定時に使う）")
    ap.add_argument("--runs-dir-a", default=None,
                    help="条件Aの runs ディレクトリ（別フォルダ比較用。既定=--runs-dir）")
    ap.add_argument("--runs-dir-b", default=None,
                    help="条件Bの runs ディレクトリ（例 ../runs_offx1.0）。既定=--runs-dir")
    ap.add_argument("--processed-root", default=os.path.join(root, "data", "processed"))
    ap.add_argument("--cond-a", type=int, default=2, help="基準条件（既定②）")
    ap.add_argument("--cond-b", type=int, default=3, help="比較条件（既定③）")
    ap.add_argument("--obs", type=int, nargs="*", default=list(range(100, 1201, 100)))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--metric", choices=["miou", "obstacle"], default="miou",
                    help="比較指標：miou（既定）or obstacle（障害物IoU）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    A, B = args.cond_a, args.cond_b
    runs_a = args.runs_dir_a or args.runs_dir
    runs_b = args.runs_dir_b or args.runs_dir
    mlabel = "mIoU" if args.metric == "miou" else "障害物IoU"
    print(f"device={device}  Δ = 条件{B} − 条件{A}（per-image {mlabel}, test n=500）")
    if runs_a != runs_b:
        print(f"条件A={A}: {runs_a}\n条件B={B}: {runs_b}")
    print("Wilcoxon符号順位検定（H0:差の中央値=0）。d_z=平均Δ/差のSD（|d_z|:0.2小/0.5中/0.8大）\n")

    hdr = (f"{'obs':>5} | {'平均Δ':>8} {'中央Δ':>8} {'勝率':>6} | "
           f"{'p値':>10} {'有意':>4} | {'d_z':>7} {'効果量':>6}")
    print(hdr); print("-" * len(hdr))
    agg = []
    for obs in args.obs:
        pa = os.path.join(runs_a, f"cond{A}_obs{obs}", "best.pt")
        pb = os.path.join(runs_b, f"cond{B}_obs{obs}", "best.pt")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"{obs:>5} | (best.pt 不足)"); continue
        ida, mia, oia = per_image_scores(pa, args.processed_root, obs, args.batch, device)
        idb, mib, oib = per_image_scores(pb, args.processed_root, obs, args.batch, device)
        if ida != idb:
            raise SystemExit(f"[ERROR] obs{obs}: ID順序が条件間で不一致。ペアにできません")
        a_sc, b_sc = (mia, mib) if args.metric == "miou" else (oia, oib)
        s = paired_stats(a_sc, b_sc)
        sig = "***" if s["p"] < 0.001 else "**" if s["p"] < 0.01 else "*" if s["p"] < 0.05 else "n.s."
        mag = ("大" if abs(s["dz"]) >= 0.8 else "中" if abs(s["dz"]) >= 0.5
               else "小" if abs(s["dz"]) >= 0.2 else "極小")
        print(f"{obs:>5} | {s['mean_d']:>+8.4f} {s['median_d']:>+8.4f} {s['win']:>6.1%} | "
              f"{s['p']:>10.2e} {sig:>4} | {s['dz']:>+7.3f} {mag:>6}")
        agg.append(s)

    if agg:
        mean_dz = float(np.mean([s["dz"] for s in agg]))
        sig_cnt = sum(s["p"] < 0.05 for s in agg)
        print(f"\n有意(p<0.05)だった obs: {sig_cnt}/{len(agg)}")
        print(f"効果量 d_z の平均: {mean_dz:+.3f} "
              f"（{'極小' if abs(mean_dz)<0.2 else '小' if abs(mean_dz)<0.5 else '中以上'}）")
        print("解釈の目安：p が有意でも d_z が極小なら『統計的には差があるが実質的な差は小さい』。")


if __name__ == "__main__":
    main()
