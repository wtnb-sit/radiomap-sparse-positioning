"""diagnose_offsets.py：DCNオフセットが学習で動いたかを診断

仮説：DCNのオフセット層はゼロ初期化＋学習率0.1倍のため、学習後もほぼ動かず、
DCNが実質「通常の畳み込み」として振る舞った → 条件③≈条件①。これを検証する。

DCNv2Block.offset_conv は 3*k*k ch を出す：先頭 2*k*k=オフセット(Δy,Δx)、
残り k*k=変調（sigmoid、初期0.5）。オフセットは torchvision DeformConv2d の
規約でピクセル(セル)単位。**実データを順伝播し、各サンプリング点が何セル動くか**を測る。

判断の目安（測位誤差はレイリー平均0.61m≈3セル）:
  平均|オフセット| ≪ 0.1セル → ほぼ不動。DCNは通常畳み込み同然（仮説を支持）
  平均|オフセット| ~ 1〜3セル → 能動的に変形。位置補正が働いている

出力：cond3 の各 obs・各DCNブロック(dcn0/1/2)について
  重みノルム / on-data 平均・p95 オフセット(セル) / 変調平均(初期0.5からのずれ)

使い方（runs と data がある機で・scripts フォルダ）:
  python diagnose_offsets.py                      # cond3 全obs
  python diagnose_offsets.py --obs 100 600 1200   # 一部のobsだけ
  python diagnose_offsets.py --batch 64
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import RadioMapDataset
from model import DCNUNet, DCNv2Block

BLOCKS = ["dcn0", "dcn1", "dcn2"]


def diagnose_one(ckpt_path, processed_root, num_obs, batch, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ck["kind"] != "dcn":
        raise SystemExit(f"[ERROR] {ckpt_path} は DCN モデルではありません（条件③のみ対象）")
    model = DCNUNet()
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    # 各 DCNv2Block.offset_conv の出力を捕捉するフック
    captured = {}
    handles = []
    for name in BLOCKS:
        blk = getattr(model, name)
        k = blk.k

        def mk(nm, kk):
            def hook(mod, inp, out):
                captured[nm] = (out.detach(), kk)
            return hook
        handles.append(blk.offset_conv.register_forward_hook(mk(name, k)))

    # 重みノルム（初期は0）
    with torch.no_grad():
        wnorm = {n: float(getattr(model, n).offset_conv.weight.norm()) for n in BLOCKS}
        bnorm = {n: float(getattr(model, n).offset_conv.bias.norm()) for n in BLOCKS}

    ds = RadioMapDataset(processed_root, "test", num_obs, ck["input_type"])
    dl = DataLoader(ds, batch_size=batch, shuffle=False)
    x, m, _ = next(iter(dl))
    with torch.no_grad():
        model(x.to(device), m.to(device))
    for h in handles:
        h.remove()

    res = {}
    for name in BLOCKS:
        o, k = captured[name]
        ck2 = k * k
        off = o[:, :2 * ck2]                       # (B,2*k*k,H,W)
        B, _, H, W = off.shape
        off = off.view(B, ck2, 2, H, W)            # (Δy,Δx) ペア
        mag = torch.sqrt((off ** 2).sum(dim=2))    # (B,k*k,H,W) 各タップの変位(セル)
        mod = torch.sigmoid(o[:, 2 * ck2:])        # 変調（初期0.5）
        res[name] = {
            "wnorm": wnorm[name], "bnorm": bnorm[name],
            "mean_mag": float(mag.mean()), "p95_mag": float(mag.flatten().quantile(0.95)),
            "max_mag": float(mag.max()),
            "mod_mean": float(mod.mean()), "mod_std": float(mod.std()),
        }
    return res


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description="DCNオフセット診断（条件③）")
    ap.add_argument("--runs-dir", default=os.path.join(root, "runs"))
    ap.add_argument("--processed-root", default=os.path.join(root, "data", "processed"))
    ap.add_argument("--obs", type=int, nargs="*", default=list(range(100, 1201, 100)))
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  対象=条件③ obs{args.obs}  （batch={args.batch}）")
    print("オフセットは torchvision DeformConv2d 規約のセル単位。"
          "測位誤差≈3セルが目安。\n")

    header = (f"{'obs':>5} {'block':>5} | {'wnorm':>7} {'bnorm':>7} | "
              f"{'mean|off|':>9} {'p95|off|':>8} {'max|off|':>8} | "
              f"{'mod_mean':>8} {'mod_std':>7}  (セル)")
    print(header); print("-" * len(header))
    all_mean = []
    for obs in args.obs:
        ck_path = os.path.join(args.runs_dir, f"cond3_obs{obs}", "best.pt")
        if not os.path.exists(ck_path):
            print(f"{obs:>5}  (best.pt なし: {ck_path})")
            continue
        res = diagnose_one(ck_path, args.processed_root, obs, args.batch, device)
        for name in BLOCKS:
            r = res[name]
            print(f"{obs:>5} {name:>5} | {r['wnorm']:>7.4f} {r['bnorm']:>7.4f} | "
                  f"{r['mean_mag']:>9.4f} {r['p95_mag']:>8.4f} {r['max_mag']:>8.3f} | "
                  f"{r['mod_mean']:>8.4f} {r['mod_std']:>7.4f}")
            all_mean.append(r["mean_mag"])
        print()

    if all_mean:
        gm = float(np.mean(all_mean))
        print(f"全体の平均|オフセット| = {gm:.4f} セル")
        if gm < 0.1:
            print("→ ほぼ不動。DCNは通常畳み込み同然に振る舞った可能性が高い"
                  "（③≈① の説明と整合）。")
        elif gm < 1.0:
            print("→ わずかに変形。位置補正としては弱い可能性。")
        else:
            print("→ 能動的に変形（1セル以上）。DCNは動いている。"
                  "効果が出ない原因は別を検討。")


if __name__ == "__main__":
    main()
