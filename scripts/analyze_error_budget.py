"""analyze_error_budget.py：障害物予測の失点を4費目に分解する

mIoUの失点（障害物クラスの FN+FP 画素）が (a)未検出 missed (b)誤検出 fp
(c)形状 shape (d)重心 centroid のどれからどれだけ出ているかを内訳化する。
損失設計の投資先をデータで決めるための分解。新規学習なし・既存 best.pt で推論のみ。

【マッチング規約は前回診断を完全流用】
  scripts/diagnose_radial_offset.py の load_model / components(ndimage.label,4連結) /
  match_components(IoU降順の貪欲・IoU=0のGTはmissed・未マッチ予測はfp) / centroid。
  2値化は argmax（障害物=class1）、test分割・GTは dataset.RadioMapDataset。

【4費目（排他・網羅、合計≈total_error）】
  total_error = Σ(FN画素) + Σ(FP画素)（障害物クラス、全test画像）
  1. missed : マッチしないGT成分の画素（全てFN）
  2. fp     : マッチしない予測成分の画素（全てFP）
  3. shape  : マッチペアで、予測成分を「予測重心→GT重心」へ整数セル平行移動して
              重ねた後に残る不一致画素(FN+FP)。＝重心を合わせても消えない形状差。
  4. centroid: マッチペアの移動前不一致 − shape。＝重心をずらしたことによる不一致。
  平行移動は round の整数セル。境界外にはみ出す画素はクリップ（捨てる）。
  4費目合計と独立計算 total_error の残差%をサニティ印字。

mIoU/障害物IoU は eval.py 出力の results/summary.csv を隣に並べる（定義を作り直さない）。

使い方（runs と data がある機で・scripts フォルダ）:
  python analyze_error_budget.py                       # 条件①②③・obs{100,600,1200}
  python analyze_error_budget.py --conditions 1 --obs 600
"""

import argparse
import csv
import os
import numpy as np
import torch

from dataset import RadioMapDataset
from diagnose_radial_offset import load_model, components, match_components, centroid


def translate_clip(mask, dr, dc):
    """mask(bool,H,W) を (dr,dc) セル平行移動。境界外はクリップ（捨てる）。"""
    H, W = mask.shape
    out = np.zeros_like(mask)
    src_r0, src_r1 = max(0, -dr), min(H, H - dr)
    src_c0, src_c1 = max(0, -dc), min(W, W - dc)
    if src_r0 >= src_r1 or src_c0 >= src_c1:
        return out
    dst_r0, dst_r1 = src_r0 + dr, src_r1 + dr
    dst_c0, dst_c1 = src_c0 + dc, src_c1 + dc
    out[dst_r0:dst_r1, dst_c0:dst_c1] = mask[src_r0:src_r1, src_c0:src_c1]
    return out


def load_eval_metrics(summary_path):
    """results/summary.csv から (condition,num_obs)->(mIoU,障害物IoU)。"""
    d = {}
    if not os.path.exists(summary_path):
        return d
    for r in csv.DictReader(open(summary_path, encoding="utf-8")):
        d[(int(r["condition"]), int(r["num_obs"]))] = (
            float(r["mIoU"]), float(r["obstacle_IoU"]))
    return d


def load_ref_counts(radial_path):
    """results/radial_offset_summary.csv から照合用のマッチ件数。"""
    d = {}
    if not os.path.exists(radial_path):
        return d
    for r in csv.DictReader(open(radial_path, encoding="utf-8")):
        d[(int(r["condition"]), int(r["num_obs"]))] = (
            int(r["n_matched"]), int(r["n_missed"]), int(r["n_fp"]))
    return d


def analyze_one(model, ds, device, verbose_n=0):
    """1モデルの全test画像を4費目に分解。集計dictと raw行、件数を返す。"""
    acc = {"missed": 0, "fp": 0, "shape": 0, "centroid": 0, "total": 0}
    raw = []
    n_matched = n_missed = n_fp = 0
    shown = 0
    with torch.no_grad():
        for idx in range(len(ds)):
            sid = ds.ids[idx]
            x, m, y = ds[idx]
            pred = (model(x[None].to(device), m[None].to(device))
                    .argmax(dim=1)[0].cpu().numpy() == 1)
            gt = (y.numpy() == 1)
            # 独立の総失点（全画像XOR = FN+FP）
            acc["total"] += int(np.logical_xor(pred, gt).sum())

            pl, npred = components(pred)
            gl, ngt = components(gt)
            matches, missed, fp = match_components(pl, npred, gl, ngt)
            n_matched += len(matches); n_missed += missed; n_fp += fp

            matched_g = {g for g, p, v in matches}
            matched_p = {p for g, p, v in matches}
            im = {"missed": 0, "fp": 0, "shape": 0, "centroid": 0}
            # 未検出GT成分（全画素FN）
            for g in range(1, ngt + 1):
                if g not in matched_g:
                    px = int((gl == g).sum()); acc["missed"] += px; im["missed"] += px
                    raw.append((sid, "missed", px))
            # 誤検出予測成分（全画素FP）
            for p in range(1, npred + 1):
                if p not in matched_p:
                    px = int((pl == p).sum()); acc["fp"] += px; im["fp"] += px
                    raw.append((sid, "fp", px))
            # マッチペア：shape と centroid
            for g, p, v in matches:
                gm = (gl == g); pm = (pl == p)
                before = int(np.logical_xor(pm, gm).sum())
                c_g = centroid(gl, g); c_p = centroid(pl, p)
                dr, dc = int(round(c_g[0] - c_p[0])), int(round(c_g[1] - c_p[1]))
                pm_s = translate_clip(pm, dr, dc)
                shape = int(np.logical_xor(pm_s, gm).sum())
                cen = before - shape
                acc["shape"] += shape; acc["centroid"] += cen
                im["shape"] += shape; im["centroid"] += cen
                raw.append((sid, "shape", shape))
                raw.append((sid, "centroid", cen))
            if shown < verbose_n:
                print(f"    [sanity id={sid}] missed={im['missed']} fp={im['fp']} "
                      f"shape={im['shape']} centroid={im['centroid']} "
                      f"(4費目和={sum(im.values())})")
                shown += 1
    return acc, raw, n_matched, n_missed, n_fp


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description="障害物予測の誤差バジェット4費目分解")
    ap.add_argument("--runs-dir", default=os.path.join(root, "runs"))
    ap.add_argument("--processed-root", default=os.path.join(root, "data", "processed"))
    ap.add_argument("--out-dir", default=os.path.join(root, "results"))
    ap.add_argument("--conditions", default="1,2,3")
    ap.add_argument("--obs", type=int, nargs="*", default=[100, 600, 1200])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    conditions = [int(c) for c in args.conditions.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    evalm = load_eval_metrics(os.path.join(args.out_dir, "summary.csv"))
    refc = load_ref_counts(os.path.join(args.out_dir, "radial_offset_summary.csv"))
    print(f"device={device}  conditions={conditions} obs={args.obs}")
    print("4費目: missed(未検出FN) / fp(誤検出FP) / shape(形状) / centroid(重心)\n")

    summary_rows, raw_rows = [], []
    for cond in conditions:
        for obs in args.obs:
            ckpt = os.path.join(args.runs_dir, f"cond{cond}_obs{obs}", "best.pt")
            if not os.path.exists(ckpt):
                print(f"[skip] {ckpt} なし"); continue
            model, ck = load_model(ckpt, device)
            ds = RadioMapDataset(args.processed_root, "test", obs, ck["input_type"])
            vb = 3 if (cond == conditions[0] and obs == args.obs[0]) else 0
            if vb:
                print(f"[sanity] 条件{cond} obs{obs} 先頭3画像の費目内訳:")
            acc, raw, nm, nmiss, nfp = analyze_one(model, ds, device, verbose_n=vb)

            budget = acc["missed"] + acc["fp"] + acc["shape"] + acc["centroid"]
            total = acc["total"]
            resid = (budget - total) / total * 100 if total > 0 else 0.0
            # マッチ件数の前回照合
            ref = refc.get((cond, obs))
            ok = "一致" if ref and (nm, nmiss, nfp) == ref else (
                f"不一致(前回{ref})" if ref else "前回データなし")
            miou, oiou = evalm.get((cond, obs), (float("nan"), float("nan")))
            n_img = len(ds)

            def pct(v): return v / total * 100 if total > 0 else 0.0
            print(f"条件{cond} obs{obs}: total_error={total}px | "
                  f"missed={acc['missed']}({pct(acc['missed']):.1f}%) "
                  f"fp={acc['fp']}({pct(acc['fp']):.1f}%) "
                  f"shape={acc['shape']}({pct(acc['shape']):.1f}%) "
                  f"centroid={acc['centroid']}({pct(acc['centroid']):.1f}%) | "
                  f"残差={resid:+.2f}% | マッチ件数照合:{ok} | mIoU={miou:.4f}")
            summary_rows.append({
                "condition": cond, "num_obs": obs, "miou": round(miou, 4),
                "obstacle_iou": round(oiou, 4), "total_error_px": total,
                "missed_px": acc["missed"], "fp_px": acc["fp"],
                "shape_px": acc["shape"], "centroid_px": acc["centroid"],
                "missed_pct": round(pct(acc["missed"]), 2),
                "fp_pct": round(pct(acc["fp"]), 2),
                "shape_pct": round(pct(acc["shape"]), 2),
                "centroid_pct": round(pct(acc["centroid"]), 2),
                "missed_px_per_img": round(acc["missed"] / n_img, 2),
                "fp_px_per_img": round(acc["fp"] / n_img, 2),
                "shape_px_per_img": round(acc["shape"] / n_img, 2),
                "centroid_px_per_img": round(acc["centroid"] / n_img, 2),
                "n_matched": nm, "n_missed": nmiss, "n_fp": nfp,
                "reconstruction_residual_pct": round(resid, 2)})
            for sid, cat, px in raw:
                raw_rows.append({"id": sid, "condition": cond, "num_obs": obs,
                                 "category": cat, "pixels": px})

    # ---- CSV ----
    cols = ["condition", "num_obs", "miou", "obstacle_iou", "total_error_px",
            "missed_px", "fp_px", "shape_px", "centroid_px",
            "missed_pct", "fp_pct", "shape_pct", "centroid_pct",
            "missed_px_per_img", "fp_px_per_img", "shape_px_per_img",
            "centroid_px_per_img", "n_matched", "n_missed", "n_fp",
            "reconstruction_residual_pct"]
    sp = os.path.join(args.out_dir, "error_budget_summary.csv")
    with open(sp, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=cols); w.writeheader()
        for r in sorted(summary_rows, key=lambda r: (r["condition"], r["num_obs"])):
            w.writerow(r)
    rp = os.path.join(args.out_dir, "error_budget_raw.csv")
    with open(rp, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["id", "condition", "num_obs", "category", "pixels"])
        w.writeheader()
        for r in raw_rows:
            w.writerow(r)
    print(f"\n[SAVED] {sp}\n[SAVED] {rp}")

    # ---- 図：obsごとに条件①②③の積み上げ棒（構成比）----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cats = ["missed", "fp", "shape", "centroid"]
    colors = {"missed": "#d64545", "fp": "#eda100", "shape": "#2a78d6", "centroid": "#1baf7a"}
    bykey = {(r["condition"], r["num_obs"]): r for r in summary_rows}

    for obs in args.obs:
        conds = [c for c in conditions if (c, obs) in bykey]
        if not conds:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        bottom = np.zeros(len(conds))
        for cat in cats:
            vals = [bykey[(c, obs)][f"{cat}_pct"] for c in conds]
            ax.bar([f"cond{c}" for c in conds], vals, bottom=bottom,
                   color=colors[cat], label=cat)
            bottom += np.array(vals)
        ax.set_ylabel("total_error に対する構成比 [%]")
        ax.set_title(f"error budget (obs{obs})")
        ax.legend(); fig.tight_layout()
        p = os.path.join(args.out_dir, f"error_budget_stacked_obs{obs}.png")
        fig.savefig(p, dpi=120); plt.close(fig); print(f"[SAVED] {p}")

    # 条件①の obs 依存
    obs_have = [o for o in args.obs if (1, o) in bykey]
    if obs_have:
        fig, ax = plt.subplots(figsize=(7, 5))
        bottom = np.zeros(len(obs_have))
        for cat in cats:
            vals = [bykey[(1, o)][f"{cat}_pct"] for o in obs_have]
            ax.bar([f"obs{o}" for o in obs_have], vals, bottom=bottom,
                   color=colors[cat], label=cat)
            bottom += np.array(vals)
        ax.set_ylabel("total_error に対する構成比 [%]")
        ax.set_title("error budget vs obs (cond1)")
        ax.legend(); fig.tight_layout()
        p = os.path.join(args.out_dir, "error_budget_vs_obs_cond1.png")
        fig.savefig(p, dpi=120); plt.close(fig); print(f"[SAVED] {p}")


if __name__ == "__main__":
    main()
