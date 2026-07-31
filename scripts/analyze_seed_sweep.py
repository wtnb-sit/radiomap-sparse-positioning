"""analyze_seed_sweep.py：シードスイープの集計と「差 vs 揺れ」判定

train_seed_sweep.py が出した runs_seedsweep/cond1_{poserr|clean}_obs{N}_seed{S}/best.pt を
test 分割で評価し、誤差バジェット4費目（絶対画素）とmIoU・障害物IoUを集計する。
主目的は **poserr と clean の差が、学習シードの揺れを超えているか** の判定。

費目の算出は analyze_error_budget.py の実装をそのまま再利用（定義を作り直さない）。
指標は**絶対画素数を主表**とする（構成比%は他費目の増減で動き誤読を招くため）。

出力（既存 results/ は変更せず results_seedsweep/ に新規作成）:
  seed_sweep_summary.csv … 1行=1run（input_mode,num_obs,seed,miou,obstacle_iou,
                            total_error_px,missed_px,fp_px,shape_px,centroid_px）
  seed_sweep_verdict.csv … 結論の表。(input_mode,num_obs)ごとに指標の
                            平均/標準偏差/最小/最大、poserr−clean の平均差、
                            その差がシードsd・レンジを超えるかの判定フラグ
  shape_px_poserr_vs_clean.png / total_error_px_poserr_vs_clean.png

使い方:
  python analyze_seed_sweep.py
"""

import argparse
import csv
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import RadioMapDataset
from metrics import miou_obstacle
from diagnose_radial_offset import load_model
from analyze_error_budget import analyze_one

MODES = ["poserr", "clean"]
METRICS = ["shape_px", "centroid_px", "total_error_px", "miou", "obstacle_iou"]


def eval_miou(model, ds, device, batch=64):
    """eval.py と同一定義の per-image mIoU / 障害物IoU の平均。"""
    dl = DataLoader(ds, batch_size=batch, shuffle=False)
    mi, oi = [], []
    with torch.no_grad():
        for x, m, y in dl:
            logits = model(x.to(device), m.to(device)).cpu()
            a, b = miou_obstacle(logits, y)
            mi.append(a); oi.append(b)
    return float(torch.cat(mi).mean()), float(torch.cat(oi).mean())


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description="シードスイープの集計と差vs揺れ判定")
    ap.add_argument("--runs-dir", default=os.path.join(root, "runs_seedsweep"))
    ap.add_argument("--processed-root", default=os.path.join(root, "data", "processed"))
    ap.add_argument("--out-dir", default=os.path.join(root, "results_seedsweep"))
    ap.add_argument("--obs", type=int, nargs="*", default=[100, 600])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"device={device}  runs={args.runs_dir}")

    rows = []
    for mode in MODES:
        for obs in args.obs:
            for s in args.seeds:
                ck = os.path.join(args.runs_dir, f"cond1_{mode}_obs{obs}_seed{s}", "best.pt")
                if not os.path.exists(ck):
                    print(f"[skip] {ck} なし"); continue
                model, meta = load_model(ck, device)
                ds = RadioMapDataset(args.processed_root, "test", obs, meta["input_type"])
                miou, oiou = eval_miou(model, ds, device)
                acc, _, nm, nmiss, nfp = analyze_one(model, ds, device)
                r = {"input_mode": mode, "num_obs": obs, "seed": s,
                     "miou": round(miou, 4), "obstacle_iou": round(oiou, 4),
                     "total_error_px": acc["total"], "missed_px": acc["missed"],
                     "fp_px": acc["fp"], "shape_px": acc["shape"],
                     "centroid_px": acc["centroid"],
                     "n_matched": nm, "n_missed": nmiss, "n_fp": nfp}
                rows.append(r)
                print(f"{mode:>7} obs{obs:<5} seed{s}: mIoU={miou:.4f} "
                      f"total={acc['total']} shape={acc['shape']} centroid={acc['centroid']} "
                      f"missed={acc['missed']} fp={acc['fp']}")

    if not rows:
        print(f"[WARN] runが見つかりません（{args.runs_dir}）。"
              f"train_seed_sweep.py --sweep を先に実行してください。")
        return

    sp = os.path.join(args.out_dir, "seed_sweep_summary.csv")
    cols = ["input_mode", "num_obs", "seed", "miou", "obstacle_iou", "total_error_px",
            "missed_px", "fp_px", "shape_px", "centroid_px", "n_matched", "n_missed", "n_fp"]
    with open(sp, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=cols); w.writeheader()
        for r in sorted(rows, key=lambda r: (r["input_mode"], r["num_obs"], r["seed"])):
            w.writerow(r)
    print(f"\n[SAVED] {sp}")

    # ---- 判定表：差 vs シードの揺れ ----
    def group(mode, obs):
        return [r for r in rows if r["input_mode"] == mode and r["num_obs"] == obs]

    verdict = []
    for obs in args.obs:
        gp, gc = group("poserr", obs), group("clean", obs)
        if not gp or not gc:
            continue
        for met in METRICS:
            vp = np.array([r[met] for r in gp], dtype=float)
            vc = np.array([r[met] for r in gc], dtype=float)
            sd_p = float(vp.std(ddof=1)) if len(vp) > 1 else 0.0
            sd_c = float(vc.std(ddof=1)) if len(vc) > 1 else 0.0
            rng_p = float(vp.max() - vp.min()); rng_c = float(vc.max() - vc.min())
            diff = float(vp.mean() - vc.mean())        # poserr − clean
            max_sd = max(sd_p, sd_c); max_rng = max(rng_p, rng_c)
            verdict.append({
                "num_obs": obs, "metric": met,
                "poserr_mean": round(vp.mean(), 4), "poserr_sd": round(sd_p, 4),
                "poserr_min": round(vp.min(), 4), "poserr_max": round(vp.max(), 4),
                "clean_mean": round(vc.mean(), 4), "clean_sd": round(sd_c, 4),
                "clean_min": round(vc.min(), 4), "clean_max": round(vc.max(), 4),
                "diff_poserr_minus_clean": round(diff, 4),
                "max_seed_sd": round(max_sd, 4), "max_seed_range": round(max_rng, 4),
                "abs_diff_over_sd": round(abs(diff) / max_sd, 2) if max_sd > 1e-12 else None,
                "exceeds_seed_sd": bool(abs(diff) > max_sd),
                "exceeds_seed_range": bool(abs(diff) > max_rng),
                "ranges_overlap": bool(not (vp.min() > vc.max() or vc.min() > vp.max()))})

    if not verdict:
        print("\n[WARN] poserr と clean の両方が揃った (num_obs) がありません。"
              "判定表は作成しません（スイープ未完了の可能性）。")
        return

    vp_path = os.path.join(args.out_dir, "seed_sweep_verdict.csv")
    with open(vp_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(verdict[0].keys())); w.writeheader()
        for r in verdict:
            w.writerow(r)
    print(f"[SAVED] {vp_path}")

    print("\n=== 判定表（差 vs シードの揺れ）===")
    print(f"{'obs':>5} {'metric':>15} {'poserr(平均±sd)':>22} {'clean(平均±sd)':>22} "
          f"{'差':>10} {'|差|/sd':>8} {'>sd':>5} {'>range':>7} {'重なり':>7}")
    for r in verdict:
        print(f"{r['num_obs']:>5} {r['metric']:>15} "
              f"{r['poserr_mean']:>12.4f}±{r['poserr_sd']:<9.4f} "
              f"{r['clean_mean']:>12.4f}±{r['clean_sd']:<9.4f} "
              f"{r['diff_poserr_minus_clean']:>+10.4f} "
              f"{str(r['abs_diff_over_sd']):>8} {str(r['exceeds_seed_sd']):>5} "
              f"{str(r['exceeds_seed_range']):>7} {str(r['ranges_overlap']):>7}")

    # ---- 図 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for met in ("shape_px", "total_error_px"):
        fig, ax = plt.subplots(figsize=(7, 5))
        for mode, col, off in (("poserr", "#eb6834", -0.06), ("clean", "#1baf7a", 0.06)):
            xs, ms, sds = [], [], []
            for i, obs in enumerate(args.obs):
                g = group(mode, obs)
                if not g:
                    continue
                v = np.array([r[met] for r in g], dtype=float)
                xs.append(i + off); ms.append(v.mean())
                sds.append(v.std(ddof=1) if len(v) > 1 else 0.0)
                ax.scatter([i + off] * len(v), v, color=col, alpha=0.5, s=25, zorder=3)
            ax.errorbar(xs, ms, yerr=sds, fmt="o-", color=col, capsize=5,
                        lw=2, ms=8, label=f"{mode} (平均±sd)", zorder=2)
        ax.set_xticks(range(len(args.obs)))
        ax.set_xticklabels([f"obs{o}" for o in args.obs])
        ax.set_ylabel(met); ax.set_title(f"{met}: poserr vs clean (3 seeds)")
        ax.legend(); ax.grid(alpha=0.3)
        p = os.path.join(args.out_dir, f"{met}_poserr_vs_clean.png")
        fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
        print(f"[SAVED] {p}")


if __name__ == "__main__":
    main()
