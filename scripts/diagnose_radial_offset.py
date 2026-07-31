"""diagnose_radial_offset.py：予測障害物の「動径方向の重心ズレ」を診断

単一TX固定のRSSから障害物を推定した際、予測障害物の重心が真値からズレる原因が
「TXから放射方向外側への系統バイアス」か「零平均ランダムズレ」かを判定する。
新規学習はしない。既存 best.pt で推論して重心ズレの分布を出すだけ。

【手順（各テスト画像）】
  1. 予測/GTマスクを scipy.ndimage.label で連結成分に分解
  2. IoU最大で貪欲マッチング（IoU=0のGT=未検出missed、未マッチ予測=誤検出fp、
     いずれもズレ統計から除外）
  3. マッチ成分ペアの重心差 Δ = c_pred − c_gt（セル・(row,col)）
  4. TXセル→GT成分重心の単位ベクトル û_r（外向き正）と直交 û_t で
     Δ_r = Δ·û_r, Δ_t = Δ·û_t を記録

TXセルは FSPL マップ（data/processed/fspl_map.npy）の argmax セル（全データ共通固定）。
予測生成は visualize_predictions.load_model、2値化は eval.py と同じ argmax（障害物=class1）、
GT・test分割は dataset.RadioMapDataset を再利用する。

使い方（runs と data がある機で・scripts フォルダ）:
  python diagnose_radial_offset.py                       # 条件①③②・obs{100,600,1200}
  python diagnose_radial_offset.py --conditions 1 --obs 600
"""

import argparse
import csv
import os
import numpy as np
import torch
from scipy import ndimage
from scipy.stats import binomtest, ttest_1samp, wilcoxon, t as t_dist

from dataset import RadioMapDataset
from model import DCNUNet
from model_ablation import MaskedCNNUNet


def load_model(ckpt_path, device):
    """visualize_predictions.load_model と同一の読み込み方法。"""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DCNUNet() if ck["kind"] == "dcn" else MaskedCNNUNet()
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, ck


def components(mask):
    """2値マスク(bool)→(ラベル配列, 成分数)。既定の4連結。"""
    lab, n = ndimage.label(mask)
    return lab, n


def centroid(lab, k):
    """ラベルkの重心(row,col)。"""
    rc = np.argwhere(lab == k)
    return rc.mean(axis=0)  # (row, col)


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union > 0 else 0.0


def match_components(pred_lab, n_pred, gt_lab, n_gt):
    """貪欲マッチング：IoU降順に (gt,pred) を1対1で対応づける。
    返り値: matches[(gt_k, pred_k, iou)], missed(int), fp(int)。"""
    pairs = []
    for g in range(1, n_gt + 1):
        gm = (gt_lab == g)
        for p in range(1, n_pred + 1):
            v = iou(gm, (pred_lab == p))
            if v > 0:
                pairs.append((v, g, p))
    pairs.sort(reverse=True)  # IoU降順
    used_g, used_p, matches = set(), set(), []
    for v, g, p in pairs:
        if g in used_g or p in used_p:
            continue
        used_g.add(g); used_p.add(p); matches.append((g, p, v))
    missed = n_gt - len(used_g)          # どの予測ともIoU>0でマッチしないGT
    fp = n_pred - len(used_p)            # どのGTともマッチしない予測
    return matches, missed, fp


def analyze(model, ds, tx_rc, device, verbose_n=0):
    """1モデルの全test画像を処理し、per-match の記録リストと件数を返す。"""
    recs = []                       # dict: id, dr, dt, dist_from_tx, gt_area
    n_missed = n_fp = n_degenerate = 0
    tx = np.asarray(tx_rc, dtype=float)
    shown = 0
    with torch.no_grad():
        for idx in range(len(ds)):
            sid = ds.ids[idx]
            x, m, y = ds[idx]
            logits = model(x[None].to(device), m[None].to(device))
            pred = (logits.argmax(dim=1)[0].cpu().numpy() == 1)
            gt = (y.numpy() == 1)
            pl, npred = components(pred)
            gl, ngt = components(gt)
            matches, missed, fp = match_components(pl, npred, gl, ngt)
            n_missed += missed; n_fp += fp
            for g, p, v in matches:
                c_gt = centroid(gl, g)
                c_pred = centroid(pl, p)
                delta = c_pred - c_gt            # (row,col)
                r_vec = c_gt - tx                # TX→GT重心
                dist = np.linalg.norm(r_vec)
                if dist < 1e-9:
                    n_degenerate += 1
                    continue
                u_r = r_vec / dist               # 外向き単位ベクトル
                u_t = np.array([-u_r[1], u_r[0]])  # 直交（90度回転）
                dr = float(delta @ u_r)          # 外向き正
                dt = float(delta @ u_t)
                recs.append({"id": sid, "dr": dr, "dt": dt,
                             "dist_from_tx": float(dist),
                             "gt_area": int((gl == g).sum())})
                if shown < verbose_n:
                    print(f"    [sanity id={sid}] GT重心=({c_gt[0]:.2f},{c_gt[1]:.2f}) "
                          f"pred重心=({c_pred[0]:.2f},{c_pred[1]:.2f}) "
                          f"Δ=({delta[0]:+.2f},{delta[1]:+.2f}) Δr={dr:+.3f}(外向き正)")
                    shown += 1
    return recs, n_missed, n_fp, n_degenerate


def summarize(recs):
    dr = np.array([r["dr"] for r in recs], dtype=float)
    n = len(dr)
    if n == 0:
        return None
    mean = float(dr.mean()); med = float(np.median(dr)); sd = float(dr.std(ddof=1)) if n > 1 else 0.0
    sem = sd / np.sqrt(n) if n > 1 else 0.0
    if n > 1 and sem > 0:
        lo, hi = t_dist.interval(0.95, n - 1, loc=mean, scale=sem)
    else:
        lo = hi = mean
    npos = int((dr > 0).sum()); nneg = int((dr < 0).sum()); nnz = npos + nneg
    p_sign = float(binomtest(npos, nnz, 0.5).pvalue) if nnz > 0 else 1.0
    p_t = float(ttest_1samp(dr, 0.0).pvalue) if n > 1 else 1.0
    try:
        p_w = float(wilcoxon(dr).pvalue) if nnz > 0 else 1.0
    except ValueError:
        p_w = 1.0
    return {"n": n, "mean_dr": mean, "median_dr": med, "std_dr": sd,
            "ci95_low": float(lo), "ci95_high": float(hi),
            "frac_outward": npos / n, "median_abs_dr": float(np.median(np.abs(dr))),
            "p_signtest": p_sign, "p_ttest": p_t, "p_wilcoxon": p_w,
            "effect_size": mean / sd if sd > 1e-12 else 0.0}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description="予測障害物の動径方向重心ズレ診断")
    ap.add_argument("--runs-dir", default=os.path.join(root, "runs"))
    ap.add_argument("--processed-root", default=os.path.join(root, "data", "processed"))
    ap.add_argument("--out-dir", default=os.path.join(root, "results"))
    ap.add_argument("--conditions", default="1,3,2")
    ap.add_argument("--obs", type=int, nargs="*", default=[100, 600, 1200])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    conditions = [int(c) for c in args.conditions.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- TXセル特定（FSPL argmax）＋サニティ印字 ----
    fspl = np.load(os.path.join(args.processed_root, "fspl_map.npy"))
    tx_r, tx_c = np.unravel_index(int(np.argmax(fspl)), fspl.shape)
    print(f"[TX] FSPL argmax セル = (row={tx_r}, col={tx_c})  値={fspl[tx_r, tx_c]:.3f}  "
          f"（FSPL最大＝TX。想定 row≈19-20,col≈24-25）")
    print(f"[TX] FSPL min={fspl.min():.3f} max={fspl.max():.3f}  device={device}\n")
    tx_rc = (float(tx_r), float(tx_c))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_rows, raw_rows = [], []
    per_obs_for_compare = {o: {} for o in args.obs}

    for cond in conditions:
        for obs in args.obs:
            ckpt = os.path.join(args.runs_dir, f"cond{cond}_obs{obs}", "best.pt")
            if not os.path.exists(ckpt):
                print(f"[skip] {ckpt} なし"); continue
            model, ck = load_model(ckpt, device)
            ds = RadioMapDataset(args.processed_root, "test", obs, ck["input_type"])
            vb = 3 if (cond == conditions[0] and obs == args.obs[0]) else 0
            if vb:
                print(f"[sanity] 条件{cond} obs{obs} 先頭マッチ成分の符号確認:")
            recs, missed, fp, degen = analyze(model, ds, tx_rc, device, verbose_n=vb)
            s = summarize(recs)
            if s is None:
                print(f"条件{cond} obs{obs}: マッチ成分0（missed={missed}, fp={fp}）"); continue
            print(f"条件{cond} obs{obs}: n_matched={s['n']} missed={missed} fp={fp} "
                  f"degen={degen} | 平均Δr={s['mean_dr']:+.3f} 中央={s['median_dr']:+.3f} "
                  f"外向き={s['frac_outward']:.1%} | p_sign={s['p_signtest']:.2e} "
                  f"p_t={s['p_ttest']:.2e} d={s['effect_size']:+.3f}")
            summary_rows.append({
                "condition": cond, "num_obs": obs, "n_matched": s["n"],
                "n_missed": missed, "n_fp": fp,
                "mean_dr": round(s["mean_dr"], 4), "median_dr": round(s["median_dr"], 4),
                "std_dr": round(s["std_dr"], 4), "ci95_low": round(s["ci95_low"], 4),
                "ci95_high": round(s["ci95_high"], 4),
                "frac_outward": round(s["frac_outward"], 4),
                "p_signtest": s["p_signtest"], "effect_size": round(s["effect_size"], 4)})
            for r in recs:
                raw_rows.append({"id": r["id"], "condition": cond, "num_obs": obs,
                                 "dr": round(r["dr"], 4), "dt": round(r["dt"], 4),
                                 "dist_from_tx": round(r["dist_from_tx"], 4),
                                 "gt_area": r["gt_area"]})
            per_obs_for_compare[obs][cond] = np.array([r["dr"] for r in recs])

            # 条件別ヒスト
            dr = np.array([r["dr"] for r in recs])
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(dr, bins=50, color="steelblue", edgecolor="white")
            ax.axvline(0, color="k", ls="-", lw=1, label="0 (ズレなし)")
            ax.axvline(dr.mean(), color="red", ls="--", lw=2, label=f"平均={dr.mean():+.3f}")
            ax.set_title(f"cond{cond} obs{obs}  radial offset Δr (outward +)  n={len(dr)}")
            ax.set_xlabel("Δr [cells] (TX outward = +)"); ax.set_ylabel("count"); ax.legend()
            p = os.path.join(args.out_dir, f"radial_offset_hist_cond{cond}_obs{obs}.png")
            fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
            print(f"  [SAVED] {p}")

    # 条件比較ヒスト（obsごと）
    for obs, byc in per_obs_for_compare.items():
        if len(byc) < 2:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
        for cond, dr in sorted(byc.items()):
            ax.hist(dr, bins=50, histtype="step", lw=2, color=colors.get(cond),
                    label=f"cond{cond} (平均{dr.mean():+.3f})")
        ax.axvline(0, color="k", ls="-", lw=1)
        ax.set_title(f"radial offset Δr 比較  obs{obs}  (outward +)")
        ax.set_xlabel("Δr [cells] (TX outward = +)"); ax.set_ylabel("count"); ax.legend()
        p = os.path.join(args.out_dir, f"radial_offset_hist_compare_obs{obs}.png")
        fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
        print(f"[SAVED] {p}")

    # CSV
    sp = os.path.join(args.out_dir, "radial_offset_summary.csv")
    with open(sp, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["condition", "num_obs", "n_matched",
            "n_missed", "n_fp", "mean_dr", "median_dr", "std_dr", "ci95_low",
            "ci95_high", "frac_outward", "p_signtest", "effect_size"])
        w.writeheader()
        for r in sorted(summary_rows, key=lambda r: (r["condition"], r["num_obs"])):
            w.writerow(r)
    rp = os.path.join(args.out_dir, "radial_offset_raw.csv")
    with open(rp, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["id", "condition", "num_obs", "dr", "dt",
                                           "dist_from_tx", "gt_area"])
        w.writeheader()
        for r in raw_rows:
            w.writerow(r)
    print(f"\n[SAVED] {sp}\n[SAVED] {rp}")


if __name__ == "__main__":
    main()
