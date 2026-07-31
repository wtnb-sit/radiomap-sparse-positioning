"""aggregate_results.py：全 test_metrics.json を集約

runs/cond<ID>_obs<N>/test_metrics.json を集め、
  ・観測点数 vs mIoU グラフ（条件別・エラーバー=mIoU_std）
  ・summary.csv（全指標）
を results/ に出力する（training_eval_spec.md §9.3）。

使い方:
  python aggregate_results.py
  python aggregate_results.py --runs-dir ../runs --out-dir ../results
"""

import argparse
import os
import glob
import json
import csv

COND_LABEL = {1: "(1) MaskedCNN+Conv, residual",
              2: "(2) MaskedCNN+DCN, raw",
              3: "(3) proposed: MaskedCNN+DCN, residual"}
METRIC_KEYS = ["mIoU", "obstacle_IoU", "F1", "HD95",
               "BoundaryF1@1", "BoundaryF1@3", "BoundaryF1@5"]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description="結果集約")
    ap.add_argument("--runs-dir", default=os.path.join(root, "runs"))
    ap.add_argument("--out-dir", default=os.path.join(root, "results"))
    args = ap.parse_args()

    rows = []
    for f in glob.glob(os.path.join(args.runs_dir, "cond*_obs*", "test_metrics.json")):
        with open(f, encoding="utf-8") as fp:
            rows.append(json.load(fp))
    if not rows:
        print(f"[WARN] test_metrics.json が見つかりません（{args.runs_dir}）")
        return
    os.makedirs(args.out_dir, exist_ok=True)

    # summary.csv
    fields = ["condition", "num_obs", "n"] + METRIC_KEYS + ["mIoU_std"]
    csv_path = os.path.join(args.out_dir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["condition"], r["num_obs"])):
            w.writerow(r)
    print(f"[SAVED] {csv_path}  （{len(rows)}件）")

    # 観測点数 vs mIoU グラフ（条件別）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by_cond = {}
    for r in rows:
        by_cond.setdefault(r["condition"], []).append(r)
    fig, ax = plt.subplots(figsize=(9, 6))
    for cond in sorted(by_cond):
        pts = sorted(by_cond[cond], key=lambda r: r["num_obs"])
        xs = [p["num_obs"] for p in pts]
        ys = [p["mIoU"] for p in pts]
        es = [p.get("mIoU_std", 0.0) for p in pts]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                    label=COND_LABEL.get(cond, f"cond{cond}"))
    ax.set_xlabel("number of observations (num_obs)")
    ax.set_ylabel("mIoU (test, macro mean)")
    ax.set_title("Observation count vs mIoU")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_png = os.path.join(args.out_dir, "num_obs_vs_mIoU.png")
    fig.tight_layout(); fig.savefig(out_png, dpi=120)
    print(f"[SAVED] {out_png}")


if __name__ == "__main__":
    main()
