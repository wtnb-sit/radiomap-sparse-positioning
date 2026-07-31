"""visualize_predictions.py：テスト予測を白黒マップで可視化

学習済みモデルでテストサンプルを推論し、正解ラベルと予測を並べて描く。
表現：**フリースペース(0)=白 / 障害物(1)=黒**（cmap='gray_r', vmin=0, vmax=1）。

各サンプルにつき横並びで:
    [正解 GT] [予測 Pred]   （--with-error 指定時は [誤差] も追加）
誤差パネル（任意）：TP=黒 / FP=赤（過検出）/ FN=青（見逃し）/ 背景=白。

前処理・入力生成は eval.py / dataset.py と完全に同一（RadioMapDataset 経由）。
条件は ckpt に記録された input_type を自動採用する。

使い方（GPU機・scripts フォルダで）:
  # 提案手法③・obs100 の test 先頭6サンプル
  python visualize_predictions.py --ckpt ../runs/cond3_obs100/best.pt --num-obs 100 --num 6
  # 特定IDを指定（誤差パネル付き）
  python visualize_predictions.py --ckpt ../runs/cond3_obs600/best.pt --num-obs 600 \
      --ids 5501,5502,5503 --with-error
  # 出力先を明示
  python visualize_predictions.py --ckpt ... --num-obs 100 --num 4 --out ../results/pred_c3.png
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import RadioMapDataset
from model import DCNUNet
from model_ablation import MaskedCNNUNet
from metrics import miou_obstacle


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DCNUNet() if ck["kind"] == "dcn" else MaskedCNNUNet()
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, ck


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description="テスト予測の白黒可視化")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num-obs", type=int, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--processed-root", default=os.path.join(root, "data", "processed"))
    ap.add_argument("--ids", default=None, help="カンマ区切りのサンプルID（例 5501,5502）")
    ap.add_argument("--num", type=int, default=6, help="--ids 未指定時に先頭から描く枚数")
    ap.add_argument("--with-error", action="store_true", help="誤差パネル(TP/FP/FN)を追加")
    ap.add_argument("--out", default=None, help="出力PNG（既定は ckpt と同じフォルダ）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(args.ckpt, device)
    input_type = ck["input_type"]
    print(f"ckpt: 条件{ck['condition']} ({ck['kind']},{input_type}) "
          f"obs{ck['num_obs']} val_mIoU={ck['val_miou']:.4f}  device={device}")

    ds = RadioMapDataset(args.processed_root, args.split, args.num_obs, input_type)

    # 描画対象のデータセット内インデックスを決める
    if args.ids:
        want = [int(s) for s in args.ids.split(",")]
        id2idx = {sid: i for i, sid in enumerate(ds.ids)}
        missing = [s for s in want if s not in id2idx]
        if missing:
            raise SystemExit(f"[ERROR] {args.split} 分割・obs{args.num_obs} に存在しないID: {missing}")
        idxs = [id2idx[s] for s in want]
    else:
        idxs = list(range(min(args.num, len(ds))))
    sids = [ds.ids[i] for i in idxs]
    print(f"{args.split}={len(ds)} から {len(idxs)}枚を描画: ids={sids}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    ncols = 3 if args.with_error else 2
    fig, axes = plt.subplots(len(idxs), ncols, figsize=(3.2 * ncols, 2.7 * len(idxs)),
                             squeeze=False)
    # 誤差用カラーマップ：0=背景(白) 1=TP(黒) 2=FP過検出(赤) 3=FN見逃し(青)
    err_cmap = ListedColormap(["white", "black", "#d64545", "#3a7bd5"])

    with torch.no_grad():
        for r, idx in enumerate(idxs):
            x, m, y = ds[idx]
            logits = model(x[None].to(device), m[None].to(device))
            pred = logits.argmax(dim=1)[0].cpu().numpy()          # (40,50) 0/1
            gt = y.numpy()                                         # (40,50) 0/1
            miou, oiou = miou_obstacle(logits.cpu(), y[None])

            # 白黒（0=白, 1=黒）
            axes[r][0].imshow(gt, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
            axes[r][1].imshow(pred, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
            axes[r][0].set_ylabel(f"id {sids[r]}", fontsize=10)
            if r == 0:
                axes[r][0].set_title("Ground truth")
                axes[r][1].set_title("Prediction")
            axes[r][1].set_xlabel(f"mIoU={float(miou):.3f}  obstIoU={float(oiou):.3f}",
                                  fontsize=9)

            if args.with_error:
                e = np.zeros_like(gt, dtype=np.uint8)
                e[(pred == 1) & (gt == 1)] = 1   # TP
                e[(pred == 1) & (gt == 0)] = 2   # FP 過検出
                e[(pred == 0) & (gt == 1)] = 3   # FN 見逃し
                axes[r][2].imshow(e, cmap=err_cmap, vmin=0, vmax=3, interpolation="nearest")
                if r == 0:
                    axes[r][2].set_title("Error (blk=TP red=FP blu=FN)")

            for c in range(ncols):
                axes[r][c].set_xticks([]); axes[r][c].set_yticks([])

    # 図中は日本語フォント非依存にするため ASCII で書く（□□化を回避）
    fig.suptitle(f"cond{ck['condition']} ({ck['kind']},{input_type}) "
                 f"{args.split} obs{args.num_obs}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                                   f"pred_maps_obs{args.num_obs}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
