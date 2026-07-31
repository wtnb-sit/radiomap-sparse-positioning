"""Step 5：model.py（提案手法：PConv + マルチスケールDCNv2 U-Net）

research_design_final.md §5 の構造を実装する。

  入力 40x50x1（正規化残差）+ マスク
   └ マスク埋め尽くしブロック：PConv(k=9,stride1)+Masked BatchNorm+ReLU ×N=4
   └ マルチスケールDCNエンコーダ：各解像度DCNv2(stride1)→skip、Strided Convで1/2
       40x50 →20x25 →10x13 →5x7（ceil方式のstrided conv、パディングなし）
   └ デコーダ（U-Net）：Upsample→skip concat→Conv×2 ×3
   └ 出力 1x1 Conv(2ch) → logits 40x50x2

確定パラメータ：k=9, N=4（§8.2）。C（特徴チャネル）=64 を既定。
DCNv2：オフセット推定層はゼロ初期化（Δp=0, Δm=0.5）。学習率0.1倍は
学習スクリプト側で offset_parameters() を別グループにして設定する（§5.5）。

損失：重み付きCE + α·Dice（40x50全域で計算、パディングなし）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


# ---- Partial Convolution (Liu et al., 2018) ------------------------------
class PartialConv2d(nn.Module):
    """マスク付き畳み込み。マスクは1チャネル共有、有効カーネル面積比でリスケール。"""

    def __init__(self, in_ch, out_ch, k, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride, padding, bias=bias)
        self.stride, self.padding = stride, padding
        self.register_buffer("ones", torch.ones(1, 1, k, k))
        self.winsize = k * k

    def forward(self, x, mask):
        # x: (B,Cin,H,W), mask: (B,1,H,W)
        with torch.no_grad():
            mask_sum = F.conv2d(mask, self.ones, stride=self.stride,
                                padding=self.padding)          # (B,1,H',W')
        raw = self.conv(x * mask)
        b = self.conv.bias.view(1, -1, 1, 1) if self.conv.bias is not None else 0.0
        scale = self.winsize / (mask_sum + 1e-8)               # 有効面積比
        out = (raw - b) * scale + b
        new_mask = (mask_sum > 0).float()
        out = out * new_mask                                   # 有効画素なしは0
        return out, new_mask


# ---- Masked BatchNorm ----------------------------------------------------
class MaskedBatchNorm2d(nn.Module):
    """有効グリッド（マスク=1）のみで統計量を計算するBatchNorm。"""

    def __init__(self, C, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.weight = nn.Parameter(torch.ones(C))
        self.bias = nn.Parameter(torch.zeros(C))
        self.register_buffer("running_mean", torch.zeros(C))
        self.register_buffer("running_var", torch.ones(C))

    def forward(self, x, mask):
        if self.training:
            m = mask                                     # (B,1,H,W) 共有
            n = m.sum().clamp(min=1.0)                   # 有効画素総数
            mean = (x * m).sum(dim=(0, 2, 3)) / n        # (C,)
            var = (((x - mean[None, :, None, None]) ** 2) * m).sum(dim=(0, 2, 3)) / n
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var)
        else:
            mean, var = self.running_mean, self.running_var
        xhat = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        return xhat * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class MaskFillingBlock(nn.Module):
    """PConv + Masked BatchNorm + ReLU を N 層。マスクを埋めながら密化する。"""

    def __init__(self, in_ch=1, C=64, k=9, N=4):
        super().__init__()
        self.pconvs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(N):
            ci = in_ch if i == 0 else C
            self.pconvs.append(PartialConv2d(ci, C, k, stride=1, padding=k // 2, bias=True))
            self.bns.append(MaskedBatchNorm2d(C))

    def forward(self, x, mask):
        for pconv, bn in zip(self.pconvs, self.bns):
            x, mask = pconv(x, mask)
            x = F.relu(bn(x, mask))
        return x, mask


# ---- DCNv2 block ---------------------------------------------------------
class DCNv2Block(nn.Module):
    """DeformConv2d(v2, 変調あり) + BN + ReLU。オフセット層はゼロ初期化。"""

    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        pad = k // 2
        self.k = k
        # 経路A：オフセット(2*k*k) + 変調(k*k) を推定。ゼロ初期化。
        self.offset_conv = nn.Conv2d(in_ch, 3 * k * k, k, padding=pad)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        # 経路B：変形サンプリング
        self.deform = DeformConv2d(in_ch, out_ch, k, padding=pad)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        o = self.offset_conv(x)
        ck = self.k * self.k
        offset = o[:, :2 * ck]
        modulation = torch.sigmoid(o[:, 2 * ck:])     # 初期は sigmoid(0)=0.5
        y = self.deform(x, offset, modulation)
        return F.relu(self.bn(y))


class Down(nn.Module):
    """Strided Conv(stride2,k3,p1) + BN + ReLU。ceil方式で 50->25->13->7。"""

    def __init__(self, C):
        super().__init__()
        self.conv = nn.Conv2d(C, C, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(C)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class UpBlock(nn.Module):
    """Upsample(skipサイズへ) → skip concat → Conv×2。任意サイズ対応。"""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, 1)
        self.conv1 = nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class DCNUNet(nn.Module):
    """提案手法：マスク埋め尽くし + マルチスケールDCN U-Net。"""

    def __init__(self, C=64, k_fill=9, N_fill=4, dcn_k=3, n_classes=2):
        super().__init__()
        self.fill = MaskFillingBlock(1, C, k_fill, N_fill)
        # エンコーダ（各解像度DCN → skip、Downで1/2）
        self.dcn0 = DCNv2Block(C, C, dcn_k)   # 40x50
        self.down0 = Down(C)                  # -> 20x25
        self.dcn1 = DCNv2Block(C, C, dcn_k)   # 20x25
        self.down1 = Down(C)                  # -> 10x13
        self.dcn2 = DCNv2Block(C, C, dcn_k)   # 10x13
        self.down2 = Down(C)                  # -> 5x7 (bottleneck)
        # デコーダ
        self.up2 = UpBlock(C, C, C)           # -> 10x13 (+skip dcn2)
        self.up1 = UpBlock(C, C, C)           # -> 20x25 (+skip dcn1)
        self.up0 = UpBlock(C, C, C)           # -> 40x50 (+skip dcn0)
        self.out_conv = nn.Conv2d(C, n_classes, 1)

    def forward(self, x, mask):
        x, mask = self.fill(x, mask)          # 密化 (B,C,40,50)
        s0 = self.dcn0(x)                      # skip 40x50
        x = self.down0(s0)
        s1 = self.dcn1(x)                      # skip 20x25
        x = self.down1(s1)
        s2 = self.dcn2(x)                      # skip 10x13
        x = self.down2(s2)                     # bottleneck 5x7
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.up0(x, s0)
        return self.out_conv(x)               # logits (B,2,40,50)

    def offset_parameters(self):
        """DCNオフセット推定層のパラメータ（学習率0.1倍グループ用）。"""
        for m in self.modules():
            if isinstance(m, DCNv2Block):
                yield from m.offset_conv.parameters()

    def base_parameters(self):
        """オフセット層以外のパラメータ。"""
        offset_ids = {id(p) for p in self.offset_parameters()}
        for p in self.parameters():
            if id(p) not in offset_ids:
                yield p


# ---- 損失 ----------------------------------------------------------------
def dice_loss(prob_obst, target_obst, eps=1e-6):
    """障害物クラスのソフトDice損失。prob/target: (B,H,W)。"""
    num = 2.0 * (prob_obst * target_obst).sum(dim=(1, 2)) + eps
    den = prob_obst.sum(dim=(1, 2)) + target_obst.sum(dim=(1, 2)) + eps
    return (1.0 - num / den).mean()


def seg_loss(logits, target, class_weights, alpha):
    """重み付きCE + α·Dice（40x50全域で計算）。

    logits: (B,2,H,W), target: (B,H,W) long, class_weights: (2,) tensor
    """
    ce = F.cross_entropy(logits, target, weight=class_weights)
    prob_obst = F.softmax(logits, dim=1)[:, 1]
    dice = dice_loss(prob_obst, (target == 1).float())
    return ce + alpha * dice, ce.detach(), dice.detach()


if __name__ == "__main__":
    # スモークテスト：順伝播・逆伝播・マスク充填・DCN初期挙動を確認
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}")
    model = DCNUNet().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    n_off = sum(p.numel() for p in model.offset_parameters())
    print(f"総パラメータ数: {n_params:,}  うちDCNオフセット層: {n_off:,}")

    B = 4
    x = torch.randn(B, 1, 40, 50, device=dev)
    # 観測点100個相当の疎マスク
    mask = torch.zeros(B, 1, 40, 50, device=dev)
    for b in range(B):
        idx = torch.randperm(2000)[:100]
        mask.view(B, -1)[b, idx] = 1.0
    x = x * mask
    target = torch.randint(0, 2, (B, 40, 50), device=dev)
    w = torch.tensor([0.19, 1.81], device=dev)

    # マスク充填の確認
    with torch.no_grad():
        _, filled = model.fill(x, mask)
    print(f"マスク充填: 入力={int(mask.sum(dim=(1,2,3))[0])} -> "
          f"出力={int(filled.sum(dim=(1,2,3))[0])} / 2000 (全埋まりが目標)")

    logits = model(x, mask)
    print(f"出力 logits: {tuple(logits.shape)} (期待 (4,2,40,50))")
    loss, ce, dice = seg_loss(logits, target, w, alpha=0.5)
    loss.backward()
    print(f"loss={loss.item():.4f} (CE={ce.item():.4f}, Dice={dice.item():.4f})")
    # DCNオフセットのゼロ初期化 → 初期勾配が流れているか
    g = next(model.dcn0.offset_conv.parameters()).grad
    print(f"DCNオフセット層の勾配ノルム: {g.norm().item():.4e} (学習開始時0初期化・勾配は非ゼロ)")
    print("スモークテスト完了")
