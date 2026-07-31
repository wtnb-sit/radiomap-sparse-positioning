"""条件①用モデル：マスク付きCNN + 通常Conv（DCNなし）

提案手法（model.py の DCNUNet）から DCNv2Block を通常のConvブロックに
差し替えたバリアント。エンコーダ/デコーダ/ダウンサンプリング/マスク埋め尽くし
ブロックは model.py の実装をそのまま流用し、DCN部分のみ置換する。

  ① vs ③（提案手法）で「DCNの効果」を分離するためのアブレーション条件。

入力は残差（条件①）。損失は model.py の seg_loss を共用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 共通部品は model.py から流用（重複実装を避ける）
from model import MaskFillingBlock, Down, UpBlock


class ConvBlock(nn.Module):
    """DCNv2Block の代替：通常 Conv(k=3,stride1,pad1) + BN + ReLU。"""

    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=1, padding=k // 2)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class MaskedCNNUNet(nn.Module):
    """条件①：マスク埋め尽くし + 通常Convエンコーダ U-Net（DCNなし）。

    DCNUNet（提案手法）と構造は同一で、各解像度のDCNv2Blockを
    ConvBlockに置換しただけ。skip接続・ダウンサンプリング・デコーダは同一。
    """

    def __init__(self, C=64, k_fill=9, N_fill=4, conv_k=3, n_classes=2):
        super().__init__()
        self.fill = MaskFillingBlock(1, C, k_fill, N_fill)
        # エンコーダ（DCNの代わりに通常Conv → skip、Downで1/2）
        self.enc0 = ConvBlock(C, C, conv_k)   # 40x50
        self.down0 = Down(C)                  # -> 20x25
        self.enc1 = ConvBlock(C, C, conv_k)   # 20x25
        self.down1 = Down(C)                  # -> 10x13
        self.enc2 = ConvBlock(C, C, conv_k)   # 10x13
        self.down2 = Down(C)                  # -> 5x7 (bottleneck)
        # デコーダ（提案手法と同一）
        self.up2 = UpBlock(C, C, C)
        self.up1 = UpBlock(C, C, C)
        self.up0 = UpBlock(C, C, C)
        self.out_conv = nn.Conv2d(C, n_classes, 1)

    def forward(self, x, mask):
        x, mask = self.fill(x, mask)
        s0 = self.enc0(x)
        x = self.down0(s0)
        s1 = self.enc1(x)
        x = self.down1(s1)
        s2 = self.enc2(x)
        x = self.down2(s2)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.up0(x, s0)
        return self.out_conv(x)

    def offset_parameters(self):
        """DCNなしのためオフセット層は存在しない（空）。"""
        return iter(())

    def base_parameters(self):
        return self.parameters()


if __name__ == "__main__":
    # スモークテスト
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MaskedCNNUNet().to(dev)
    n = sum(p.numel() for p in model.parameters())
    print(f"device={dev}  条件①モデル 総パラメータ数: {n:,}")
    B = 4
    x = torch.randn(B, 1, 40, 50, device=dev)
    mask = torch.zeros(B, 1, 40, 50, device=dev)
    for b in range(B):
        idx = torch.randperm(2000)[:100]
        mask.view(B, -1)[b, idx] = 1.0
    x = x * mask
    with torch.no_grad():
        _, filled = model.fill(x, mask)
    print(f"マスク充填: {int(mask.sum(dim=(1,2,3))[0])} -> {int(filled.sum(dim=(1,2,3))[0])}/2000")
    logits = model(x, mask)
    print(f"出力 logits: {tuple(logits.shape)} (期待 (4,2,40,50))")
    logits.sum().backward()
    print("スモークテスト完了")
