from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import MODELS  # สำคัญ: ใช้ MODELS.register


# -----------------------------
# Plain U-Net
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


@MODELS.register("Unet")  # <-- ต้องตรงกับ config: model.name
class UNetDenoise(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()

        ch_in = in_channels
        for ch_out in features:
            self.downs.append(DoubleConv(ch_in, ch_out))
            self.pools.append(nn.MaxPool2d(2))
            ch_in = ch_out

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        ch_in = features[-1] * 2
        for ch_out in reversed(features):
            self.ups.append(nn.ConvTranspose2d(ch_in, ch_out, 2, stride=2))
            self.up_convs.append(DoubleConv(ch_in, ch_out))
            ch_in = ch_out

        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for up, conv, skip in zip(self.ups, self.up_convs, skips):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = conv(x)

        return self.head(x)


# -----------------------------
# UCX / MICXN-style (ConvNeXt blocks)
# -----------------------------
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or (not self.training):
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4, drop_path: float = 0.0, layer_scale_init: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones((dim,))) if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        shortcut = x
        x = self.d
