from __future__ import annotations
import torch.nn as nn
from ..registry import MODELS

@MODELS.register("dncnn")
class DnCNN(nn.Module):
    # DnCNN residual denoiser (x - noise)
    # NOTE: this follows the structure in the user's DNCNN.py:
    # first conv+relu, then `depth` times (conv+bn+relu), then last conv.
    def __init__(self, depth: int = 20, num_channels: int = 64, image_channels: int = 1):
        super().__init__()
        layers = []

        layers.append(nn.Conv2d(image_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(depth):
            layers.append(nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_channels))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(num_channels, image_channels, kernel_size=3, stride=1, padding=1, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.dncnn(x)
        return x - noise
