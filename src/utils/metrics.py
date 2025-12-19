from __future__ import annotations
import math
import torch
import torch.nn.functional as F

def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0, eps: float = 1e-8) -> float:
    mse = F.mse_loss(pred, target, reduction="mean").item()
    return float(10.0 * math.log10((data_range * data_range) / max(mse, eps)))

def _gaussian_window(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    return g

def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0, window_size: int = 11, sigma: float = 1.5, eps: float = 1e-8) -> float:
    # Basic dependency-free SSIM for NCHW
    device, dtype = pred.device, pred.dtype
    C = pred.shape[1]
    g1 = _gaussian_window(window_size, sigma, device, dtype).view(1, 1, 1, -1)
    g2 = _gaussian_window(window_size, sigma, device, dtype).view(1, 1, -1, 1)

    def blur(x):
        x = F.conv2d(x, g1.expand(C,1,1,window_size), padding=(0, window_size//2), groups=C)
        x = F.conv2d(x, g2.expand(C,1,window_size,1), padding=(window_size//2, 0), groups=C)
        return x

    mu_x = blur(pred)
    mu_y = blur(target)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = blur(pred * pred) - mu_x2
    sigma_y2 = blur(target * target) - mu_y2
    sigma_xy = blur(pred * target) - mu_xy

    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / (den + eps)
    return float(ssim_map.mean().item())
