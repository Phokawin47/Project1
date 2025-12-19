import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data import BrainTumorDataset
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import math
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 2) U-Net (เรียบง่าย ไม่มีของเสริม)
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
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

class UNetDenoise(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch_in = in_channels
        for ch_out in features:
            self.downs.append(DoubleConv(ch_in, ch_out))
            self.pools.append(nn.MaxPool2d(2))
            ch_in = ch_out

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        ch_in = features[-1]*2
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
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = conv(x)

        return self.head(x)

# -----------------------------
# 3) Early Stopping Class
# -----------------------------
class EarlyStopping:
    def __init__(self, patience=7, restore_best_weights=True, monitor='val_psnr', mode='max'):
        """
        Early stopping to stop training when validation metric stops improving.
        
        Args:
            patience (int): Number of epochs to wait after last improvement
            restore_best_weights (bool): Whether to restore model weights from the best epoch
            monitor (str): Metric to monitor ('val_psnr' or 'val_loss')
            mode (str): 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.monitor = monitor
        self.mode = mode
        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = None
        self.best_weights = None
        
        if mode == 'max':
            self.monitor_op = lambda current, best: current > best
        else:
            self.monitor_op = lambda current, best: current < best
    
    def __call__(self, current_score, model):
        """
        Check if training should stop.
        
        Args:
            current_score (float): Current validation score
            model: PyTorch model to save weights from
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(model)
        elif self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            self.wait = 0
            self.save_checkpoint(model)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.wait
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        return False
    
    def save_checkpoint(self, model):
        """Save model weights."""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()

# -----------------------------
# 4) Utils: PSNR + แสดงภาพ
# -----------------------------
def psnr(pred, target, max_val=1.0, eps=1e-8):
    mse = torch.mean((pred - target) ** 2)
    if mse <= eps:
        return torch.tensor(100.0, device=pred.device)
    return 20.0 * torch.log10(max_val / torch.sqrt(mse + eps))

def ssim(pred, target, window_size=11, sigma=1.5, k1=0.01, k2=0.03, max_val=1.0):
    """Calculate SSIM between two images"""
    def gaussian_window(size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.outer(g).unsqueeze(0).unsqueeze(0)
    
    window = gaussian_window(window_size, sigma).to(pred.device)
    mu1 = F.conv2d(pred, window, padding=window_size//2, groups=pred.shape[1])
    mu2 = F.conv2d(target, window, padding=window_size//2, groups=target.shape[1])
    
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
    
    sigma1_sq = F.conv2d(pred*pred, window, padding=window_size//2, groups=pred.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(target*target, window, padding=window_size//2, groups=target.shape[1]) - mu2_sq
    sigma12 = F.conv2d(pred*target, window, padding=window_size//2, groups=pred.shape[1]) - mu1_mu2
    
    c1, c2 = (k1*max_val)**2, (k2*max_val)**2
    ssim_map = ((2*mu1_mu2 + c1)*(2*sigma12 + c2)) / ((mu1_sq + mu2_sq + c1)*(sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


# =============================
# 2-b) UCX / MICXN-style U-Net (ConvNeXt blocks) + (optional) Data Consistency layer
#     (ref: "A Denoising UNet Model with ConvNeXt Block for MRI Reconstruction", 2024)
# =============================
from typing import Optional

class LayerNorm2d(nn.Module):
    """LayerNorm over channel dimension for NCHW tensors (as used in ConvNeXt).
    Applies nn.LayerNorm to channels in channels-last format then permutes back.
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W) -> (N, H, W, C) -> LN -> (N, C, H, W)
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x

class DropPath(nn.Module):
    """Stochastic depth. Disabled by default (drop_prob=0)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or (not self.training):
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class ConvNeXtBlock(nn.Module):
    """ConvNeXt block (simplified) for 2D images.
    Paper differences vs classic U-Net:
      - depthwise conv with larger kernel (7x7)
      - LayerNorm + GELU
      - 1x1 expansions (implemented as Linear in channels-last)
    """
    def __init__(self, dim: int, mlp_ratio: int = 4, drop_path: float = 0.0, layer_scale_init: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # 7x7 depthwise
        self.norm = LayerNorm2d(dim)
        hidden_dim = int(dim * mlp_ratio)
        # ConvNeXt uses pointwise "MLP" in channels-last space
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones((dim,))) if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)  # LN over channels
        # channels-last MLP
        x = x.permute(0, 2, 3, 1)  # (N,H,W,C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x * self.gamma
        x = x.permute(0, 3, 1, 2)  # (N,C,H,W)
        x = shortcut + self.drop_path(x)
        return x

class ConvNeXtStage(nn.Module):
    """Stacked ConvNeXt blocks at a single resolution."""
    def __init__(self, dim: int, depth: int, mlp_ratio: int = 4, drop_path: float = 0.0):
        super().__init__()
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock(dim=dim, mlp_ratio=mlp_ratio, drop_path=drop_path)
            for _ in range(int(depth))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

class ResidualFuse(nn.Module):
    """Fuse skip + upsample feature with a 1x1 reduction then ConvNeXt stage, with residual."""
    def __init__(self, in_dim: int, out_dim: int, depth: int, mlp_ratio: int = 4):
        super().__init__()
        self.reduce = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.stage = ConvNeXtStage(out_dim, depth=depth, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        return x + self.stage(x)

class UCXUNetDenoise(nn.Module):
    """UCX-style U-Net using ConvNeXt blocks (paper: encoder stage depths 3,3,9,3).
    Notes:
      - Uses SConv (3x3 stride-2 conv) for downsampling.
      - Uses TConv (3x3 transposed conv) for upsampling.
      - Decoder uses ConvNeXt fusing blocks (residual) to mimic 'residual blocks between encoder/decoder' mention.
      - For pure image denoising (e.g., Rician), call forward(x) as usual.
      - For complex MRI pipelines, set in_channels=2, out_channels=2 (real+imag).
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        dims=(64, 128, 256, 512),
        enc_depths=(3, 3, 9, 3),
        dec_depths=None,
        mlp_ratio: int = 4,
        drop_path: float = 0.0,
    ):
        super().__init__()
        assert len(dims) == 4, "dims should be a 4-tuple like (64,128,256,512)"
        assert len(enc_depths) == 4, "enc_depths should be a 4-tuple like (3,3,9,3)"
        if dec_depths is None:
            # symmetric-ish decoder (paper doesn't specify; this is a practical mirror)
            dec_depths = (enc_depths[2], enc_depths[1], enc_depths[0], enc_depths[0])
        self.stem = nn.Conv2d(in_channels, dims[0], kernel_size=3, padding=1)

        # Encoder stages
        self.enc1 = ConvNeXtStage(dims[0], depth=enc_depths[0], mlp_ratio=mlp_ratio, drop_path=drop_path)
        self.down1 = nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=2, padding=1)  # SConv

        self.enc2 = ConvNeXtStage(dims[1], depth=enc_depths[1], mlp_ratio=mlp_ratio, drop_path=drop_path)
        self.down2 = nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=2, padding=1)

        self.enc3 = ConvNeXtStage(dims[2], depth=enc_depths[2], mlp_ratio=mlp_ratio, drop_path=drop_path)
        self.down3 = nn.Conv2d(dims[2], dims[3], kernel_size=3, stride=2, padding=1)

        self.bottleneck = ConvNeXtStage(dims[3], depth=enc_depths[3], mlp_ratio=mlp_ratio, drop_path=drop_path)

        # Decoder
        self.up3 = nn.ConvTranspose2d(dims[3], dims[2], kernel_size=3, stride=2, padding=1, output_padding=1)  # TConv
        self.fuse3 = ResidualFuse(in_dim=dims[2]*2, out_dim=dims[2], depth=dec_depths[0], mlp_ratio=mlp_ratio)

        self.up2 = nn.ConvTranspose2d(dims[2], dims[1], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.fuse2 = ResidualFuse(in_dim=dims[1]*2, out_dim=dims[1], depth=dec_depths[1], mlp_ratio=mlp_ratio)

        self.up1 = nn.ConvTranspose2d(dims[1], dims[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.fuse1 = ResidualFuse(in_dim=dims[0]*2, out_dim=dims[0], depth=dec_depths[2], mlp_ratio=mlp_ratio)

        self.dec0 = ConvNeXtStage(dims[0], depth=dec_depths[3], mlp_ratio=mlp_ratio, drop_path=drop_path)

        self.head = nn.Conv2d(dims[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)

        e1 = self.enc1(x0)
        x = self.down1(e1)

        e2 = self.enc2(x)
        x = self.down2(e2)

        e3 = self.enc3(x)
        x = self.down3(e3)

        x = self.bottleneck(x)

        x = self.up3(x)
        if x.shape[-2:] != e3.shape[-2:]:
            x = F.interpolate(x, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse3(torch.cat([e3, x], dim=1))

        x = self.up2(x)
        if x.shape[-2:] != e2.shape[-2:]:
            x = F.interpolate(x, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse2(torch.cat([e2, x], dim=1))

        x = self.up1(x)
        if x.shape[-2:] != e1.shape[-2:]:
            x = F.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse1(torch.cat([e1, x], dim=1))

        x = self.dec0(x)
        return self.head(x)

class DataConsistencyLayer(nn.Module):
    """Single-coil DC layer (k-space hard replacement at sampled locations).
    Expects complex image represented as 2 channels: [real, imag].
    If y/mask are None, returns the input unchanged (so you can still use this for pure denoising).
    """
    def __init__(self, norm: str = "ortho"):
        super().__init__()
        self.norm = norm

    def forward(self, x_ri: torch.Tensor, y_ri: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if y_ri is None or mask is None:
            return x_ri

        # Convert to complex tensors
        x = torch.complex(x_ri[:, 0], x_ri[:, 1])
        y = torch.complex(y_ri[:, 0], y_ri[:, 1])

        # FFT2 -> replace sampled points -> IFFT2
        k = torch.fft.fft2(x, norm=self.norm)
        mask_c = mask.to(k.dtype)
        # mask expected broadcastable to (N,H,W)
        k = mask_c * y + (1.0 - mask_c) * k
        x_dc = torch.fft.ifft2(k, norm=self.norm)

        out = torch.stack([x_dc.real, x_dc.imag], dim=1)
        return out

class MICXN(nn.Module):
    """MICXN = UCX + DC layer (paper). For denoising-only, you can ignore y/mask.
    Forward signatures:
      - denoising: out = model(x)
      - reconstruction: out = model(zf_img, y_ri, mask)
    """
    def __init__(self, in_channels: int = 2, out_channels: int = 2, dims=(64,128,256,512), enc_depths=(3,3,9,3)):
        super().__init__()
        self.ucx = UCXUNetDenoise(in_channels=in_channels, out_channels=out_channels, dims=dims, enc_depths=enc_depths)
        self.dc = DataConsistencyLayer()

    def forward(self, x: torch.Tensor, y_ri: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.ucx(x)
        x = self.dc(x, y_ri=y_ri, mask=mask)
        return x
