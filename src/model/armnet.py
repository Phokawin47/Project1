from __future__ import annotations
import torch
import torch.nn as nn
# from ..registry import MODELS
try:
    from ..registry import MODELS
except ImportError:
    # for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.registry import MODELS


class ARMInception(nn.Module):
    def __init__(self, channels):
        super().__init__()
        c = channels // 4

        self.b1 = nn.Sequential(
            nn.Conv2d(channels, c, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(c, c, kernel_size=(3, 1), padding=(1, 0), dilation=1),
            nn.Conv2d(c, c, kernel_size=(1, 3), padding=(0, 1), dilation=1),
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(channels, c, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(c, c, kernel_size=(3, 1), padding=(2, 0), dilation=2),
            nn.Conv2d(c, c, kernel_size=(1, 3), padding=(0, 2), dilation=2),
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(channels, c, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(c, c, kernel_size=(3, 1), padding=(3, 0), dilation=3),
            nn.Conv2d(c, c, kernel_size=(1, 3), padding=(0, 3), dilation=3),
        )

        self.b4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(channels, c, 1),
        )

        self.fuse = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        x = torch.cat(
            [self.b1(x), self.b2(x), self.b3(x), self.b4(x)],
            dim=1,
        )
        return self.fuse(x)
    
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(
            self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        )


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ARM_Net_Inception_Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.inception = ARMInception(channels)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()
        self.cbam = CBAM(channels)

    def forward(self, x):
        r = x
        x = self.inception(x)
        x = x.permute(0,2,3,1)
        x = self.norm(x)
        x = x.permute(0,3,1,2)
        x = self.act(x)
        x = self.cbam(x)
        return x + r


class ARMNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        feat_channels: int = 64,
        num_layers: int = 6,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 7, padding=3),
            nn.GELU(),
            nn.Conv2d(feat_channels, feat_channels, 5, padding=2),
            nn.GELU(),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.GELU()
        )


        self.body = nn.Sequential(
            *[ARM_Net_Inception_Block(feat_channels) for _ in range(num_layers)]
        )

        self.tail = nn.Conv2d(feat_channels, feat_channels, 1)


    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        return self.tail(x)

class AdaptiveLocalFilter(nn.Module):
    """
    Placeholder for Eq.(4) in the paper.
    """
    def __init__(self):
        super().__init__()
        self.blur = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        return self.blur(x)

class MaskCNN(nn.Module):
    def __init__(self, image_channels: int = 1):
        super().__init__()

        self.adp_filter = AdaptiveLocalFilter()
        self.arm = ARMNet(
            image_channels,
            num_layers=9,   
        )

        self.out = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_f = self.adp_filter(x)
        feat = self.arm(x_f)
        return self.out(feat)

class ADPBranch(nn.Module):
    def __init__(self, image_channels: int = 1):
        super().__init__()

        self.adp_filter = AdaptiveLocalFilter()

        # Three parallel ARM-Nets (num_layers=6)
        self.arm_filtered = ARMNet(image_channels, num_layers=6)
        self.arm_original = ARMNet(image_channels, num_layers=6)
        self.arm_masked   = ARMNet(image_channels, num_layers=6)

        # Fusion
        self.arm_fusion = ARMNet(64 * 2, num_layers=5)

        # Residual & attention (separate ARM-Nets)
        self.arm_attention = ARMNet(64, num_layers=5)

        self.residual = nn.Conv2d(64, image_channels, 3, padding=1)
        self.attention = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, x_masked):
        x_adp = self.adp_filter(x_masked)

        f1 = self.arm_filtered(x_adp)
        f2 = self.arm_original(x)

        x_c = torch.cat([f1, f2], dim=1)

        x_fused = self.arm_fusion(x_c)

        r_adp = x_fused
        w_adp = self.attention(x_fused)

        i_adp = x_adp + r_adp

        return i_adp, w_adp

class OptimizableDCT(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        lp = x * self.alpha
        hp = x * (1 - self.alpha)
        return lp, hp

class DCTBranch(nn.Module):
    def __init__(self, image_channels: int = 1):
        super().__init__()

        self.dct = OptimizableDCT()

        self.arm_lp       = ARMNet(image_channels, num_layers=6)
        self.arm_hp       = ARMNet(image_channels, num_layers=6)
        self.arm_original = ARMNet(image_channels, num_layers=6)

        self.arm_fusion = ARMNet(64 * 3, num_layers=5)
        self.arm_attention = ARMNet(64, num_layers=5)

        self.residual = nn.Conv2d(64, image_channels, 3, padding=1)
        self.attention = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, x_masked):
        lp, hp = self.dct(x_masked)

        f1 = self.arm_lp(lp)
        f2 = self.arm_hp(hp)
        f3 = self.arm_original(x)

        x_c = torch.cat([f1, f2, f3], dim=1)

        fused = self.arm_fusion(x_c)

        r_dct = fused

        w_dct = self.arm_attention(fused)

        i_dct = lp + r_dct
        return i_dct, w_dct


@MODELS.register("armnet_mri_denoiser_paper")
class ARMNetMRIDenoiserPaper(nn.Module):
    def __init__(self, image_channels: int = 1):
        super().__init__()

        self.mask_cnn = MaskCNN(image_channels)
        self.adp_branch = ADPBranch(image_channels)
        self.dct_branch = DCTBranch(image_channels)

    def forward(self, x):
        # Mask CNN
        mask = self.mask_cnn(x)
        x_masked = x * mask

        # ADP branch
        i_adp, w_adp = self.adp_branch(x, x_masked)

        # DCT branch
        i_dct, w_dct = self.dct_branch(x, x_masked)

        # Final fusion
        out = (w_adp * i_adp) + (w_dct * i_dct)
        return out
    
# if __name__ == "__main__":
#     import torch
#     from torchsummary import summary

#     # -----------------------------
#     # Model config
#     # -----------------------------
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = ARMNetMRIDenoiserPaper(
#         image_channels=1
#     ).to(device)

#     # -----------------------------
#     # Print high-level structure
#     # -----------------------------
#     print("\n===== MODEL STRUCTURE =====\n")
#     print(model)

#     # -----------------------------
#     # Torchsummary (layer-by-layer)
#     # -----------------------------
#     print("\n===== TORCHSUMMARY =====\n")
#     summary(
#         model,
#         input_size=(1, 256, 256),  # (C, H, W)
#         device=str(device),
#     )

#     # -----------------------------
#     # Optional: forward test
#     # -----------------------------
#     x = torch.randn(1, 1, 256, 256).to(device)
#     y = model(x)

#     print("\n===== FORWARD CHECK =====")
#     print(f"Input shape : {x.shape}")
#     print(f"Output shape: {y.shape}")
