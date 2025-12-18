import torch
import torch.nn as nn
from torchsummary import summary

# ==========================================================
# Generator: Encoder-Decoder with Skip Connections (U-Net style)
# As described in:
# "Boosting MRI Image Denoising With Conditional GANs"
# ==========================================================

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, norm=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if norm:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    """
    Encoder-Decoder Generator
    - 8 encoder layers
    - 8 decoder layers
    - Kernel: 4x4, stride=2
    - Skip connections between symmetric layers
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # ---------------- Encoder ----------------
        self.e1 = ConvBlock(in_channels, 64, norm=False)
        self.e2 = ConvBlock(64, 128)
        self.e3 = ConvBlock(128, 256)
        self.e4 = ConvBlock(256, 512)
        self.e5 = ConvBlock(512, 512)
        self.e6 = ConvBlock(512, 512)
        self.e7 = ConvBlock(512, 512)
        self.e8 = ConvBlock(512, 512)

        # ---------------- Decoder ----------------
        self.d1 = DeconvBlock(512, 512, dropout=True)
        self.d2 = DeconvBlock(1024, 512, dropout=True)
        self.d3 = DeconvBlock(1024, 512, dropout=True)
        self.d4 = DeconvBlock(1024, 512)
        self.d5 = DeconvBlock(1024, 256)
        self.d6 = DeconvBlock(512, 128)
        self.d7 = DeconvBlock(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)

        # Decoder with skip connections
        d1 = self.d1(e8)
        d2 = self.d2(torch.cat([d1, e7], dim=1))
        d3 = self.d3(torch.cat([d2, e6], dim=1))
        d4 = self.d4(torch.cat([d3, e5], dim=1))
        d5 = self.d5(torch.cat([d4, e4], dim=1))
        d6 = self.d6(torch.cat([d5, e3], dim=1))
        d7 = self.d7(torch.cat([d6, e2], dim=1))

        out = self.final(torch.cat([d7, e1], dim=1))
        return out


# ==========================================================
# Discriminator: PatchGAN (Markovian Discriminator)
# ==========================================================

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator
    - Input: concatenated (noisy, clean/denoised)
    - 5 Conv layers, kernel 4x4
    """
    def __init__(self, in_channels=2):
        super().__init__()

        def disc_block(in_c, out_c, stride, norm=True):
            layers = [
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=False)
            ]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            disc_block(in_channels, 64, stride=2, norm=False),
            disc_block(64, 128, stride=2),
            disc_block(128, 256, stride=2),
            disc_block(256, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, noisy, target):
        x = torch.cat([noisy, target], dim=1)
        return self.model(x)


# # ==========================================================
# # Loss Functions
# # ==========================================================

# class GANLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bce = nn.BCELoss()

#     def forward(self, pred, is_real):
#         target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
#         return self.bce(pred, target)


# # ==========================================================
# # Builder functions (for Train.py)
# # ==========================================================

# def build_generator(device):
#     model = Generator()
#     return model.to(device)


# def build_discriminator(device):
#     model = Discriminator()
#     return model.to(device)


# def build_optimizers(G, D, lr=2e-4):
#     opt_G = torch.optim.SGD(G.parameters(), lr=lr, momentum=0.5)
#     opt_D = torch.optim.SGD(D.parameters(), lr=lr, momentum=0.5)
#     return opt_G, opt_D


# # ==========================================================
# # Quick structure check
# # ==========================================================
# class DiscriminatorWrapper(nn.Module):
#     def __init__(self, D):
#         super().__init__()
#         self.D = D

#     def forward(self, x):
#         # x shape: (B, 2, H, W)
#         noisy = x[:, 0:1, :, :]
#         target = x[:, 1:2, :, :]
#         return self.D(noisy, target)


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     G = build_generator(device)
#     D = build_discriminator(device)

#     G.eval()
#     D.eval()

#     with torch.no_grad():
#         x = torch.randn(1, 1, 256, 256).to(device)
#         y = G(x)
#         d_out = D(x, y)

#     print("Generator output:", y.shape)
#     print("Discriminator output:", d_out.shape)

#     print("\n=== Generator Summary ===")
#     summary(G, input_size=(1, 256, 256))

#     print("\n=== Discriminator Summary ===")
#     D_wrap = DiscriminatorWrapper(D)
#     summary(D_wrap, input_size=(2, 256, 256))

