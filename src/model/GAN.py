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






# Generator output: torch.Size([1, 1, 256, 256])
# Discriminator output: torch.Size([1, 1, 30, 30])

# === Generator Summary ===
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 64, 128, 128]           1,024
#               ReLU-2         [-1, 64, 128, 128]               0
#          ConvBlock-3         [-1, 64, 128, 128]               0
#             Conv2d-4          [-1, 128, 64, 64]         131,072
#        BatchNorm2d-5          [-1, 128, 64, 64]             256
#               ReLU-6          [-1, 128, 64, 64]               0
#          ConvBlock-7          [-1, 128, 64, 64]               0
#             Conv2d-8          [-1, 256, 32, 32]         524,288
#        BatchNorm2d-9          [-1, 256, 32, 32]             512
#              ReLU-10          [-1, 256, 32, 32]               0
#         ConvBlock-11          [-1, 256, 32, 32]               0
#            Conv2d-12          [-1, 512, 16, 16]       2,097,152
#       BatchNorm2d-13          [-1, 512, 16, 16]           1,024
#              ReLU-14          [-1, 512, 16, 16]               0
#         ConvBlock-15          [-1, 512, 16, 16]               0
#            Conv2d-16            [-1, 512, 8, 8]       4,194,304
#       BatchNorm2d-17            [-1, 512, 8, 8]           1,024
#              ReLU-18            [-1, 512, 8, 8]               0
#         ConvBlock-19            [-1, 512, 8, 8]               0
#            Conv2d-20            [-1, 512, 4, 4]       4,194,304
#       BatchNorm2d-21            [-1, 512, 4, 4]           1,024
#              ReLU-22            [-1, 512, 4, 4]               0
#         ConvBlock-23            [-1, 512, 4, 4]               0
#            Conv2d-24            [-1, 512, 2, 2]       4,194,304
#       BatchNorm2d-25            [-1, 512, 2, 2]           1,024
#              ReLU-26            [-1, 512, 2, 2]               0
#         ConvBlock-27            [-1, 512, 2, 2]               0
#            Conv2d-28            [-1, 512, 1, 1]       4,194,304
#       BatchNorm2d-29            [-1, 512, 1, 1]           1,024
#              ReLU-30            [-1, 512, 1, 1]               0
#         ConvBlock-31            [-1, 512, 1, 1]               0
#   ConvTranspose2d-32            [-1, 512, 2, 2]       4,194,304
#       BatchNorm2d-33            [-1, 512, 2, 2]           1,024
#              ReLU-34            [-1, 512, 2, 2]               0
#           Dropout-35            [-1, 512, 2, 2]               0
#       DeconvBlock-36            [-1, 512, 2, 2]               0
#   ConvTranspose2d-37            [-1, 512, 4, 4]       8,388,608
#       BatchNorm2d-38            [-1, 512, 4, 4]           1,024
#              ReLU-39            [-1, 512, 4, 4]               0
#           Dropout-40            [-1, 512, 4, 4]               0
#       DeconvBlock-41            [-1, 512, 4, 4]               0
#   ConvTranspose2d-42            [-1, 512, 8, 8]       8,388,608
#       BatchNorm2d-43            [-1, 512, 8, 8]           1,024
#              ReLU-44            [-1, 512, 8, 8]               0
#           Dropout-45            [-1, 512, 8, 8]               0
#       DeconvBlock-46            [-1, 512, 8, 8]               0
#   ConvTranspose2d-47          [-1, 512, 16, 16]       8,388,608
#       BatchNorm2d-48          [-1, 512, 16, 16]           1,024
#              ReLU-49          [-1, 512, 16, 16]               0
#       DeconvBlock-50          [-1, 512, 16, 16]               0
#   ConvTranspose2d-51          [-1, 256, 32, 32]       4,194,304
#       BatchNorm2d-52          [-1, 256, 32, 32]             512
#              ReLU-53          [-1, 256, 32, 32]               0
#       DeconvBlock-54          [-1, 256, 32, 32]               0
#   ConvTranspose2d-55          [-1, 128, 64, 64]       1,048,576
#       BatchNorm2d-56          [-1, 128, 64, 64]             256
#              ReLU-57          [-1, 128, 64, 64]               0
#       DeconvBlock-58          [-1, 128, 64, 64]               0
#   ConvTranspose2d-59         [-1, 64, 128, 128]         262,144
#       BatchNorm2d-60         [-1, 64, 128, 128]             128
#              ReLU-61         [-1, 64, 128, 128]               0
#       DeconvBlock-62         [-1, 64, 128, 128]               0
#   ConvTranspose2d-63          [-1, 1, 256, 256]           2,049
#              Tanh-64          [-1, 1, 256, 256]               0
# ================================================================
# Total params: 54,408,833
# Trainable params: 54,408,833
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.25
# Forward/backward pass size (MB): 115.97
# Params size (MB): 207.55
# Estimated Total Size (MB): 323.77
# ----------------------------------------------------------------

# === Discriminator Summary ===
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 64, 128, 128]           2,048
#               ReLU-2         [-1, 64, 128, 128]               0
#             Conv2d-3          [-1, 128, 64, 64]         131,072
#        BatchNorm2d-4          [-1, 128, 64, 64]             256
#               ReLU-5          [-1, 128, 64, 64]               0
#             Conv2d-6          [-1, 256, 32, 32]         524,288
#        BatchNorm2d-7          [-1, 256, 32, 32]             512
#               ReLU-8          [-1, 256, 32, 32]               0
#             Conv2d-9          [-1, 512, 31, 31]       2,097,152
#       BatchNorm2d-10          [-1, 512, 31, 31]           1,024
#              ReLU-11          [-1, 512, 31, 31]               0
#            Conv2d-12            [-1, 1, 30, 30]           8,193
#           Sigmoid-13            [-1, 1, 30, 30]               0
#     Discriminator-14            [-1, 1, 30, 30]               0
# ================================================================
# Total params: 2,764,545
# Trainable params: 2,764,545
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.50
# Forward/backward pass size (MB): 45.28
# Params size (MB): 10.55
# Estimated Total Size (MB): 56.33
# ----------------------------------------------------------------
