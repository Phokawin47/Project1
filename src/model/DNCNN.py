import torch
import torch.nn as nn
import torch
from torchsummary import summary


class DnCNN(nn.Module):
    """
    DnCNN model for grayscale image denoising (288x288)
    Architecture:
    - Input: 1-channel grayscale image
    - 20 convolutional layers, 64 filters, 3x3 kernel, stride=1, padding=1
    - BatchNorm + ReLU after each conv (except last layer has no BN/ReLU)
    - Output: residual (noise) prediction
    """
    def __init__(self, depth: int = 20, num_channels: int = 64, image_channels: int = 1):
        super(DnCNN, self).__init__()

        layers = []

        # -------- First layer (Conv + ReLU) --------
        layers.append(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            )
        )
        layers.append(nn.ReLU(inplace=True))

        # -------- Middle layers (Conv + BN + ReLU) --------
        for _ in range(depth):
            layers.append(
                nn.Conv2d(
                    in_channels=num_channels,
                    out_channels=num_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                )
            )
            layers.append(nn.BatchNorm2d(num_channels))
            layers.append(nn.ReLU(inplace=True))

        # -------- Last layer (Conv only) --------
        layers.append(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=image_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )
        )

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass
        DnCNN learns residual (noise), so output = x - predicted_noise
        """
        noise = self.dncnn(x)
        return x - noise


# ------------------------------------------------------------------
# ส่วนนี้เป็นพวกขั้นตอนขอกการฝึกตามวิจัย ถ้าจะใช้ค่อยว่ากันอีกที
# ------------------------------------------------------------------

# def build_loss():
#     """MSE loss for denoising"""
#     return nn.MSELoss()


# def build_optimizer(model, lr=1e-3):
#     """
#     SGD optimizer configuration (as specified)
#     - momentum = 0.9
#     - weight_decay = 1e-4
#     """
#     optimizer = torch.optim.SGD(
#         model.parameters(),
#         lr=lr,
#         momentum=0.9,
#         weight_decay=1e-4
#     )
#     return optimizer


# def build_scheduler(optimizer):
#     """
#     Learning rate scheduler:
#     - lr starts at 0.001
#     - divide by 10 after epoch 10 and 20
#     """
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(
#         optimizer,
#         milestones=[10, 20],
#         gamma=0.1
#     )
#     return scheduler


# ------------------------------------------------------------------
# ส่วนทดสอบการใช้งานและดูโครงสร้าง
# ------------------------------------------------------------------
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = DnCNN(
#         depth=20,
#         num_channels=64,
#         image_channels=1
#     )
#     model.to(device)

#     # Dummy grayscale input (batch_size=1, 1 channel, 288x288)
#     x = torch.randn(1, 1, 288, 288).to(device)
#     y = model(x)

#     print("Input shape :", x.shape)
#     print("Output shape:", y.shape)
    

#     summary(model, input_size=(1, 288, 288))





# Input shape : torch.Size([1, 1, 288, 288])
# Output shape: torch.Size([1, 1, 288, 288])
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 64, 288, 288]             640
#               ReLU-2         [-1, 64, 288, 288]               0
#             Conv2d-3         [-1, 64, 288, 288]          36,864
#        BatchNorm2d-4         [-1, 64, 288, 288]             128
#               ReLU-5         [-1, 64, 288, 288]               0
#             Conv2d-6         [-1, 64, 288, 288]          36,864
#        BatchNorm2d-7         [-1, 64, 288, 288]             128
#               ReLU-8         [-1, 64, 288, 288]               0
#             Conv2d-9         [-1, 64, 288, 288]          36,864
#       BatchNorm2d-10         [-1, 64, 288, 288]             128
#              ReLU-11         [-1, 64, 288, 288]               0
#            Conv2d-12         [-1, 64, 288, 288]          36,864
#       BatchNorm2d-13         [-1, 64, 288, 288]             128
#              ReLU-14         [-1, 64, 288, 288]               0
#            Conv2d-15         [-1, 64, 288, 288]          36,864
#       BatchNorm2d-16         [-1, 64, 288, 288]             128
#              ReLU-17         [-1, 64, 288, 288]               0
#            Conv2d-18         [-1, 64, 288, 288]          36,864
#       BatchNorm2d-19         [-1, 64, 288, 288]             128
#              ReLU-20         [-1, 64, 288, 288]               0
#            Conv2d-21         [-1, 64, 288, 288]          36,864
#       BatchNorm2d-22         [-1, 64, 288, 288]             128
#              ReLU-23         [-1, 64, 288, 288]               0
#            Conv2d-24         [-1, 64, 288, 288]          36,864
#       BatchNorm2d-25         [-1, 64, 288, 288]             128
#              ReLU-26         [-1, 64, 288, 288]               0
#            Conv2d-27         [-1, 64, 288, 288]          36,864
#       BatchNorm2d-28         [-1, 64, 288, 288]             128
#              ReLU-29         [-1, 64, 288, 288]               0
#            Conv2d-30         [-1, 64, 288, 288]          36,864
#       BatchNorm2d-31         [-1, 64, 288, 288]             128
#              ReLU-32         [-1, 64, 288, 288]               0
#            Conv2d-33         [-1, 64, 288, 288]          36,864
#       BatchNorm2d-34         [-1, 64, 288, 288]             128
#              ReLU-35         [-1, 64, 288, 288]               0
#            Conv2d-36         [-1, 64, 288, 288]          36,864
#       BatchNorm2d-37         [-1, 64, 288, 288]             128
#              ReLU-38         [-1, 64, 288, 288]               0
#            Conv2d-39         [-1, 64, 288, 288]          36,864
#       BatchNorm2d-40         [-1, 64, 288, 288]             128
#              ReLU-41         [-1, 64, 288, 288]               0
#            Conv2d-42         [-1, 64, 288, 288]          36,864
#       BatchNorm2d-43         [-1, 64, 288, 288]             128
#              ReLU-44         [-1, 64, 288, 288]               0
#            Conv2d-45         [-1, 64, 288, 288]          36,864
#       BatchNorm2d-46         [-1, 64, 288, 288]             128
#              ReLU-47         [-1, 64, 288, 288]               0
#            Conv2d-48         [-1, 64, 288, 288]          36,864
#       BatchNorm2d-49         [-1, 64, 288, 288]             128
#              ReLU-50         [-1, 64, 288, 288]               0
#            Conv2d-51         [-1, 64, 288, 288]          36,864
#       BatchNorm2d-52         [-1, 64, 288, 288]             128
#              ReLU-53         [-1, 64, 288, 288]               0
#            Conv2d-54         [-1, 64, 288, 288]          36,864
#       BatchNorm2d-55         [-1, 64, 288, 288]             128
#              ReLU-56         [-1, 64, 288, 288]               0
#            Conv2d-57         [-1, 64, 288, 288]          36,864
#       BatchNorm2d-58         [-1, 64, 288, 288]             128
#              ReLU-59         [-1, 64, 288, 288]               0
#            Conv2d-60         [-1, 64, 288, 288]          36,864
#       BatchNorm2d-61         [-1, 64, 288, 288]             128
#              ReLU-62         [-1, 64, 288, 288]               0
#            Conv2d-63          [-1, 1, 288, 288]             576
# ================================================================
# Total params: 741,056
# Trainable params: 741,056
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.32
# Forward/backward pass size (MB): 2511.63
# Params size (MB): 2.83
# Estimated Total Size (MB): 2514.78
# ----------------------------------------------------------------
