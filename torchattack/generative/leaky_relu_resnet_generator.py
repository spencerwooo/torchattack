import torch
import torch.nn as nn
import torch.nn.functional as f

# To control feature map in generator
ngf = 64


def fused_leaky_relu(
    input: torch.Tensor,
    bias: torch.Tensor,
    negative_slope: float = 0.2,
    scale: float = 2**0.5,
) -> torch.Tensor:
    return f.leaky_relu(input + bias, negative_slope) * scale


class FusedLeakyReLU(nn.Module):
    def __init__(
        self, channel: int, negative_slope: float = 0.2, scale: float = 2**0.5
    ) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)
        return out


class ResNetGenerator(nn.Module):
    def __init__(self, inception: bool = False) -> None:
        """Generator network (ResNet).

        Args:
            inception: if True crop layer will be added to go from 3x300x300 to
            3x299x299. Defaults to False.
        """

        super(ResNetGenerator, self).__init__()

        # Input_size = 3, n, n
        self.inception = inception
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            FusedLeakyReLU(ngf),
        )

        # Input size = 3, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            FusedLeakyReLU(ngf * 2),
        )

        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            FusedLeakyReLU(ngf * 4),
        )

        # Input size = 3, n/4, n/4
        # Residual Blocks: 6
        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)
        self.resblock3 = ResidualBlock(ngf * 4)
        self.resblock4 = ResidualBlock(ngf * 4)
        self.resblock5 = ResidualBlock(ngf * 4)
        self.resblock6 = ResidualBlock(ngf * 4)

        # Input size = 3, n/4, n/4
        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(
                ngf * 4,
                ngf * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ngf * 2),
            FusedLeakyReLU(ngf * 2),
        )

        # Input size = 3, n/2, n/2
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(
                ngf * 2,
                ngf,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ngf),
            FusedLeakyReLU(ngf),
        )

        # Input size = 3, n, n
        self.blockf = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0)
        )

        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.upsampl1(x)
        x = self.upsampl2(x)
        x = self.blockf(x)
        if self.inception:
            x = self.crop(x)
        return (torch.tanh(x) + 1) / 2  # Output range [0 1]


class ResidualBlock(nn.Module):
    def __init__(self, num_filters: int) -> None:
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(num_filters),
            FusedLeakyReLU(num_filters),
            nn.Dropout(0.5),
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(num_filters),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual: torch.Tensor = self.block(x)
        return x + residual
