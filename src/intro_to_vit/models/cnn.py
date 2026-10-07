"""A small ResNet-style CNN used as the convolutional baseline."""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with batch norm and a residual (skip) connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """Create the block; adds a 1x1 projection on the skip path if shapes change."""
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return relu(body(x) + skip(x))."""
        return self.act(self.body(x) + self.skip(x))


class SmallResNet(nn.Module):
    """Three-stage ResNet; each stage halves resolution and widens channels."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        widths: tuple[int, ...] = (32, 64, 128),
    ):
        """Build the CNN. Defaults give a ~0.7M parameter model for CIFAR-10."""
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        stages = []
        prev = widths[0]
        for i, w in enumerate(widths):
            stride = 1 if i == 0 else 2
            stages += [ResidualBlock(prev, w, stride), ResidualBlock(w, w)]
            prev = w
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits of shape (B, num_classes)."""
        x = self.stages(self.stem(x))
        return self.head(self.pool(x).flatten(1))
