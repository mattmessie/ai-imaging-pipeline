"""Task 3: U-Net architecture.

Matches src from the course's LAB_CNN_unet_segmentation.ipynb exactly (same
DoubleConv block, same 3-level encoder + bottleneck structure, same
base=16 channel count, same ConvTranspose2d upsampling) -- deliberately
not altered, so this is genuinely "the provided small U-Net" the
assignment brief refers to, just adapted to load real nuclei images
instead of the lab's synthetic on-the-fly generator (see train_unet.py).
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two 3x3 convolutions with batch norm and ReLU, the basic U-Net block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """3-level encoder + bottleneck + 3-level decoder, base=16 channels
    (16 -> 32 -> 64 -> 128 at the bottleneck). Fully convolutional, so it
    works at any input resolution divisible by 8 (256x256 here, vs. the
    lab's 128x128 synthetic images -- no architecture change needed).
    """

    def __init__(self, in_ch=1, out_ch=1, base=16):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        # Bottleneck
        self.bottleneck = DoubleConv(base * 4, base * 8)
        # Decoder
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)  # base*8 because of skip concat
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        # Output
        self.out_conv = nn.Conv2d(base, out_ch, kernel_size=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder path (keep features for skip connections)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
        # Decoder path with skip connections
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))  # skip from e3
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # skip from e2
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # skip from e1
        return self.out_conv(d1)  # raw logits
