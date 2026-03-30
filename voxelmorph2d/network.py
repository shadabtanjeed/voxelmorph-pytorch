import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxelMorphUNet(nn.Module):

    # Modified UNet for VoxelMorph 2D registration on 32x32 inputs.

    # Input:  [B, 2, 32, 32]  — channel 0: fixed image, channel 1: moving image
    # Output: [B, 2, 32, 32]  — displacement field (dx, dy) per pixel: Flow

    # Modified to use 3 levels of downsampling
    def __init__(self):
        super().__init__()

        # 32×32 → 16×16
        self.enc1_conv = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.enc1_down = nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=2)

        # 16×16 → 8×8
        self.enc2_conv = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.enc2_down = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2)

        # 8×8 → 4×4
        self.enc3_conv = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enc3_down = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2)

        # Bottleneck
        self.bottleneck_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bottleneck_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # 4×4 → 8×8
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3_conv1 = nn.Conv2d(
            128, 64, kernel_size=3, padding=1
        )  # 64 up + 64 skip
        self.dec3_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # 8×8 → 16×16
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  # 32 up + 32 skip
        self.dec2_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        # 16×16 → 32×32  (skip from enc1_conv: 16ch)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1_conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)  # 16 up + 16 skip
        self.dec1_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        # 2-channel displacement field
        self.flow_head = nn.Conv2d(16, 2, kernel_size=3, padding=1)

        # Small init so the starting warp is close to an identity transform
        nn.init.normal_(self.flow_head.weight, mean=0.0, std=1e-5)
        if self.flow_head.bias is not None:
            nn.init.zeros_(self.flow_head.bias)

    def forward(self, x):
        # x: [B, 2, 32, 32]  (fixed image stacked with moving image)

        # Encoder
        skip1 = F.leaky_relu(
            self.enc1_conv(x), 0.2
        )  # [B, 16, 32, 32]  ← saved for skip
        feat1 = F.leaky_relu(self.enc1_down(skip1), 0.2)  # [B, 16, 16, 16]

        skip2 = F.leaky_relu(
            self.enc2_conv(feat1), 0.2
        )  # [B, 32, 16, 16]  ← saved for skip
        feat2 = F.leaky_relu(self.enc2_down(skip2), 0.2)  # [B, 32,  8,  8]

        skip3 = F.leaky_relu(
            self.enc3_conv(feat2), 0.2
        )  # [B, 64,  8,  8]  ← saved for skip
        feat3 = F.leaky_relu(self.enc3_down(skip3), 0.2)  # [B, 64,  4,  4]

        # Bottleneck
        feat_bn = F.leaky_relu(self.bottleneck_conv1(feat3), 0.2)  # [B, 128, 4, 4]
        feat_bn = F.leaky_relu(self.bottleneck_conv2(feat_bn), 0.2)  # [B, 128, 4, 4]

        # Decoder
        up3 = self.up3(feat_bn)  # [B,  64,  8,  8]
        dec3 = F.leaky_relu(
            self.dec3_conv1(torch.cat([up3, skip3], dim=1)), 0.2
        )  # [B, 64, 8, 8]
        dec3 = F.leaky_relu(self.dec3_conv2(dec3), 0.2)  # [B, 64, 8, 8]

        up2 = self.up2(dec3)  # [B,  32, 16, 16]
        dec2 = F.leaky_relu(
            self.dec2_conv1(torch.cat([up2, skip2], dim=1)), 0.2
        )  # [B, 32, 16, 16]
        dec2 = F.leaky_relu(self.dec2_conv2(dec2), 0.2)  # [B, 32, 16, 16]

        up1 = self.up1(dec2)  # [B,  16, 32, 32]
        dec1 = F.leaky_relu(
            self.dec1_conv1(torch.cat([up1, skip1], dim=1)), 0.2
        )  # [B, 16, 32, 32]
        dec1 = F.leaky_relu(self.dec1_conv2(dec1), 0.2)  # [B, 16, 32, 32]

        # Displacement field
        flow = self.flow_head(dec1)  # [B,  2, 32, 32]

        return flow
