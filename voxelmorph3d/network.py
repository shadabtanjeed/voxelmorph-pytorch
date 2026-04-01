import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxelMorphUNet(nn.Module):

    # Modified UNet for VoxelMorph 3D registration on 160x192x224 inputs.

    # Input:  [B, 2, 160, 192, 224]  — channel 0: fixed image, channel 1: moving image
    # Output: [B, 3, 160, 192, 224]  — displacement field (dx, dy, dz) per voxel: Flow

    # 4 levels of downsampling and upsampling
    def __init__(self):
        super().__init__()

        # 160×192×224 → 80×96×112
        self.enc1_conv = nn.Conv3d(2, 32, kernel_size=3, padding=1)
        self.enc1_down = nn.Conv3d(32, 32, kernel_size=3, padding=1, stride=2)

        # 80×96×112 → 40×48×56
        self.enc2_conv = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.enc2_down = nn.Conv3d(64, 64, kernel_size=3, padding=1, stride=2)

        # 40×48×56 → 20×24×28
        self.enc3_conv = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.enc3_down = nn.Conv3d(128, 128, kernel_size=3, padding=1, stride=2)

        # 20×24×28 → 10×12×14
        self.enc4_conv = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.enc4_down = nn.Conv3d(256, 256, kernel_size=3, padding=1, stride=2)

        # Bottleneck
        self.bottleneck_conv1 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.bottleneck_conv2 = nn.Conv3d(512, 512, kernel_size=3, padding=1)

        # 10×12×14 → 20×24×28
        self.up4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec4_conv1 = nn.Conv3d(
            512, 256, kernel_size=3, padding=1
        )  # 256 up + 256 skip
        self.dec4_conv2 = nn.Conv3d(256, 256, kernel_size=3, padding=1)

        # 20×24×28 → 40×48×56
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3_conv1 = nn.Conv3d(
            256, 128, kernel_size=3, padding=1
        )  # 128 up + 128 skip
        self.dec3_conv2 = nn.Conv3d(128, 128, kernel_size=3, padding=1)

        # 40×48×56 → 80×96×112
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2_conv1 = nn.Conv3d(128, 64, kernel_size=3, padding=1)  # 64 up + 64 skip
        self.dec2_conv2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        # 80×96×112 → 160×192×224
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1_conv1 = nn.Conv3d(64, 32, kernel_size=3, padding=1)  # 32 up + 32 skip
        self.dec1_conv2 = nn.Conv3d(32, 32, kernel_size=3, padding=1)

        # 3-channel displacement field (dx, dy, dz)
        self.flow_head = nn.Conv3d(32, 3, kernel_size=3, padding=1)

        # Small init so the starting warp is close to an identity transform
        nn.init.normal_(self.flow_head.weight, mean=0.0, std=1e-5)
        if self.flow_head.bias is not None:
            nn.init.zeros_(self.flow_head.bias)

    def forward(self, x):
        # x: [B, 2, 160, 192, 224]  (fixed image stacked with moving image)

        # Encoder
        skip1 = F.leaky_relu(self.enc1_conv(x), 0.2)  # [B, 32, 160, 192, 224]
        feat1 = F.leaky_relu(self.enc1_down(skip1), 0.2)  # [B, 32, 80, 96, 112]

        skip2 = F.leaky_relu(self.enc2_conv(feat1), 0.2)  # [B, 64, 80, 96, 112]
        feat2 = F.leaky_relu(self.enc2_down(skip2), 0.2)  # [B, 64, 40, 48, 56]

        skip3 = F.leaky_relu(self.enc3_conv(feat2), 0.2)  # [B, 128, 40, 48, 56]
        feat3 = F.leaky_relu(self.enc3_down(skip3), 0.2)  # [B, 128, 20, 24, 28]

        skip4 = F.leaky_relu(self.enc4_conv(feat3), 0.2)  # [B, 256, 20, 24, 28]
        feat4 = F.leaky_relu(self.enc4_down(skip4), 0.2)  # [B, 256, 10, 12, 14]

        # Bottleneck
        feat_bn = F.leaky_relu(self.bottleneck_conv1(feat4), 0.2)  # [B, 512, 10, 12, 14]
        feat_bn = F.leaky_relu(self.bottleneck_conv2(feat_bn), 0.2)  # [B, 512, 10, 12, 14]

        # Decoder
        up4 = self.up4(feat_bn)  # [B, 256, 20, 24, 28]
        dec4 = F.leaky_relu(
            self.dec4_conv1(torch.cat([up4, skip4], dim=1)), 0.2
        )  # [B, 256, 20, 24, 28]
        dec4 = F.leaky_relu(self.dec4_conv2(dec4), 0.2)  # [B, 256, 20, 24, 28]

        up3 = self.up3(dec4)  # [B, 128, 40, 48, 56]
        dec3 = F.leaky_relu(
            self.dec3_conv1(torch.cat([up3, skip3], dim=1)), 0.2
        )  # [B, 128, 40, 48, 56]
        dec3 = F.leaky_relu(self.dec3_conv2(dec3), 0.2)  # [B, 128, 40, 48, 56]

        up2 = self.up2(dec3)  # [B, 64, 80, 96, 112]
        dec2 = F.leaky_relu(
            self.dec2_conv1(torch.cat([up2, skip2], dim=1)), 0.2
        )  # [B, 64, 80, 96, 112]
        dec2 = F.leaky_relu(self.dec2_conv2(dec2), 0.2)  # [B, 64, 80, 96, 112]

        up1 = self.up1(dec2)  # [B, 32, 160, 192, 224]
        dec1 = F.leaky_relu(
            self.dec1_conv1(torch.cat([up1, skip1], dim=1)), 0.2
        )  # [B, 32, 160, 192, 224]
        dec1 = F.leaky_relu(self.dec1_conv2(dec1), 0.2)  # [B, 32, 160, 192, 224]

        # Displacement field
        flow = self.flow_head(dec1)  # [B, 3, 160, 192, 224]

        return flow
