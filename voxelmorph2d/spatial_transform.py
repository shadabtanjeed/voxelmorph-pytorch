import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    """
    2D Spatial Transformer for VoxelMorph.

    Warps a moving image using a dense displacement field (flow).

    Input:
        img:  [B, C, H, W]  — moving image
        flow: [B, 2, H, W]  — displacement field (dx, dy) in pixels
    Output:
              [B, C, H, W]  — warped image
    """

    def __init__(self, mode="bilinear"):
        super().__init__()
        self.mode = mode

    def forward(self, img, flow):
        B, _, H, W = flow.shape  # using 2 directly here causes syntax errors

        # Identity grid
        # basically a reference matrix of size 32x32 where each entry contains its own (x, y) coordinates
        # but why not use a matrix array instead?
        # meshgrid handles dtype, device efficiently
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, dtype=flow.dtype, device=flow.device),
            torch.arange(W, dtype=flow.dtype, device=flow.device),
            indexing="ij",  # it means H varies along rows and W varies along columns
        )

        # [H, W, 2] -> [B, H, W, 2], last dim is (x, y)
        # basically adding a batch dimension and replicating the grid for each item in the batch,
        # so we have a grid of coordinates for each image in the batch
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        # flow: [B, 2, H, W] -> [B, H, W, 2]
        new_locs = grid + flow.permute(0, 2, 3, 1)

        # the new locations are in pixel coordinates,
        # but grid_sample expects normalized coordinates in the range [-1, 1]
        new_locs[..., 0] = 2.0 * new_locs[..., 0] / (W - 1) - 1.0  # x
        new_locs[..., 1] = 2.0 * new_locs[..., 1] / (H - 1) - 1.0  # y

        return F.grid_sample(
            img, new_locs, mode=self.mode, padding_mode="border", align_corners=True
        )
