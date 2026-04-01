import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    """
    3D Spatial Transformer for VoxelMorph.

    Warps a moving image using a dense displacement field (flow).

    Input:
        img:  [B, C, D, H, W]  — moving image
        flow: [B, 3, D, H, W]  — displacement field (dx, dy, dz) in voxels
    Output:
              [B, C, D, H, W]  — warped image
    """

    def __init__(self, mode="trilinear"):
        super().__init__()
        self.mode = mode

    def forward(self, img, flow):
        B, _, D, H, W = flow.shape  # using 3 directly causes issues

        # Identity grid
        # Create coordinate grids for D, H, W dimensions
        grid_z, grid_y, grid_x = torch.meshgrid(
            torch.arange(D, dtype=flow.dtype, device=flow.device),
            torch.arange(H, dtype=flow.dtype, device=flow.device),
            torch.arange(W, dtype=flow.dtype, device=flow.device),
            # not sure why it's ij
            # it makes more sense to have ijk maybe
            indexing="ij",
        )

        # [D, H, W, 3] -> [B, D, H, W, 3], last dim is (x, y, z)
        grid = (
            torch.stack([grid_x, grid_y, grid_z], dim=-1)
            .unsqueeze(0)
            .expand(B, -1, -1, -1, -1)
        )

        # flow: [B, 3, D, H, W] -> [B, D, H, W, 3]
        new_locs = grid + flow.permute(0, 2, 3, 4, 1)

        # the new locations are in pixel coordinates,
        # but grid_sample expects normalized coordinates in the range [-1, 1]        new_locs[..., 0] = 2.0 * new_locs[..., 0] / (W - 1) - 1.0  # x
        new_locs[..., 1] = 2.0 * new_locs[..., 1] / (H - 1) - 1.0  # y
        new_locs[..., 2] = 2.0 * new_locs[..., 2] / (D - 1) - 1.0  # z

        return F.grid_sample(
            img, new_locs, mode=self.mode, padding_mode="border", align_corners=True
        )
