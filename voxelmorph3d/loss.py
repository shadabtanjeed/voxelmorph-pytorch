import torch
import torch.nn.functional as F


def ncc_loss(fixed, moved, win=9):
    """
    Local Normalized Cross-Correlation loss for 3D volumes.

    Args:
        fixed: [B, 1, D, H, W]
        moved: [B, 1, D, H, W]
        win:   local window size (paper uses 9)
    """
    pad = win // 2

    f_mean = F.avg_pool3d(fixed, kernel_size=win, stride=1, padding=pad)
    m_mean = F.avg_pool3d(moved, kernel_size=win, stride=1, padding=pad)

    # Mean-centred patches
    f_c = fixed - f_mean
    m_c = moved - m_mean

    # Local cross-correlation components
    cross = F.avg_pool3d(f_c * m_c, kernel_size=win, stride=1, padding=pad)

    # Normalize by local variance
    f_var = F.avg_pool3d(f_c * f_c, kernel_size=win, stride=1, padding=pad)
    m_var = F.avg_pool3d(m_c * m_c, kernel_size=win, stride=1, padding=pad)

    ncc = (cross * cross) / (f_var * m_var + 1e-5)

    return -ncc.mean()


def smoothness_loss(flow):
    """
    Diffusion regularizer on flow gradients for 3D volumes.

    Args:
        flow: [B, 3, D, H, W]
    """
    dx = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]  # differences along width
    dy = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]  # differences along height
    dz = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]  # differences along depth

    return (dx**2).mean() + (dy**2).mean() + (dz**2).mean()


def dice_loss(fixed_seg, moved_seg, num_labels=35):
    """
    Dice loss for segmentation maps.

    Computes 1 - Dice similarity for each label (1-35) and averages.
    Uses trilinear interpolation to handle floating point segmentation values.

    Args:
        fixed_seg: [B, 1, D, H, W] - fixed segmentation labels
        moved_seg: [B, 1, D, H, W] - warped moving segmentation labels
        num_labels: number of segmentation labels (default 35)
    """
    total_dice = 0.0

    for label in range(1, num_labels + 1):
        # Create binary maps for this label
        fixed_binary = (fixed_seg == label).float()
        moved_binary = (moved_seg == label).float()

        # Compute Dice similarity
        intersection = (fixed_binary * moved_binary).sum()
        dice_coeff = (2.0 * intersection) / (
            fixed_binary.sum() + moved_binary.sum() + 1e-5
        )

        total_dice += dice_coeff

    # Subtract from 1 as we want to minimize the loss (maximize similarity)
    avg_dice = total_dice / num_labels
    return 1.0 - avg_dice


def voxelmorph_loss(fixed, moved, flow, fixed_seg, moved_seg, lambda_=0.01, gamma=0.01):
    sim_loss = ncc_loss(fixed, moved)

    smooth_loss = smoothness_loss(flow)

    seg_loss = dice_loss(fixed_seg, moved_seg)

    return sim_loss + (lambda_ * smooth_loss) + (gamma * seg_loss)
