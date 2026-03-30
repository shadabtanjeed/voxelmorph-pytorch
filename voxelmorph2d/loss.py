import torch.nn.functional as F


def ncc_loss(fixed, moved, win=9):
    """
    Local Normalized Cross-Correlation loss.

    Args:
        fixed: [B, 1, H, W]
        moved: [B, 1, H, W]
        win:   local window size (paper uses 9)
    """
    pad = win // 2

    f_mean = F.avg_pool2d(fixed, kernel_size=win, stride=1, padding=pad)
    m_mean = F.avg_pool2d(moved, kernel_size=win, stride=1, padding=pad)

    # Mean-centred patches
    f_c = fixed - f_mean
    m_c = moved - m_mean

    # Local cross-correlation components
    cross = F.avg_pool2d(
        f_c * m_c, kernel_size=win, stride=1, padding=pad
    )  # finds if the local window has similar patterns in fixed and moved

    # we want to match shape, not intensity
    # so we normalize by local variance to prevent the network from just matching brightness
    f_var = F.avg_pool2d(f_c * f_c, kernel_size=win, stride=1, padding=pad)
    m_var = F.avg_pool2d(m_c * m_c, kernel_size=win, stride=1, padding=pad)

    ncc = (cross * cross) / (f_var * m_var + 1e-5)

    return -ncc.mean()  # we want to maximize NCC, so return negative for minimization


def smoothness_loss(flow):
    """
    Diffusion regularizer on flow gradients.

    Args:
        flow: [B, 2, H, W]
    """
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]  # horizontal differences
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]  # vertical differences

    return (dx**2).mean() + (dy**2).mean()


def voxelmorph_loss(fixed, moved, flow, lambda_=0.01):
    return ncc_loss(fixed, moved) + lambda_ * smoothness_loss(flow)
