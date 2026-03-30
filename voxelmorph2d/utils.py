def jaccard_index(seg_fixed, seg_moved, threshold=0.5):
    """
    Jaccard Index or Intersection over Union between two segmentation maps.

    Args:
        seg_fixed: [B, 1, H, W]  — fixed image segmentation (ground truth)
        seg_moved: [B, 1, H, W]  — warped moving image segmentation
        threshold
    Returns:
        mean Jaccard index across the batch (scalar)
    """

    f = (seg_fixed > threshold).float()
    m = (seg_moved > threshold).float()

    intersection = (f * m).sum(dim=(1, 2, 3))
    union = (f + m).clamp(max=1).sum(dim=(1, 2, 3))

    iou = intersection / (union + 1e-6)  # add small epsilon to avoid division by zero

    return iou.mean()
