def dice_similarity(fixed_seg, moved_seg, num_labels=35):
    """Average per-label binary Dice over labels 1..num_labels."""
    total = 0.0
    for label in range(1, num_labels + 1):
        f = (fixed_seg == label).float()
        m = (moved_seg == label).float()
        intersection = (f * m).sum()
        total += (2.0 * intersection) / (f.sum() + m.sum() + 1e-6)
    return total / num_labels
