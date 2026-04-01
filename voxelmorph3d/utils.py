def dice_similarity(img_fixed, img_moved):

    intersection = (img_fixed * img_moved).sum(dim=(2, 3, 4))
    dice = (2.0 * intersection) / (
        img_fixed.sum(dim=(2, 3, 4)) + img_moved.sum(dim=(2, 3, 4)) + 1e-6
    )
    dice = dice.mean(dim=1)

    return dice.mean()
