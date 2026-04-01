import argparse


def get_config():
    parser = argparse.ArgumentParser(description="VoxelMorph 3D Training")

    # Required
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to OASIS dataset root directory (must contain train/ and val/ subfolders)",
    )

    # Training
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--lambda_", type=float, default=0.01, help="Smoothness regularization weight"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.01, help="Segmentation loss weight"
    )

    parser.add_argument(
        "--patience", type=int, default=8, help="Early stopping patience (epochs)"
    )
    parser.add_argument(
        "--val_batches",
        type=int,
        default=5,
        help="Number of random batches used for validation each epoch",
    )

    # Output
    parser.add_argument(
        "--out_dir",
        type=str,
        default="trained_models",
        help="Root directory where run folders are saved",
    )

    return parser.parse_args()
