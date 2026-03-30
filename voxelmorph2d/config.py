import argparse


def get_config():
    parser = argparse.ArgumentParser(description="VoxelMorph 2D Training")

    # Required
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to mnist_normalized_32x32.npz")

    # Training
    parser.add_argument("--num_epochs",  type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--lambda_",     type=float, default=0.01,
                        help="Smoothness regularization weight")
    parser.add_argument("--patience",    type=int,   default=5,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--val_batches", type=int,   default=50,
                        help="Number of random batches used for validation each epoch")

    # Output
    parser.add_argument("--out_dir", type=str, default="trained_models",
                        help="Root directory where run folders are saved")

    return parser.parse_args()
