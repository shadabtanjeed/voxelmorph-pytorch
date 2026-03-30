import os
import json
import random
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from config import get_config
from dataset import MNISTDataset
from network import VoxelMorphUNet
from spatial_transform import SpatialTransformer
from loss import voxelmorph_loss


def run_epoch(loader, model, stn, optimizer, lambda_, device, train=True):
    model.train(train)
    total_loss = 0.0
    n = 0

    with torch.set_grad_enabled(train):
        for fixed, moving in loader:
            fixed = fixed.to(device)
            moving = moving.to(device)

            flow = model(torch.cat([fixed, moving], dim=1))
            moved = stn(moving, flow)
            loss = voxelmorph_loss(fixed, moved, flow, lambda_)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n += 1

    return total_loss / n


def save_loss_plot(train_losses, val_losses, out_path):
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VoxelMorph Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving run to: {run_dir}")

    train_dataset = MNISTDataset(cfg.data_path, split="train")
    test_dataset = MNISTDataset(cfg.data_path, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_size = cfg.val_batches * cfg.batch_size
    val_indices = random.sample(
        range(len(test_dataset)), min(val_size, len(test_dataset))
    )
    val_loader = DataLoader(
        Subset(test_dataset, val_indices),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = VoxelMorphUNet().to(device)
    stn = SpatialTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_count = 0
    log = []

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = run_epoch(
            train_loader, model, stn, optimizer, cfg.lambda_, device, train=True
        )
        val_loss = run_epoch(
            val_loader, model, stn, optimizer, cfg.lambda_, device, train=False
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        entry = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        log.append(entry)
        print(
            f"Epoch {epoch:03d}/{cfg.num_epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
            print(f"  → saved best model (val={best_val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    torch.save(model.state_dict(), os.path.join(run_dir, "final_model.pth"))

    # Loss plot
    save_loss_plot(train_losses, val_losses, os.path.join(run_dir, "loss.png"))

    # Training log
    with open(os.path.join(run_dir, "log.json"), "w") as f:
        json.dump({"config": vars(cfg), "log": log}, f, indent=2)

    print(f"\nDone. Artefacts saved to {run_dir}/")
    print(f"  best_model.pth  final_model.pth  loss.png  log.json")


if __name__ == "__main__":
    main()
