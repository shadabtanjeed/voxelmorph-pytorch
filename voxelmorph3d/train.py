import os
import json
import random
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import get_config
from dataset import OASISDataset
from network import VoxelMorphUNet
from spatial_transform import SpatialTransformer
from loss import voxelmorph_loss
from utils import dice_similarity


def run_epoch(loader, model, stn, optimizer, cfg, device, train=True, desc=""):
    model.train(train)
    total_loss = 0.0
    n = 0

    with torch.set_grad_enabled(train):
        for fixed, moving, fixed_seg, moving_seg in tqdm(loader, desc=desc, leave=False):
            fixed      = fixed.to(device)
            moving     = moving.to(device)
            fixed_seg  = fixed_seg.to(device)
            moving_seg = moving_seg.to(device)

            flow  = model(torch.cat([fixed, moving], dim=1))
            moved = stn(moving, flow)

            # Warp segmentation with trilinear interpolation, then round back to integer labels
            moved_seg = stn(moving_seg.float(), flow).round()

            loss = voxelmorph_loss(
                fixed, moved, flow,
                fixed_seg, moved_seg,
                lambda_=cfg.lambda_,
                gamma=cfg.gamma,
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n += 1

    return total_loss / n


def eval_dice(dataset, model, stn, device):
    model.eval()
    total_dice = 0.0

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Dice eval", leave=False):
            fixed, moving, fixed_seg, moving_seg = dataset[idx]
            fixed      = fixed.unsqueeze(0).to(device)
            moving     = moving.unsqueeze(0).to(device)
            moving_seg = moving_seg.unsqueeze(0).to(device)
            fixed_seg  = fixed_seg.unsqueeze(0).to(device)

            flow       = model(torch.cat([fixed, moving], dim=1))
            moved_seg  = stn(moving_seg.float(), flow).round()

            total_dice += dice_similarity(fixed_seg, moved_seg).item()

    return total_dice / len(dataset)


def save_loss_plot(train_losses, val_losses, out_path):
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VoxelMorph 3D Training Loss")
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

    train_dataset = OASISDataset(cfg.data_path, split="train")
    val_dataset   = OASISDataset(cfg.data_path, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_size = cfg.val_batches * cfg.batch_size
    val_indices = random.sample(range(len(val_dataset)), min(val_size, len(val_dataset)))
    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model     = VoxelMorphUNet().to(device)
    stn       = SpatialTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_count = 0
    log = []

    epoch_bar = tqdm(range(1, cfg.num_epochs + 1), desc="Epochs")
    for epoch in epoch_bar:
        train_loss = run_epoch(
            train_loader, model, stn, optimizer, cfg, device,
            train=True, desc=f"Epoch {epoch} train",
        )
        val_loss = run_epoch(
            val_loader, model, stn, optimizer, cfg, device,
            train=False, desc=f"Epoch {epoch} val",
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        entry = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}

        if epoch % 5 == 0:
            dice = eval_dice(val_dataset, model, stn, device)
            entry["dice"] = dice
            epoch_bar.write(
                f"Epoch {epoch:03d}/{cfg.num_epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  dice={dice:.4f}"
            )
        else:
            epoch_bar.write(
                f"Epoch {epoch:03d}/{cfg.num_epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}"
            )

        log.append(entry)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
            epoch_bar.write(f"  → saved best model (val={best_val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                epoch_bar.write(f"Early stopping at epoch {epoch}.")
                break

    torch.save(model.state_dict(), os.path.join(run_dir, "final_model.pth"))

    save_loss_plot(train_losses, val_losses, os.path.join(run_dir, "loss.png"))

    with open(os.path.join(run_dir, "log.json"), "w") as f:
        json.dump({"config": vars(cfg), "log": log}, f, indent=2)

    print(f"\nDone. Artefacts saved to {run_dir}/")
    print(f"  best_model.pth  final_model.pth  loss.png  log.json")


if __name__ == "__main__":
    main()
