import torch
from torch.utils.data import Dataset
import numpy as np


class MNISTDataset(Dataset):
    def __init__(self, npz_path, split="train"):
        data = np.load(npz_path)
        if split == "train":
            self.images = data["train_images"]
            self.labels = data["train_labels"]
        else:
            self.images = data["test_images"]
            self.labels = data["test_labels"]

        self.class_indices = {
            c: np.where(self.labels == c)[0] for c in np.unique(self.labels)
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fixed_image = self.images[idx]
        label = self.labels[idx]

        moving_idx = np.random.choice(self.class_indices[label])
        moving_image = self.images[moving_idx]

        fixed_image = torch.from_numpy(fixed_image).unsqueeze(0).float()
        moving_image = torch.from_numpy(moving_image).unsqueeze(0).float()

        return fixed_image, moving_image
