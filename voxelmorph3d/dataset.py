import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import os
from pathlib import Path


class OASISDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        """
        Args:
            root_dir: Path to directory containing 'train' and 'val' subdirectories
            split: Either "train" or "val"
        """
        self.root_dir = root_dir
        self.split = split

        split_dir = os.path.join(root_dir, split)
        self.image_paths = []

        for subject_folder in sorted(os.listdir(split_dir)):
            subject_path = os.path.join(split_dir, subject_folder)
            if os.path.isdir(subject_path):
                norm_file = os.path.join(subject_path, "aligned_norm.nii.gz")
                seg_file  = os.path.join(subject_path, "aligned_seg35.nii.gz")
                if os.path.exists(norm_file) and os.path.exists(seg_file):
                    self.image_paths.append((norm_file, seg_file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        moving_norm_path, moving_seg_path = self.image_paths[idx]

        # Randomly choose a fixed subject different from moving
        fixed_idx = np.random.randint(0, len(self.image_paths) - 1)
        if fixed_idx >= idx:
            fixed_idx += 1
        fixed_norm_path, fixed_seg_path = self.image_paths[fixed_idx]

        moving_img = nib.load(moving_norm_path).get_fdata()
        fixed_img  = nib.load(fixed_norm_path).get_fdata()
        moving_seg = nib.load(moving_seg_path).get_fdata()
        fixed_seg  = nib.load(fixed_seg_path).get_fdata()

        moving_tensor     = torch.from_numpy(moving_img).unsqueeze(0).float()
        fixed_tensor      = torch.from_numpy(fixed_img).unsqueeze(0).float()
        moving_seg_tensor = torch.from_numpy(moving_seg).unsqueeze(0).float()
        fixed_seg_tensor  = torch.from_numpy(fixed_seg).unsqueeze(0).float()

        return fixed_tensor, moving_tensor, fixed_seg_tensor, moving_seg_tensor
