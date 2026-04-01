# VoxelMorph 3D - PyTorch Implementation

3D deformable image registration using deep learning.

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Setup

1. Download the OASIS dataset from: https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md
2. Prepare the dataset using `notebooks/eda.ipynb`
3. Expected directory structure:
   ```
   data/
     train/
       subject_001/
         aligned_norm.nii.gz
         aligned_seg35.nii.gz
       subject_002/...
     val/
       subject_N/...
   ```

## Training

```bash
python train.py --data_path /path/to/data/
```

## Inference

Open and run `notebooks/inference.ipynb` to visualize registration results.
