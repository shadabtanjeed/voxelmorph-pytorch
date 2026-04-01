# Get Started

- Install dependencies: `pip install -r requirements.txt`
- Prepare the OASIS dataset. The root directory must have the following structure:
  ```
  data/
    train/
      subject_001/
        aligned_seg35.nii.gz
      subject_002/
        aligned_seg35.nii.gz
      ...
    val/
      subject_N/
        aligned_seg35.nii.gz
      ...
  ```
- Train the model with `train.py`. Provide the path to the dataset root, for example: `python train.py --data_path data/`.
- To visualize results, run `notebooks/inference.ipynb`.
