"""
Dataset class for knee OA radiographs.

Preprocessing pipeline (applied in sequence):
1. Resize to 224x224
2. CLAHE (clip_limit=2.0, tile_grid=8x8)
3. Normalize to [-1, 1]
4. Replicate grayscale to 3 channels
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class CLAHE:
    """Contrast-limited adaptive histogram equalization."""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=tile_grid_size
        )

    def __call__(self, img):
        return self.clahe.apply(img)


class KneeOADataset(Dataset):
    """
    Dataset for knee OA radiographs with KL grade annotations.

    Expected directory structure:
        root/{0,1,2,3,4}/*.png   (or .jpg, .jpeg)

    Grade folders can also be named '0Normal', '1Doubtful', etc.
    Binary label: 0 = KL01 (grade 0-1), 1 = KL234 (grade 2-4).

    Args:
        root: Path to dataset directory containing grade subfolders
        augment_fn: Optional function applied to each image tensor during
                    __getitem__. Used for artifact augmentation during
                    training. NOT applied to validation or test sets.
    """

    def __init__(self, root, augment_fn=None):
        self.samples = []
        self.labels = []
        self.grades = []
        self.clahe = CLAHE()
        self.augment_fn = augment_fn

        root = Path(root)
        for grade_dir in sorted(root.iterdir()):
            if not grade_dir.is_dir():
                continue

            grade_name = grade_dir.name
            if grade_name.isdigit():
                grade = int(grade_name)
            elif grade_name[0].isdigit():
                grade = int(grade_name[0])
            else:
                continue

            label = 0 if grade <= 1 else 1  # KL01 vs KL234

            for p in grade_dir.glob("*"):
                if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    self.samples.append(str(p))
                    self.labels.append(label)
                    self.grades.append(grade)

        grades_arr = np.array(self.grades)
        labels_arr = np.array(self.labels)

        self.n_per_grade = {g: int((grades_arr == g).sum()) for g in range(5)}
        self.n_kl01 = int((labels_arr == 0).sum())
        self.n_kl234 = int((labels_arr == 1).sum())

        print(f"  Total: {len(self)} images")
        for g in range(5):
            if self.n_per_grade[g] > 0:
                cls = "KL01" if g <= 1 else "KL234"
                print(f"  Grade {g} ({cls}): {self.n_per_grade[g]}")
        print(f"  KL01: {self.n_kl01} | KL234: {self.n_kl234}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load as grayscale
        img = cv2.imread(self.samples[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.array(Image.open(self.samples[idx]).convert("L"))

        # Preprocessing pipeline
        img = cv2.resize(img, (224, 224))
        img = self.clahe(img)
        img = (img.astype(np.float32) / 127.5) - 1.0
        img = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1)

        # Apply augmentation ONLY if augment_fn is set (training only)
        if self.augment_fn is not None:
            img = self.augment_fn(img)

        return img, self.labels[idx], self.grades[idx], idx
