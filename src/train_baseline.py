"""
Train baseline (E0) models for knee OA classification.

Trains 5 EfficientNet-V2-S models with independent random seeds.
No artifact augmentation is applied during training.

Usage:
    python src/train_baseline.py --data_dir data/ --output_dir outputs/
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from sklearn.metrics import balanced_accuracy_score

from dataset import KneeOADataset
from model import Classifier

# Configuration (matches manuscript Section 2.4)
SEEDS = [42, 123, 456, 789, 1024]
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 15


def train_model(seed, train_root, val_root, save_path, device):
    """
    Train one baseline model.

    Training details (from manuscript):
    - BCEWithLogitsLoss with inverse class-frequency weighting
    - AdamW optimizer (lr=1e-4, wd=1e-2)
    - Cosine annealing with warm restarts (T_0=10, T_mult=2)
    - Early stopping on validation balanced accuracy (patience=15)
    - ImageNet pretrained weights
    """
    print(f"\n  Seed: {seed}")

    # Reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Datasets — NO augmentation for baseline
    train_ds = KneeOADataset(train_root, augment_fn=None)
    val_ds = KneeOADataset(val_root, augment_fn=None)

    # Inverse class-frequency weighting
    n_kl01 = train_ds.n_kl01
    n_kl234 = train_ds.n_kl234
    total = n_kl01 + n_kl234
    weight_kl01 = total / (2 * n_kl01)
    weight_kl234 = total / (2 * n_kl234)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # Model with ImageNet pretrained backbone
    model = Classifier(dropout=0.2).to(device)
    pretrained = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    pretrained_dict = pretrained.state_dict()
    model_dict = model.net.state_dict()
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items()
        if k in model_dict and "classifier" not in k
    }
    model_dict.update(pretrained_dict)
    model.net.load_state_dict(model_dict)
    model.float()

    # Loss with class weighting
    pos_weight = torch.tensor([weight_kl234 / weight_kl01]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Training loop with early stopping
    best_val_balacc = 0.0
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        # Train
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []

        for imgs, lbls, grades, idxs in train_loader:
            imgs = imgs.float().to(device)
            lbls = lbls.float().to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(lbls.cpu().numpy().astype(int))

        scheduler.step()
        train_loss /= len(train_ds)
        train_balacc = balanced_accuracy_score(train_labels, train_preds)

        # Validate
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for imgs, lbls, grades, idxs in val_loader:
                imgs = imgs.float().to(device)
                logits = model(imgs)
                preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(lbls.numpy())

        val_balacc = balanced_accuracy_score(val_labels, val_preds)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1:3d}: loss={train_loss:.4f} "
                f"train_ba={train_balacc:.4f} val_ba={val_balacc:.4f}"
            )

        # Early stopping
        if val_balacc > best_val_balacc:
            best_val_balacc = val_balacc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    print(f"  Best val BalAcc: {best_val_balacc:.4f}")
    print(f"  Saved: {save_path}")

    del model
    torch.cuda.empty_cache()

    return best_val_balacc


def main():
    parser = argparse.ArgumentParser(description="Train baseline (E0) models")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_dir = os.path.join(args.output_dir, "models", "baseline")
    os.makedirs(model_dir, exist_ok=True)

    train_root = os.path.join(args.data_dir, "train")
    val_root = os.path.join(args.data_dir, "val")

    print("=" * 60)
    print("BASELINE MODEL TRAINING (E0)")
    print("=" * 60)

    for seed in SEEDS:
        model_name = f"E0_s{seed}"
        save_path = os.path.join(model_dir, f"{model_name}.pt")

        if os.path.exists(save_path):
            print(f"\n  {model_name}: already exists, skipping")
            continue

        train_model(
            seed=seed,
            train_root=train_root,
            val_root=val_root,
            save_path=save_path,
            device=device,
        )

    print("\nAll baseline models trained.")


if __name__ == "__main__":
    main()
