# Artifact-Triggered Vulnerability and Augmentation-Based Mitigation in a Deep Learning Classifier for Knee Osteoarthritis

Code repository for the manuscript: *"Artifact-Triggered Vulnerability and Augmentation-Based Mitigation in a Deep Learning Classifier for Knee Osteoarthritis"*.

## Overview

This repository contains the code for:
1. **Baseline model training** — EfficientNet-V2-S for binary knee OA classification (KL 0-1 vs KL 2-4)
2. **Artifact pattern generation** — Four parameterized perturbation patterns at three severity levels
3. **Augmentation-based mitigation** — Re-training with random artifact augmentation during training
4. **Evaluation** — Comprehensive metrics computation under all perturbation conditions

## Repository Structure

```
├── src/
│   ├── dataset.py            # Dataset class with CLAHE preprocessing
│   ├── model.py              # EfficientNet-V2-S classifier definition
│   ├── artifacts.py          # Four artifact pattern implementations
│   ├── train_baseline.py     # Train baseline (E0) models across 5 seeds
│   ├── train_augmented.py    # Train augmentation-based (E1) models
│   ├── evaluate.py           # Evaluate models under perturbation conditions
│   └── metrics.py            # Metric computation functions
├── configs/
│   └── experiment.yaml       # All hyperparameters and configuration
├── requirements.txt
└── README.md
```

## Data

This study uses two publicly available datasets:

- **Internal dataset (training/validation/test):** Chen P. Knee Osteoarthritis Severity Grading Dataset. Mendeley Data, V1, 2018. [doi:10.17632/56rmx5bjcr.1](https://doi.org/10.17632/56rmx5bjcr.1)
- **External validation:** Gornale S, Patravali P. Digital Knee X-ray Images (MedicalExpert-II annotations). Mendeley Data, V1, 2020. [doi:10.17632/t9ndx37v5h.1](https://doi.org/10.17632/t9ndx37v5h.1)

Download both datasets and organize as:
```
data/
├── train/{0,1,2,3,4}/       # Internal training set (n=5,778)
├── val/{0,1,2,3,4}/         # Internal validation set (n=826)
├── test/{0,1,2,3,4}/        # Internal test set (n=1,656)
└── external_val/{0,1,2,3,4}/ # External validation (n=1,650)
```

## Reproducing Results

### 1. Train Baseline Models (E0)

Trains 5 models with independent random seeds (42, 123, 456, 789, 1024):

```bash
python src/train_baseline.py --data_dir data/ --output_dir outputs/
```

### 2. Train Augmentation-Based Models (E1)

Re-trains 5 models with 40% artifact augmentation probability during training:

```bash
python src/train_augmented.py --data_dir data/ --output_dir outputs/
```

### 3. Evaluate Under Perturbation

Evaluates all models under all pattern × severity combinations:

```bash
python src/evaluate.py --data_dir data/ --model_dir outputs/models/ --output_dir outputs/results/
```

## Key Configuration

| Parameter | Value |
|---|---|
| Architecture | EfficientNet-V2-S |
| Input size | 224 × 224 |
| Preprocessing | CLAHE (clip=2.0, tile=8×8) → normalize [-1, 1] |
| Optimizer | AdamW (lr=1e-4, wd=1e-2) |
| Scheduler | Cosine annealing with warm restarts |
| Loss | BCEWithLogitsLoss (inverse class-frequency weighting) |
| Batch size | 32 |
| Early stopping | Patience 15, monitoring validation balanced accuracy |
| Seeds | 42, 123, 456, 789, 1024 |
| Augmentation probability (E1) | 40% |
| Augmentation severity range (E1) | [0.5, 1.5] |

## Artifact Patterns

| Pattern | Parameters (at α=1.0) |
|---|---|
| Horizontal lines | 10px thick, intensity 0.5, every 30px |
| Checkerboard | 16×16 blocks, ±0.3 modulation |
| Black bars | 25px top/bottom, 15px left, value −1.0 |
| Grid overlay | 2px lines every 20px, intensity 0.6 |

All parameters scale linearly with the severity multiplier α ∈ {0.5, 1.0, 1.5}. Patterns are applied **after** all preprocessing steps on normalized tensors.

## Data Leakage Prevention

- Artifact augmentation is applied **only to training images** during E1 model training
- Validation images used for early stopping receive **no augmentation**
- The same train/validation/test partition is used for both E0 and E1 models
- The external validation set is **never** used during training or validation
- Perturbation patterns for evaluation are applied at inference time only

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (tested on NVIDIA A100 80GB)

## License

MIT
