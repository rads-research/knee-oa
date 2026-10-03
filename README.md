# Performance Degradation of a Deep Learning Knee Osteoarthritis Classifier Under Synthetic Radiographic Artifacts: Effect of Artifact-Based Training Augmentation

Code for the manuscript *"Performance Degradation of a Deep Learning Knee Osteoarthritis Classifier Under Synthetic Radiographic Artifacts: Effect of Artifact-Based Training Augmentation"*.

## Overview

This repository contains code for:

1. **Baseline model training**: EfficientNet-V2-S for binary knee OA classification (KL 0-1 vs KL 2-4)
2. **Artifact pattern generation**: four synthetic artifact-inspired degradations at three severity levels
3. **Augmentation-trained models**: re-training with random artifact augmentation
4. **Evaluation**: classification metrics on clean and degraded images for the internal test set and the external validation set

## Repository Structure

```
├── src/
│   ├── dataset.py            # Dataset class and preprocessing (resize, CLAHE, normalization)
│   ├── model.py              # EfficientNet-V2-S classifier
│   ├── artifacts.py          # Artifact pattern implementations
│   ├── train_baseline.py     # Baseline models (E0), 5 seeds
│   ├── train_augmented.py    # Augmentation-trained models (E1), 5 seeds
│   ├── evaluate.py           # Evaluation on clean and degraded images
│   └── metrics.py            # Metric functions
├── configs/
│   └── experiment.yaml       # Hyperparameters and configuration
├── requirements.txt
└── README.md
```

## Data

Two publicly available datasets were used:

- **Internal dataset (training, validation, test):** Chen P. Knee Osteoarthritis Severity Grading Dataset. Mendeley Data, V1, 2018. [doi:10.17632/56rmx5bjcr.1](https://doi.org/10.17632/56rmx5bjcr.1). Derived from the Osteoarthritis Initiative (OAI). License: CC BY 4.0.
- **External validation:** Gornale S, Patravali P. Digital Knee X-ray Images. Mendeley Data, V1, 2020. [doi:10.17632/t9ndx37v5h.1](https://doi.org/10.17632/t9ndx37v5h.1). MedicalExpert-II annotations were used. License: CC BY 4.0.

Download both datasets and organize them as follows:

```
data/
├── train/{0,1,2,3,4}/         # Internal training set (n=5,778)
├── val/{0,1,2,3,4}/           # Internal validation set (n=826)
├── test/{0,1,2,3,4}/          # Internal test set (n=1,656)
└── external_val/{0,1,2,3,4}/  # External validation set (n=1,650)
```

## Reproducing Results

### 1. Train baseline models (E0)

Trains five models with seeds 42, 123, 456, 789, and 1024. No artifact augmentation is used.

```bash
python src/train_baseline.py --data_dir data/ --output_dir outputs/
```

### 2. Train augmentation-trained models (E1)

Same setup as E0, except each training image has a 40% probability of receiving a randomly selected artifact pattern with severity sampled uniformly from 0.5 to 1.5.

```bash
python src/train_augmented.py --data_dir data/ --output_dir outputs/
```

### 3. Evaluate on clean and degraded images

Evaluates all models on clean images and on each pattern at each severity level, for both the internal test set and the external validation set.

```bash
python src/evaluate.py --data_dir data/ --model_dir outputs/models/ --output_dir outputs/results/
```

## Configuration

| Parameter | Value |
|---|---|
| Architecture | EfficientNet-V2-S (ImageNet-pretrained) |
| Input size | 224 × 224 |
| Preprocessing | Resize, CLAHE (clip 2.0, tile 8×8), normalize to [-1, 1], grayscale replicated to 3 channels |
| Loss | BCEWithLogitsLoss with inverse class-frequency weighting |
| Optimizer | AdamW (lr 1e-4, weight decay 1e-2) |
| Scheduler | Cosine annealing with warm restarts (T0 = 10, T_mult = 2) |
| Dropout | 0.2 |
| Batch size | 32 |
| Early stopping | Patience 15, validation balanced accuracy, max 100 epochs |
| Seeds | 42, 123, 456, 789, 1024 |
| Decision threshold | 0.5 |
| Augmentation probability (E1) | 0.4 |
| Augmentation severity range (E1) | 0.5 to 1.5 |

## Artifact Patterns

| Pattern | Parameters at α = 1.0 | Scaled by α | Operation |
|---|---|---|---|
| Horizontal lines | 10 px bands, intensity 0.5, every 30 px starting at row 20 | Thickness, intensity | Replaces pixel values |
| Checkerboard | 16 × 16 px blocks, +0.3 (even blocks) / -0.15 (odd blocks) | Offset magnitude | Added to pixel values |
| Black bars | 25 px top and bottom, 15 px left, value -1.0 | Bar width | Replaces pixel values |
| Grid overlay | 2 px lines every 20 px, intensity 0.6 | Thickness, intensity | Replaces pixel values |

Severity levels are α = 0.5, 1.0, and 1.5. Line spacing and block size are fixed. Intensities refer to the normalized [-1, 1] range, and all outputs are clipped to [-1, 1]. Patterns are applied after all preprocessing steps.

## Data Leakage Prevention

- Artifact augmentation is applied only to training images of E1 models.
- Validation images used for early stopping are not augmented.
- E0 and E1 models use the same training, validation, and test partition.
- The external validation set is not used for training or model selection.
- Artifact patterns for evaluation are applied at inference time only.

## Notes on Reported Values

- Mean ± SD across seeds uses the population standard deviation (`np.std`, ddof=0).
- Calibration (ECE, Brier score), AUPRC, image similarity (SSIM, PSNR), and statistical analyses (paired t-tests, Holm-Bonferroni correction, Cohen's d) were computed separately from model outputs. These scripts are available from the corresponding author on request.

## Requirements

- Python 3.10+
- PyTorch 2.0+ and torchvision 0.15+
- CUDA-capable GPU (tested on an NVIDIA A100 80GB)

Install dependencies:

```bash
pip install -r requirements.txt
```

## License

MIT
