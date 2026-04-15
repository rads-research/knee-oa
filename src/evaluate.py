"""
Evaluate trained models under all perturbation conditions.

For each model (E0 and E1) x dataset (internal, external):
  - Apply each pattern at each severity level
  - Compute all metrics
  - Save results as JSON and CSV

Usage:
    python src/evaluate.py --data_dir data/ --model_dir outputs/models/ --output_dir outputs/results/
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import KneeOADataset
from model import load_model
from artifacts import ArtifactPatterns, PATTERN_NAMES, SEVERITY_LEVELS
from metrics import compute_metrics, compute_per_grade_metrics

SEEDS = [42, 123, 456, 789, 1024]


def evaluate_single_model(model, data_loader, pattern_name, alpha, device):
    """Evaluate one model under one perturbation condition."""
    all_probs = []
    all_preds = []
    all_labels = []
    all_grades = []

    with torch.no_grad():
        for img, lbl, grade, idx in data_loader:
            img = img.float()

            # Apply perturbation AFTER preprocessing (on normalized tensor)
            perturbed = ArtifactPatterns.apply_pattern(
                img[0], pattern_name, alpha=alpha
            ).unsqueeze(0).to(device)

            logits = model(perturbed)
            prob = torch.sigmoid(logits).cpu().numpy()[0]
            prob = float(np.clip(np.nan_to_num(prob, nan=0.5), 0, 1))
            pred = int(prob >= 0.5)

            all_probs.append(prob)
            all_preds.append(pred)
            all_labels.append(lbl.item())
            all_grades.append(grade.item())

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    grade_metrics = compute_per_grade_metrics(all_labels, all_preds, all_grades)
    metrics["per_grade"] = grade_metrics

    return metrics, all_probs, all_preds, all_labels, all_grades


def evaluate_experiment(model_paths, data_loader, dataset_label, device):
    """
    Evaluate all models across all conditions for one experiment x dataset.

    Returns:
        Dictionary with per-seed and aggregated results
    """
    # Build conditions: clean + 4 patterns x 3 severities = 13
    conditions = [("clean", 1.0, "clean")]
    for pname in PATTERN_NAMES[1:]:
        for alpha in SEVERITY_LEVELS:
            sev_label = f"{int(alpha * 100)}pct"
            cond_key = f"{pname}_{sev_label}"
            conditions.append((pname, alpha, cond_key))

    all_results = {}

    for model_name, model_path in model_paths.items():
        print(f"  {model_name}...")
        model = load_model(model_path, device)

        all_results[model_name] = {}
        for pname, alpha, cond_key in tqdm(conditions, desc=f"    {model_name}", leave=False):
            metrics, probs, preds, labels, grades = evaluate_single_model(
                model, data_loader, pname, alpha, device
            )
            all_results[model_name][cond_key] = metrics

        del model
        torch.cuda.empty_cache()

    # Aggregate across seeds
    aggregated = {}
    for _, _, cond_key in conditions:
        cond_metrics = {}
        for metric_name in all_results[list(all_results.keys())[0]][cond_key]:
            if metric_name == "per_grade":
                continue
            values = [
                all_results[mn][cond_key][metric_name]
                for mn in all_results
            ]
            cond_metrics[f"{metric_name}_mean"] = float(np.mean(values))
            cond_metrics[f"{metric_name}_std"] = float(np.std(values))

        # Per-grade aggregation
        for g in range(5):
            grade_accs = []
            for mn in all_results:
                pg = all_results[mn][cond_key].get("per_grade", {}).get(g, {})
                if pg.get("n", 0) > 0:
                    grade_accs.append(pg["accuracy"])
            if grade_accs:
                cond_metrics[f"grade_{g}_accuracy_mean"] = float(np.mean(grade_accs))
                cond_metrics[f"grade_{g}_accuracy_std"] = float(np.std(grade_accs))

        aggregated[cond_key] = cond_metrics

    return {"per_seed": all_results, "aggregated": aggregated}


def main():
    parser = argparse.ArgumentParser(description="Evaluate models under perturbation")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load datasets
    print("\nLoading internal test set...")
    test_ds = KneeOADataset(os.path.join(args.data_dir, "test"))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    print("\nLoading external validation set...")
    ext_ds = KneeOADataset(os.path.join(args.data_dir, "external_val"))
    ext_loader = DataLoader(ext_ds, batch_size=1, shuffle=False, num_workers=2)

    # Define experiments
    experiments = [
        ("E0", "baseline", "Baseline models"),
        ("E1", "augmented", "Augmentation-trained models"),
    ]

    datasets = [
        ("internal", test_loader, "Internal test set"),
        ("external", ext_loader, "External validation"),
    ]

    for exp_prefix, model_subdir, exp_label in experiments:
        # Find model files
        model_dir = os.path.join(args.model_dir, model_subdir)
        model_paths = {}
        for seed in SEEDS:
            name = f"{exp_prefix}_s{seed}"
            path = os.path.join(model_dir, f"{name}.pt")
            if os.path.exists(path):
                model_paths[name] = path

        if not model_paths:
            print(f"\nNo {exp_label} found in {model_dir}, skipping")
            continue

        print(f"\n{'=' * 60}")
        print(f"EVALUATING: {exp_label} ({len(model_paths)} models)")
        print(f"{'=' * 60}")

        for ds_label, ds_loader, ds_name in datasets:
            print(f"\n  Dataset: {ds_name}")
            results = evaluate_experiment(model_paths, ds_loader, ds_label, device)

            # Save results
            out_file = os.path.join(
                args.output_dir, f"{exp_prefix}_{ds_label}_results.json"
            )
            with open(out_file, "w") as f:
                json.dump(results["aggregated"], f, indent=2)
            print(f"  Saved: {out_file}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
