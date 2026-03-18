"""
evaluate.py – Compute calibration and accuracy metrics for a trained model.

Metrics computed:
  - Top-1 Accuracy
  - Expected Calibration Error (ECE-15, equal-width bins)
  - Maximum Calibration Error (MCE)
  - Negative Log-Likelihood (NLL)
  - Brier Score

Usage:
    python evaluate.py --dataset cifar10 \
        --arch resnet20 --checkpoint results/student_cifar10_resnet20_standard_kd_seed42.pth \
        --output results/eval_student_cifar10_resnet20_standard_kd_seed42.json
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from models import get_model
from train_teacher import get_dataloaders

NUM_BINS = 15


def compute_metrics(logits_all: np.ndarray, labels_all: np.ndarray, num_bins: int = NUM_BINS):
    """
    Compute calibration and accuracy metrics from logits and labels.

    Args:
        logits_all: (N, C) numpy array of raw logits
        labels_all: (N,)  numpy array of integer class labels
        num_bins:   number of equal-width confidence bins for ECE/MCE

    Returns:
        dict with keys: top1_acc, ece, mce, nll, brier_score
    """
    probs = softmax(logits_all)                          # (N, C)
    confs = probs.max(axis=1)                            # (N,)
    preds = probs.argmax(axis=1)                         # (N,)
    accs  = (preds == labels_all).astype(float)          # (N,)
    n     = len(labels_all)

    # ECE and MCE
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece_sum, mce = 0.0, 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confs > lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confs[mask].mean()
        bin_acc  = accs[mask].mean()
        bin_prop = mask.sum() / n
        cal_err  = abs(bin_conf - bin_acc)
        ece_sum += bin_prop * cal_err
        mce = max(mce, cal_err)

    # NLL
    log_probs = np.log(np.clip(probs[np.arange(n), labels_all], 1e-12, 1.0))
    nll = -log_probs.mean()

    # Brier Score (multi-class)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), labels_all] = 1.0
    brier = ((probs - one_hot) ** 2).sum(axis=1).mean()

    return {
        "top1_acc":    float(accs.mean() * 100),
        "ece":         float(ece_sum),
        "mce":         float(mce),
        "nll":         float(nll),
        "brier_score": float(brier),
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp    = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        logits = model(inputs).cpu()
        all_logits.append(logits)
        all_labels.append(labels)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--arch",       default="resnet20")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--num_bins",   type=int, default=NUM_BINS)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_loader, num_classes = get_dataloaders(args.dataset, batch_size=256)

    model = get_model(args.arch, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    logits, labels = collect_logits(model, test_loader, device)
    metrics = compute_metrics(logits, labels, num_bins=args.num_bins)

    print(f"Top-1 Acc : {metrics['top1_acc']:.2f}%")
    print(f"ECE-{args.num_bins:<2d}   : {metrics['ece']:.4f}")
    print(f"MCE       : {metrics['mce']:.4f}")
    print(f"NLL       : {metrics['nll']:.4f}")
    print(f"Brier     : {metrics['brier_score']:.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({**metrics, "checkpoint": args.checkpoint, "dataset": args.dataset,
                   "arch": args.arch, "num_bins": args.num_bins}, f, indent=2)
    print(f"\nMetrics saved to {args.output}")


if __name__ == "__main__":
    main()
