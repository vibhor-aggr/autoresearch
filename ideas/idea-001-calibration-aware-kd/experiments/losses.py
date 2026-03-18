"""
losses.py – Loss functions for Calibration-Aware Knowledge Distillation (CA-KD).

Implements:
  - standard_kd_loss   : Hinton et al. (2015) KD loss
  - soft_ece_loss      : differentiable approximation of Expected Calibration Error
  - cakd_loss          : combined CA-KD objective = KD + λ * ECE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Standard Knowledge-Distillation loss (Hinton et al., 2015)
# ---------------------------------------------------------------------------

def standard_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.9,
) -> torch.Tensor:
    """
    L_KD = alpha * L_CE(student_logits, labels)
         + (1 - alpha) * tau^2 * KL(teacher_soft || student_soft)

    Args:
        student_logits: raw logits from student  (N, C)
        teacher_logits: raw logits from teacher  (N, C)
        labels:         ground-truth class indices (N,)
        temperature:    softening temperature tau
        alpha:          weight for hard-label cross-entropy

    Returns:
        Scalar loss tensor.
    """
    ce_loss = F.cross_entropy(student_logits, labels)

    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    kd_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temperature ** 2)

    return alpha * ce_loss + (1.0 - alpha) * kd_loss


# ---------------------------------------------------------------------------
# Differentiable soft-binning ECE
# ---------------------------------------------------------------------------

def soft_ece_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 15,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Differentiable approximation of the Expected Calibration Error (ECE).

    Instead of hard binning, each sample is assigned to each bin with a soft
    weight proportional to a triangular kernel centred on the bin midpoint.
    This keeps the operation differentiable w.r.t. logits.

    Args:
        logits:   raw logits from model  (N, C)
        labels:   ground-truth class indices (N,)
        num_bins: number of equal-width confidence bins
        eps:      small value for numerical stability

    Returns:
        Scalar soft-ECE loss in [0, 1].
    """
    probs = F.softmax(logits, dim=1)                          # (N, C)
    confidences, predictions = torch.max(probs, dim=1)        # (N,)
    accuracies = predictions.eq(labels).float()               # (N,)

    bin_boundaries = torch.linspace(0.0, 1.0, num_bins + 1, device=logits.device)
    bin_midpoints  = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])   # (B,)
    bin_width      = 1.0 / num_bins

    # Soft assignment weight: triangular kernel, width = bin_width
    # weight[i, b] = max(0, 1 - |conf_i - mid_b| / (bin_width))
    conf_expand = confidences.unsqueeze(1)               # (N, 1)
    mid_expand  = bin_midpoints.unsqueeze(0)             # (1, B)
    weights     = F.relu(1.0 - (conf_expand - mid_expand).abs() / bin_width)  # (N, B)

    weight_sum      = weights.sum(dim=0) + eps           # (B,)
    bin_conf        = (weights * conf_expand).sum(dim=0) / weight_sum   # (B,)
    bin_acc         = (weights * accuracies.unsqueeze(1)).sum(dim=0) / weight_sum  # (B,)
    bin_proportion  = weight_sum / (weights.sum() + eps)  # (B,)

    ece = (bin_proportion * (bin_conf - bin_acc).abs()).sum()
    return ece


# ---------------------------------------------------------------------------
# Calibration-Aware KD (CA-KD) objective
# ---------------------------------------------------------------------------

def cakd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.9,
    lambda_cal: float = 0.1,
    num_bins: int = 15,
) -> torch.Tensor:
    """
    L_CAKD = L_KD + lambda_cal * L_cal

    Args:
        student_logits: (N, C)
        teacher_logits: (N, C)
        labels:         (N,)
        temperature:    KD temperature
        alpha:          KD hard-label weight
        lambda_cal:     calibration loss weight
        num_bins:       bins for soft-ECE

    Returns:
        Scalar loss tensor.
    """
    kd   = standard_kd_loss(student_logits, teacher_logits, labels, temperature, alpha)
    cal  = soft_ece_loss(student_logits, labels, num_bins)
    return kd + lambda_cal * cal
