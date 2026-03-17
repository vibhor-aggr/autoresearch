"""
test_losses.py – Unit tests for losses.py

Run with:
    cd experiments && python -m pytest tests/test_losses.py -v
"""

import sys
import os

import pytest
import torch
import torch.nn.functional as F

# Make the experiments directory importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from losses import standard_kd_loss, soft_ece_loss, cakd_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perfect_model_logits(labels: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """
    Return logits where the correct class has logit 100 and all others have 0.
    A softmax over these is essentially a one-hot → confidence ≈ 1.0, accuracy = 1.0
    → ECE ≈ |1.0 - 1.0| = 0.
    """
    n = labels.size(0)
    logits = torch.zeros(n, num_classes)
    logits[torch.arange(n), labels] = 100.0
    return logits


def _uniform_logits(n: int, num_classes: int = 10) -> torch.Tensor:
    """All-zero logits → uniform distribution → confidence = 1/C."""
    return torch.zeros(n, num_classes)


# ---------------------------------------------------------------------------
# standard_kd_loss tests
# ---------------------------------------------------------------------------

class TestStandardKDLoss:

    def test_returns_scalar(self):
        s = torch.randn(32, 10)
        t = torch.randn(32, 10)
        y = torch.randint(0, 10, (32,))
        loss = standard_kd_loss(s, t, y)
        assert loss.ndim == 0, "Loss should be a scalar"

    def test_non_negative(self):
        s = torch.randn(32, 10)
        t = torch.randn(32, 10)
        y = torch.randint(0, 10, (32,))
        loss = standard_kd_loss(s, t, y)
        assert loss.item() >= 0.0

    def test_perfect_student_low_loss(self):
        """When student exactly matches teacher at temperature 1, KL divergence = 0."""
        t = torch.randn(32, 10)
        y = torch.randint(0, 10, (32,))
        loss = standard_kd_loss(t.clone(), t, y, temperature=1.0, alpha=0.0)
        assert loss.item() < 1e-4, f"KL should be ~0 when student=teacher, got {loss.item()}"

    def test_alpha_boundary_ce_only(self):
        """alpha=1.0 → loss equals cross-entropy."""
        s = torch.randn(32, 10)
        t = torch.randn(32, 10)
        y = torch.randint(0, 10, (32,))
        kd_loss = standard_kd_loss(s, t, y, alpha=1.0)
        ce_loss = F.cross_entropy(s, y)
        assert abs(kd_loss.item() - ce_loss.item()) < 1e-5

    def test_gradients_flow(self):
        """Gradients should flow through the student logits."""
        s = torch.randn(32, 10, requires_grad=True)
        t = torch.randn(32, 10)
        y = torch.randint(0, 10, (32,))
        loss = standard_kd_loss(s, t, y)
        loss.backward()
        assert s.grad is not None
        assert not torch.isnan(s.grad).any()


# ---------------------------------------------------------------------------
# soft_ece_loss tests
# ---------------------------------------------------------------------------

class TestSoftECELoss:

    def test_returns_scalar(self):
        logits = torch.randn(64, 10)
        labels = torch.randint(0, 10, (64,))
        ece = soft_ece_loss(logits, labels)
        assert ece.ndim == 0

    def test_in_range(self):
        logits = torch.randn(64, 10)
        labels = torch.randint(0, 10, (64,))
        ece = soft_ece_loss(logits, labels)
        assert 0.0 <= ece.item() <= 1.0, f"ECE should be in [0,1], got {ece.item()}"

    def test_perfect_model_near_zero(self):
        """Perfect model: confidence = accuracy for every sample → ECE ≈ 0."""
        n = 200
        labels = torch.randint(0, 10, (n,))
        logits = _perfect_model_logits(labels, num_classes=10)
        ece = soft_ece_loss(logits, labels, num_bins=15)
        assert ece.item() < 0.05, f"Perfect model should have ECE ≈ 0, got {ece.item()}"

    def test_gradients_flow(self):
        logits = torch.randn(64, 10, requires_grad=True)
        labels = torch.randint(0, 10, (64,))
        ece = soft_ece_loss(logits, labels)
        ece.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_different_num_bins(self):
        """ECE should be computable for any reasonable number of bins."""
        logits = torch.randn(64, 10)
        labels = torch.randint(0, 10, (64,))
        for nb in [5, 10, 15, 20]:
            ece = soft_ece_loss(logits, labels, num_bins=nb)
            assert 0.0 <= ece.item() <= 1.0


# ---------------------------------------------------------------------------
# cakd_loss tests
# ---------------------------------------------------------------------------

class TestCAKDLoss:

    def test_returns_scalar(self):
        s = torch.randn(32, 10)
        t = torch.randn(32, 10)
        y = torch.randint(0, 10, (32,))
        loss = cakd_loss(s, t, y)
        assert loss.ndim == 0

    def test_lambda_zero_equals_standard_kd(self):
        """With lambda_cal=0, CA-KD should equal standard KD."""
        s = torch.randn(32, 10)
        t = torch.randn(32, 10)
        y = torch.randint(0, 10, (32,))
        ca_loss  = cakd_loss(s, t, y, lambda_cal=0.0)
        std_loss = standard_kd_loss(s, t, y)
        assert abs(ca_loss.item() - std_loss.item()) < 1e-5

    def test_positive_lambda_increases_loss(self):
        """Adding calibration loss with λ > 0 should generally increase total loss."""
        torch.manual_seed(0)
        s = torch.randn(64, 10)
        t = torch.randn(64, 10)
        y = torch.randint(0, 10, (64,))
        std_loss = standard_kd_loss(s, t, y)
        ca_loss  = cakd_loss(s, t, y, lambda_cal=1.0)
        # CA-KD ≥ standard KD because we add a non-negative term
        assert ca_loss.item() >= std_loss.item() - 1e-5

    def test_gradients_flow(self):
        s = torch.randn(32, 10, requires_grad=True)
        t = torch.randn(32, 10)
        y = torch.randint(0, 10, (32,))
        loss = cakd_loss(s, t, y)
        loss.backward()
        assert s.grad is not None
        assert not torch.isnan(s.grad).any()
