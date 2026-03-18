"""
test_gradient_noise_optimizer.py – Unit tests for the gradient noise optimizer.

Run with:
    cd experiments && python -m pytest tests/test_gradient_noise_optimizer.py -v
"""

import sys
import os

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from gradient_noise_optimizer import GradientNoiseOptimizer, AdaptiveGradientNoiseOptimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_model_and_loss():
    model = nn.Linear(8, 2)
    x     = torch.randn(4, 8)
    y     = torch.randint(0, 2, (4,))
    return model, x, y


def _run_one_step(optimizer_cls, base_cls=SGD, **gn_kwargs):
    """Run a single forward+backward+step and return gradients before/after noise."""
    model, x, y = _simple_model_and_loss()
    base_opt  = base_cls(model.parameters(), lr=1e-3)
    optimizer = optimizer_cls(base_opt, **gn_kwargs)

    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()

    # Capture gradients before noise injection (deep copy)
    pre_noise_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    # step() injects noise internally then calls base optimizer
    optimizer.step()

    return pre_noise_grads, model, optimizer


# ---------------------------------------------------------------------------
# GradientNoiseOptimizer tests
# ---------------------------------------------------------------------------

class TestGradientNoiseOptimizer:

    def test_step_executes(self):
        """Basic smoke test: step should not raise."""
        model, x, y = _simple_model_and_loss()
        base = SGD(model.parameters(), lr=1e-3)
        opt  = GradientNoiseOptimizer(base, eta=1e-3, gamma=0.55)
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        opt.step()  # should not raise

    def test_step_count_increments(self):
        model, x, y = _simple_model_and_loss()
        base = SGD(model.parameters(), lr=1e-3)
        opt  = GradientNoiseOptimizer(base, eta=1e-3, gamma=0.55)
        assert opt._step_count == 0
        for _ in range(3):
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            opt.step()
        assert opt._step_count == 3

    def test_sigma_decreases_over_steps(self):
        model, x, y = _simple_model_and_loss()
        base = SGD(model.parameters(), lr=1e-3)
        opt  = GradientNoiseOptimizer(base, eta=1.0, gamma=0.55)
        sigmas = []
        for _ in range(5):
            sigmas.append(opt.current_sigma)
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            opt.step()
        # Each σ should be ≥ next σ
        for i in range(len(sigmas) - 1):
            assert sigmas[i] >= sigmas[i + 1], \
                f"Sigma should be non-increasing: σ[{i}]={sigmas[i]} > σ[{i+1}]={sigmas[i+1]}"

    def test_eta_zero_no_noise(self):
        """With eta=0 the gradients should not change due to noise."""
        model, x, y = _simple_model_and_loss()
        base = AdamW(model.parameters(), lr=1e-3)
        opt  = GradientNoiseOptimizer(base, eta=0.0, gamma=0.55)
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        pre_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        # Manually call _inject_noise to check it does nothing
        opt._inject_noise()
        for n, p in model.named_parameters():
            if p.grad is not None:
                assert torch.allclose(p.grad, pre_grads[n]), \
                    f"Gradient changed despite eta=0 for parameter {n}"

    def test_invalid_eta_raises(self):
        model, _, _ = _simple_model_and_loss()
        base = SGD(model.parameters(), lr=1e-3)
        with pytest.raises(ValueError):
            GradientNoiseOptimizer(base, eta=-1.0)

    def test_invalid_gamma_raises(self):
        model, _, _ = _simple_model_and_loss()
        base = SGD(model.parameters(), lr=1e-3)
        with pytest.raises(ValueError):
            GradientNoiseOptimizer(base, gamma=-0.1)

    def test_invalid_base_optimizer_raises(self):
        with pytest.raises(TypeError):
            GradientNoiseOptimizer("not_an_optimizer")

    def test_state_dict_roundtrip(self):
        model, x, y = _simple_model_and_loss()
        base = SGD(model.parameters(), lr=1e-3)
        opt  = GradientNoiseOptimizer(base, eta=1e-3, gamma=0.55)
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        opt.step()

        sd = opt.state_dict()
        assert sd["step_count"] == 1
        assert sd["eta"]   == 1e-3
        assert sd["gamma"] == 0.55

        # Reload
        model2, _, _ = _simple_model_and_loss()
        base2 = SGD(model2.parameters(), lr=1e-3)
        opt2  = GradientNoiseOptimizer(base2, eta=1e-3, gamma=0.55)
        opt2.load_state_dict(sd)
        assert opt2._step_count == 1

    def test_zero_grad_works(self):
        model, x, y = _simple_model_and_loss()
        base = SGD(model.parameters(), lr=1e-3)
        opt  = GradientNoiseOptimizer(base)
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        opt.zero_grad()
        for p in model.parameters():
            assert p.grad is None or (p.grad == 0).all()


# ---------------------------------------------------------------------------
# AdaptiveGradientNoiseOptimizer tests
# ---------------------------------------------------------------------------

class TestAdaptiveGradientNoiseOptimizer:

    def test_step_executes(self):
        model, x, y = _simple_model_and_loss()
        base = AdamW(model.parameters(), lr=2e-5)
        opt  = AdaptiveGradientNoiseOptimizer(base, eta=1e-2, gamma=0.55)
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        opt.step()

    def test_step_count_increments(self):
        model, x, y = _simple_model_and_loss()
        base = SGD(model.parameters(), lr=1e-3)
        opt  = AdaptiveGradientNoiseOptimizer(base, eta=1e-3)
        for _ in range(4):
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            opt.step()
        assert opt._step_count == 4

    def test_inherits_from_gn(self):
        assert issubclass(AdaptiveGradientNoiseOptimizer, GradientNoiseOptimizer)
