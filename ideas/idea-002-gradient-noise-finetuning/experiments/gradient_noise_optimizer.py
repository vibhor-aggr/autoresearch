"""
gradient_noise_optimizer.py – Gradient-noise optimizer wrapper for PyTorch.

Wraps any base optimizer and injects calibrated Gaussian noise into gradients
before each parameter update, following the annealed schedule of
Neelakantan et al. (2015) adapted for transformer fine-tuning.

Two variants:
  - GradientNoiseOptimizer     : fixed noise scale (σ_t = η / (1 + t)^γ)
  - AdaptiveGradientNoiseOptimizer : norm-adaptive (σ_t = η * ‖g_t‖ / (1 + t)^γ)
"""

import torch
from torch.optim import Optimizer


class GradientNoiseOptimizer(Optimizer):
    """
    Wraps a base optimizer; injects isotropic Gaussian noise into gradients.

    Noise schedule: σ_t = eta / (1 + t)^gamma

    Args:
        base_optimizer: an already-constructed PyTorch Optimizer (e.g. AdamW)
        eta:   initial noise standard deviation (default: 1e-3)
        gamma: noise decay rate (default: 0.55)
    """

    def __init__(self, base_optimizer: Optimizer, eta: float = 1e-3, gamma: float = 0.55):
        if not isinstance(base_optimizer, Optimizer):
            raise TypeError(f"base_optimizer must be a PyTorch Optimizer, got {type(base_optimizer)}")
        if eta < 0:
            raise ValueError(f"eta must be non-negative, got {eta}")
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")

        self._base_optimizer = base_optimizer
        self.eta   = eta
        self.gamma = gamma
        self._step_count = 0

        # Expose param_groups so schedulers etc. can find them
        self.param_groups = base_optimizer.param_groups
        self.state        = base_optimizer.state
        self.defaults     = base_optimizer.defaults

    # ------------------------------------------------------------------
    # Delegate attribute access to base optimizer for compatibility
    # ------------------------------------------------------------------

    def zero_grad(self, set_to_none: bool = True):
        self._base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "base": self._base_optimizer.state_dict(),
            "step_count": self._step_count,
            "eta": self.eta,
            "gamma": self.gamma,
        }

    def load_state_dict(self, state_dict: dict):
        self._base_optimizer.load_state_dict(state_dict["base"])
        self._step_count = state_dict.get("step_count", 0)
        self.eta         = state_dict.get("eta",   self.eta)
        self.gamma       = state_dict.get("gamma", self.gamma)
        self.param_groups = self._base_optimizer.param_groups
        self.state        = self._base_optimizer.state

    # ------------------------------------------------------------------
    # Core: inject noise then delegate to base optimizer
    # ------------------------------------------------------------------

    def _noise_std(self) -> float:
        """Compute σ_t = eta / (1 + t)^gamma."""
        return self.eta / ((1.0 + self._step_count) ** self.gamma)

    def _inject_noise(self):
        """Add Gaussian noise to all parameter gradients in-place."""
        sigma = self._noise_std()
        if sigma == 0.0:
            return
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                noise = torch.randn_like(p.grad) * sigma
                p.grad.data.add_(noise)

    def step(self, closure=None):
        self._inject_noise()
        loss = self._base_optimizer.step(closure)
        self._step_count += 1
        return loss

    @property
    def current_sigma(self) -> float:
        return self._noise_std()


class AdaptiveGradientNoiseOptimizer(GradientNoiseOptimizer):
    """
    Norm-adaptive variant: σ_t = eta * ‖g_t‖_2 / (1 + t)^gamma

    The noise scale is proportional to the current gradient norm, so the
    signal-to-noise ratio remains constant across training steps.
    """

    def _inject_noise(self):
        # Compute global gradient L2 norm
        total_norm_sq = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    total_norm_sq += p.grad.data.norm(2).item() ** 2
        grad_norm = total_norm_sq ** 0.5

        sigma = self.eta * grad_norm / ((1.0 + self._step_count) ** self.gamma)
        if sigma == 0.0:
            return
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                noise = torch.randn_like(p.grad) * sigma
                p.grad.data.add_(noise)
