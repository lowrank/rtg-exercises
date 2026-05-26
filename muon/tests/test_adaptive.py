"""Tests for the adaptive optimizer."""

import numpy as np
import torch
import torch.nn as nn
import pytest

from muon.adaptive_optimizer import AdaptiveMuon
from muon.coeff_search import STANDARD_NS, JORDAN_NS


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc(x)


class TestAdaptiveMuon:
    """Tests for the adaptive optimizer (students implement the adaptivity)."""

    def test_initializes_with_standard_ns(self):
        """Default init should use standard NS (a=1.5)."""
        model = TinyModel()
        opt = AdaptiveMuon(model)
        assert opt.a_target == 1.5
        a, b, c = opt._get_coeffs(opt.a_target)
        assert abs(a - 1.5) < 0.01

    def test_loss_decreases_after_step(self):
        """Loss should decrease after one step."""
        torch.manual_seed(0)
        model = TinyModel()
        opt = AdaptiveMuon(model, a_init=3.44,
                           coeff_table={3.44: JORDAN_NS})
        x = torch.randn(8, 16)
        y = torch.randn(8, 4)

        loss_before = nn.MSELoss()(model(x), y).item()
        opt.zero_grad()
        nn.MSELoss()(model(x), y).backward()
        opt.step()
        loss_after = nn.MSELoss()(model(x), y).item()

        assert loss_after < loss_before, "Loss did not decrease"

    def test_a_stays_in_bounds(self):
        """a should stay in [1.5, 4.0]."""
        model = TinyModel()
        opt = AdaptiveMuon(model, a_init=1.5)
        for _ in range(100):
            opt.step(loss_val=0.1)
        assert 1.5 <= opt.a_target <= 4.0

    def test_a_increases_when_stable(self):
        """After many stable steps, a should increase above initial value."""
        torch.manual_seed(0)
        model = TinyModel()
        opt = AdaptiveMuon(model, a_init=1.5)

        # Feed decreasing losses (simulating stable training)
        for i in range(500):
            opt.step(loss_val=1.0 / (1 + i * 0.01))

        assert opt.a_target > 1.5, (
            f"a should increase during stable training. "
            f"Got a={opt.a_target:.2f}.  Check your loss_ema and stable_count logic."
        )

    def test_a_decreases_on_spike(self):
        """A large loss spike should decrease a."""
        torch.manual_seed(0)
        model = TinyModel()
        opt = AdaptiveMuon(model, a_init=1.5)

        # Build up stable history
        for _ in range(200):
            opt.step(loss_val=0.1)

        a_before = opt.a_target

        # Inject a spike
        opt.step(loss_val=100.0)

        assert opt.a_target < a_before, (
            f"a should decrease after a spike. "
            f"Before: {a_before:.2f}, After: {opt.a_target:.2f}"
        )
