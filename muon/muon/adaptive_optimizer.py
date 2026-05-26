"""Adaptive Muon: self-tuning NS coefficients via loss-stability feedback.

Students implement the loss-based adaptivity logic in step().
The optimizer starts conservative (low a) and becomes aggressive (high a)
as training stabilizes.
"""

import numpy as np
import torch
from torch.optim import Optimizer

from .newton_schulz import iterate_polynomial
from .coeff_search import STANDARD_NS, JORDAN_NS


def _apply_ns(M, a, b, c, steps=5):
    """Apply Newton-Schulz iteration with given coefficients. (Provided)"""
    X = M / (M.norm() + 1e-7)
    transposed = M.size(0) > M.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    if transposed:
        X = X.T
    return X


class AdaptiveMuon(Optimizer):
    """Muon with self-tuning Newton-Schulz coefficients.

    The coefficient 'a' adapts based on training stability:
      - Spike detected (loss > 3x EMA): decrease a (conservative)
      - Stable for 100+ steps:            increase a (aggressive)

    Parameters
    ----------
    model : nn.Module
    lr_muon : float        Learning rate for matrix parameters.
    lr_adam : float        Learning rate for vector parameters.
    momentum : float       Momentum coefficient.
    a_init : float         Starting coefficient (default: standard NS).
    coeff_table : dict     Lookup table {a_target: (a, b, c)}.
    ns_steps : int         NS iterations per step.
    """

    def __init__(self, model, lr_muon=0.02, lr_adam=0.001, momentum=0.95,
                 a_init=3.0, coeff_table=None, ns_steps=5):
        matrix_params, vector_params = [], []
        for p in model.parameters():
            (matrix_params if p.dim() == 2 else vector_params).append(p)

        defaults = dict(lr_m=lr_muon, lr_a=lr_adam, mom=momentum)
        super().__init__([
            {'params': matrix_params, 'mu': True},
            {'params': vector_params, 'mu': False},
        ], defaults)

        self.a_target = a_init
        self.ns_steps = ns_steps

        # Default coefficient table if none provided
        if coeff_table is None:
            self.coeff_table = {
                1.5: STANDARD_NS, 3.44: JORDAN_NS
                # After implementing find_fast_coeffs, add your result here!
            }
        else:
            self.coeff_table = coeff_table

        # TODO: Initialize adaptivity state
        # self.loss_ema = None       # exponential moving average of loss
        # self.stable_count = 0      # consecutive stable steps
        pass

    def _get_coeffs(self, a_target):
        """Return (a,b,c) for the given target a using nearest-key lookup."""
        keys = sorted(self.coeff_table.keys())
        best_key = keys[0]
        for k in keys:
            if k <= a_target:
                best_key = k
        return self.coeff_table[best_key]

    @torch.no_grad()
    def step(self, loss_val=None):
        """Perform one optimizer step.

        Pass loss_val to enable coefficient adaptivity.

        TODO: Implement the adaptivity logic:
          1. Update loss_ema: ema = 0.99 * ema + 0.01 * loss_val
          2. If loss_val > 3.0 * ema: spike → decrease a by 0.2
          3. Else: increment stable_count.
             If stable_count > 100: increase a by 0.05
          4. Clamp a to [1.5, 4.0]
          5. Look up (a,b,c) from coeff_table using _get_coeffs
        """
        # TODO: Your code here
        # Hints:
        #   self.a_target  — current coefficient
        #   self.loss_ema  — EMA of loss (tracked internally)
        #   self.stable_count — consecutive steps without spike
        #   self._get_coeffs(a) — returns (a,b,c) triple

        # Fallback: use current a_target without adaptation
        a, b, c = self._get_coeffs(self.a_target)

        for g in self.param_groups:
            if g.get('mu'):
                for p in g['params']:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if 'b' not in st:
                        st['b'] = torch.zeros_like(p)

                    st['b'].mul_(g['mom']).add_(p.grad, alpha=1 - g['mom'])
                    M = p.grad.add(st['b'], alpha=g['mom'])
                    O = _apply_ns(M, a, b, c, steps=self.ns_steps)
                    p.sub_(O, alpha=g['lr_m'])
            else:
                for p in g['params']:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if 's' not in st:
                        st['s'] = 0
                        st['m'] = torch.zeros_like(p)
                        st['v'] = torch.zeros_like(p)
                    st['s'] += 1
                    st['m'].mul_(0.9).add_(p.grad, alpha=0.1)
                    st['v'].mul_(0.999).addcmul_(p.grad, p.grad, value=0.001)
                    m_hat = st['m'] / (1 - 0.9 ** st['s'])
                    v_hat = st['v'] / (1 - 0.999 ** st['s'])
                    p.sub_(m_hat / (v_hat.sqrt() + 1e-8), alpha=g['lr_a'])
