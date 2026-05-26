#!/usr/bin/env python3
"""Experiment driver: compare fixed vs adaptive NS coefficients."""

import numpy as np
import matplotlib.pyplot as plt
import torch

from muon.train import generate_random_function, train_one
from muon.coeff_search import STANDARD_NS, JORDAN_NS, find_fast_coeffs
from muon.adaptive_optimizer import AdaptiveMuon


def make_fixed_opt(ns_coeffs):
    """Create a non-adaptive Muon optimizer with fixed coefficients."""
    a, b, c = ns_coeffs

    def factory(model):
        table = {a: (a, b, c)}  # only one entry → never changes
        from muon.adaptive_optimizer import AdaptiveMuon
        return AdaptiveMuon(model, a_init=a, coeff_table=table)

    return factory


# ── Configurations ──
k_max = 10
n_funcs = 8
epochs = 2000

print(f'Adaptive NS Comparison (K_max={k_max}, {n_funcs} functions)')
print(f'{"Method":<30s} {"final":>10s} {"min":>10s} {"spikes":>8s}')
print('-' * 62)

for name, opt_factory in [
    ('Standard NS (a=1.5, fixed)',
     make_fixed_opt(STANDARD_NS)),
    ('Jordan (a=3.44, fixed)',
     make_fixed_opt(JORDAN_NS)),
    # After implementing find_fast_coeffs, uncomment:
    # fast = find_fast_coeffs()
    # ('FastNS (your best, fixed)', make_fixed_opt(fast)),
    ('Adaptive (a_init=1.5)',
     lambda m: AdaptiveMuon(m, a_init=1.5)),
]:
    finals, mins, spikes_l = [], [], []
    for func_idx in range(n_funcs):
        target_fn = generate_random_function(func_idx * 100 + 42, k_max)
        arr = train_one(target_fn, opt_factory, epochs=epochs,
                        seed=func_idx * 10 + 42)
        rm = np.minimum.accumulate(arr)
        finals.append(arr[-1])
        mins.append(arr.min())
        spikes_l.append(int(np.sum(arr[100:] > 3 * rm[100:])))

    print(f'{name:<30s} {np.mean(finals):>10.2e} {np.mean(mins):>10.2e} '
          f'{np.mean(spikes_l):>8.1f}')

# ── Plot one example ──
target_fn = generate_random_function(42, k_max)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

def moving_average(arr, window=100):
    alpha = 2.0 / (window + 1)
    result = np.zeros_like(arr)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result

colors = {'Standard NS': 'blue', 'Jordan': 'red',
          'FastNS': '#2ca02c', 'Adaptive': 'purple'}

for name, factory in [
    ('Standard NS', make_fixed_opt(STANDARD_NS)),
    ('Jordan', make_fixed_opt(JORDAN_NS)),
    # After implementing find_fast_coeffs, uncomment:
    # ('FastNS', make_fixed_opt(find_fast_coeffs())),
    ('Adaptive', lambda m: AdaptiveMuon(m, a_init=1.5)),
]:
    arr = train_one(target_fn, factory, epochs=epochs, seed=42)
    smoothed = moving_average(arr, window=100)
    # Raw loss (faded)
    ax1.semilogy(arr, color=colors[name], lw=0.3, alpha=0.3)
    ax2.semilogy(arr[-500:], color=colors[name], lw=0.3, alpha=0.3)
    # Smoothed (bold)
    ax1.semilogy(smoothed, color=colors[name], lw=1.5, alpha=0.9, label=name)
    ax2.semilogy(smoothed[-500:], color=colors[name], lw=1.5, alpha=0.9)

ax1.set_xlabel('Epoch'); ax1.set_ylabel('Train MSE')
ax1.set_title(f'Loss (faded: raw, bold: smoothed, K_max={k_max})')
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Epoch'); ax2.set_ylabel('Train MSE')
ax2.set_title('Zoom: last 500 epochs')
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('adaptive_ns_comparison.png', dpi=200)
plt.close()
print('\nSaved adaptive_ns_comparison.png')
