"""
Spin Glass Experiment Runner
==============================
Benchmark optimizers and plot results.

Usage:
    python experiment.py                    # run all benchmarks
    python experiment.py --n-runs 500       # custom number of runs
    python experiment.py --optimizer gd     # single optimizer
"""
import numpy as np
import time
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spin_glass_model import SpinGlass
from optimizers import ALL_OPTIMIZERS


def benchmark(model, optimizer_fn, n_runs=100, **opt_kwargs):
    """Run optimizer from `n_runs` random initializations.
    Returns array of final energies (H/N)."""
    energies = []
    for i in range(n_runs):
        s0 = model.random_init(seed=1000 + i)
        e, _ = optimizer_fn(model, s0, **opt_kwargs)
        energies.append(e)
    return np.array(energies)


def run_all(model, optimizers=None, n_runs=100, defaults=True):
    """Benchmark multiple optimizers. Returns dict of results."""
    if optimizers is None:
        optimizers = ALL_OPTIMIZERS

    results = {}
    for name, fn in optimizers.items():
        t0 = time.time()
        energies = benchmark(model, fn, n_runs=n_runs)
        elapsed = time.time() - t0
        results[name] = {
            'energies': energies,
            'mean': energies.mean(),
            'std': energies.std(),
            'min': energies.min(),
            'max': energies.max(),
            'unique': len(set(round(e, 5) for e in energies)),
            'time': elapsed,
        }
        print(f"  {name:15s}: mean={energies.mean():.4f}  "
              f"std={energies.std():.4f}  "
              f"min={energies.min():.4f}  "
              f"unique={results[name]['unique']}/{n_runs}  "
              f"{elapsed:.1f}s")
    return results


def plot_comparison(results, p=3, save_path='optimizer_comparison.png'):
    """Plot histograms comparing optimizers."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'gd': 'steelblue', 'momentum': 'darkorange', 'adam': 'green',
              'langevin': 'purple', 'annealing': 'crimson', 'newton': 'brown'}

    Ec = 2 * np.sqrt((p-1)/p)
    n = len(results)
    fig, axes = plt.subplots(2, max(3, (n+1)//2), figsize=(5*max(3,(n+1)//2), 9))

    # Combined KDE
    ax = axes[0, 0]
    x_all = np.linspace(-2.0, 0.5, 300)
    for name, data in results.items():
        e = data['energies']
        try:
            from scipy import stats
            kde = stats.gaussian_kde(e)
            ax.plot(x_all, kde(x_all), color=colors.get(name, 'gray'),
                    lw=2, label=f"{name} (min={e.min():.3f})")
        except Exception:
            ax.hist(e, bins=30, density=True, alpha=0.3,
                    color=colors.get(name, 'gray'), label=name)
    ax.axvline(-Ec, color='red', linestyle='--', lw=2,
               label=f'$-E_c = -2\\sqrt{{(p-1)/p}} = {-Ec:.3f}$')
    ax.set_xlabel('$H/N$'); ax.set_ylabel('Density')
    ax.set_title('KDE: All Optimizers'); ax.legend(fontsize=7)

    # Individual histograms
    for idx, (name, data) in enumerate(results.items()):
        if idx + 1 >= axes.size:
            break
        row = (idx + 1) // axes.shape[1]
        col = (idx + 1) % axes.shape[1]
        ax = axes[row, col]
        e = data['energies']
        ax.hist(e, bins=25, color=colors.get(name, 'steelblue'),
                edgecolor='white', alpha=0.8, density=True)
        ax.axvline(data['mean'], color='blue', linestyle=':', lw=2,
                   label=f"mean={data['mean']:.4f}")
        ax.axvline(data['min'], color='darkred', linestyle='-', lw=1,
                   label=f"best={data['min']:.4f}")
        ax.set_xlabel('$H/N$')
        ax.set_title(f"{name} ({data['unique']} unique)")
        ax.legend(fontsize=7)

    # Hide unused axes
    for idx in range(len(results) + 1, axes.size):
        row = idx // axes.shape[1]
        col = idx % axes.shape[1]
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved to {save_path}")


def main():
    p = argparse.ArgumentParser(description="Spin glass optimizer benchmark")
    p.add_argument("--N", type=int, default=50)
    p.add_argument("--p", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-runs", type=int, default=200)
    p.add_argument("--optimizer", type=str, default=None,
                   help="Run single optimizer (gd, momentum, ...)")
    p.add_argument("--save", type=str, default='optimizer_comparison.png')
    args = p.parse_args()

    print(f"Spin Glass: N={args.N}, p={args.p}, seed={args.seed}, runs={args.n_runs}s")
    model = SpinGlass(N=args.N, p=args.p, seed=args.seed)
    print(f"Running {args.n_runs} initializations per optimizer...\n")

    if args.optimizer:
        optimizers = {args.optimizer: ALL_OPTIMIZERS[args.optimizer]}
    else:
        optimizers = ALL_OPTIMIZERS

    results = run_all(model, optimizers=optimizers, n_runs=args.n_runs)

    if len(results) > 1:
        plot_comparison(results, p=args.p, save_path=args.save)


if __name__ == '__main__':
    main()
