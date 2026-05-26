# Random Landscapes: The $p$-Spin Spherical Spin Glass

## 1. Objective

The $p$-spin spherical spin glass is a random non-convex function from statistical physics. Its landscape serves as a toy model for neural network loss surfaces.

**Hamiltonian** (Auffinger, Ben Arous & Černý, 2013, Eq. 2.2):

$$H_{N,p}(\sigma) = \frac{1}{N^{(p-1)/2}} \sum_{i_1,\ldots,i_p=1}^{N} J_{i_1\ldots i_p}\, \sigma_{i_1} \cdots \sigma_{i_p}$$

- Spins $\sigma \in \mathbb{R}^N$, constraint: $\frac{1}{N}\sum \sigma_i^2 = 1$ (sphere of radius $\sqrt{N}$)
- Couplings $J_{i_1\ldots i_p} \sim \mathcal{N}(0, 1)$ i.i.d. standard Gaussian
- Sum over **all ordered** $p$-tuples ($N^p$ terms) — $J_{1,2,3}$ and $J_{2,1,3}$ are independent
- At a random point: $H/N \approx 0$ (fluctuations $\sim 1/\sqrt{N}$)

---

## 2. Key Quantities ($p=3$)

$$E_c(p) = 2\sqrt{\frac{p-1}{p}} \approx 1.633 \quad \text{(threshold parameter, } N\to\infty \text{)}$$

The sequence $E_k(p)$ is strictly decreasing and converges to $E_c$:

| Threshold | Energy $H/N$ | Critical points |
|---|---|---|
| $-E_c$ | $\approx -1.633$ | Above: high-index saddles **dominate** (probabilistically) |
| $(-E_c, -E_1)$ | Below $-E_c$ | Minima + saddles of all indices |
| $\vdots$ | deeper | Index upper bound decreases |
| $(-E_0, -E_c)$ | $(\approx -1.657, -1.633)$ | **Only minima** at the bottom |

**Key**: The threshold is probabilistic, not absolute. At finite $N$, minima exist above $-E_c$ too — they're just not dominant. As $N \to \infty$, minima above $-E_c$ become exponentially rare relative to saddles.

---

## 3. Files

| File | Purpose |
|---|---|
| `spin_glass_model.py` | `SpinGlass` class: energy, gradient, covariant Hessian |
| `optimizers.py` | GD, momentum, ... (implement by yourself) |
| `experiment.py` | Benchmark runner + plotting (CLI via argparse) |
| `spin_glass.py` | Legacy standalone script |

### Quick Start

```python
from spin_glass_model import SpinGlass
from optimizers import gd

model = SpinGlass(N=80, p=3, seed=42)

# Single run
e, sigma = gd(model, model.random_init(), lr=0.005, steps=5000)
evals, neg = model.hessian_spectrum(sigma)
print(f"H/N = {e:.4f}, index = {neg}/{model.N-1}, lambda_min = {evals[0]:.4f}")
```

### Write a Custom Optimizer

```python
def my_optimizer(model, sigma0, **kwargs):
    """Must return (final_energy, final_sigma)."""
    s = model.project(sigma0)
    for _ in range(kwargs.get('steps', 500)):
        g = model.gradient(s)  # tangent-projected
        s = model.project(s - kwargs['lr'] * g)
    return model.energy(s), s
```

### CLI

```bash
python experiment.py --N 50 --n-runs 200              # all optimizers
python experiment.py --N 80 --n-runs 500 --optimizer gd   # single method
```

---

## 6. Experiments

### Experiment 1: Basic GD

```python
model = SpinGlass(N=50, p=3, seed=42)
energies = []
for i in range(100):
    e, _ = gd(model, model.random_init(1000+i), lr=0.005, steps=5000)
    energies.append(e)
print(f"mean={np.mean(energies):.3f}, best={np.min(energies):.3f}")
```

### Experiment 2: Check the Index

```python
e, s = gd(model, model.random_init(), lr=0.005, steps=5000)
evals, neg = model.hessian_spectrum(s)
print(f"index = {neg}/{model.N-1}, lambda_min = {evals[0]:.4f}")
```

### Experiment 3: Multiple Minima

```python
sigmas = [gd(model, model.random_init(5000+i), lr=0.005, steps=5000)[1] for i in range(50)]
overlaps = [abs(np.dot(sigmas[i], sigmas[j]))/model.N for i in range(50) for j in range(i)]
print(f"mean overlap = {np.mean(overlaps):.3f}")  # should be << 1
```

### Experiment 4: Compare Optimizers

```python
from experiment import run_all
from optimizers import gd, momentum, adam
results = run_all(model, {'gd': gd, 'momentum': momentum, 'adam': adam}, n_runs=100)
```

---

## 8. References

1. Auffinger, A., Ben Arous, G., & Černý, J. (2013). Random matrices and complexity of spin glasses. *Comm. Pure Appl. Math.* 66(2), 165–201.
2. Choromanska, A. et al. (2015). The Loss Surfaces of Multilayer Networks. *AISTATS*.
3. Crisanti, A. & Sommers, H.-J. (1992). The spherical $p$-spin interaction spin glass model: the statics. *Z. Phys. B* 87, 341–354.
