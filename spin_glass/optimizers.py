"""
Optimizers for Spin Glass on the Sphere
========================================
Each optimizer takes a SpinGlass model + initial sigma and returns
the final energy (H/N) and the final sigma.

All optimizers respect the spherical constraint ||sigma||^2 = N.

To add your own optimizer, define a function with signature:
    my_optimizer(model, sigma0, **kwargs) -> (final_energy, final_sigma)
"""
import numpy as np

# ── Gradient Descent ────────────────────────────────────────────────
def gd(model, sigma0, lr=0.02, steps=500, record_every=None):
    """Plain gradient descent on the sphere."""
    s = model.project(sigma0)
    for _ in range(steps):
        s = model.project(s - lr * model.gradient(s))
    return model.energy(s), s


# ── GD with Momentum (Heavy Ball) ───────────────────────────────────
def momentum(model, sigma0, lr=0.02, beta=0.9, steps=500):
    """Gradient descent with Polyak momentum."""
    s = model.project(sigma0)
    v = np.zeros(model.N)
    for _ in range(steps):
        v = beta * v + lr * model.gradient(s)
        s = model.project(s - v)
    return model.energy(s), s


# ── Registry ───────────────────────────────────────────────────────
ALL_OPTIMIZERS = {
    'gd': gd,
    'momentum': momentum
}
