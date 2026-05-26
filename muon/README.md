# Newton-Schulz Coefficients for Muon

## Overview

The Muon optimizer uses Newton-Schulz (NS) iteration to orthogonalize gradient
momentum matrices. The NS iteration uses a degree-5 polynomial:

$$f(x) = ax + bx^3 + cx^5$$

acting on the singular values of the momentum matrix. Different choices of
$(a,b,c)$ produce different convergence behavior:

- **Standard NS** ($a=1.5, b=-0.5, c=0$): safe but slow
- **Jordan NS** ($a=3.44, b=-4.78, c=2.03$): fast but can oscillate
- **Adaptive**: starts conservative, becomes aggressive as training stabilizes

**Your task**: explore the coefficient space, find stable $(a,b,c)$ triples,
and implement an adaptive scheduling mechanism.

## Learning Objectives

1. Understand how NS polynomial coefficients control convergence speed
2. Implement stability constraints (5-iteration band, orbit boundedness)
3. Search for optimal coefficients via random search
4. Implement loss-stability feedback to self-tune coefficients during training
5. Observe the speed-stability Pareto frontier

## Structure

```
adaptive_ns_project/
├── README.md
├── requirements.txt
├── run.py                       ← experiment driver
├── muon/
│   ├── __init__.py
│   ├── newton_schulz.py         ← NS iteration + polynomial
│   ├── coeff_search.py          ← coefficient search (TODO)
│   ├── adaptive_optimizer.py    ← adaptive Muon (TODO)
│   └── train.py                 ← training loop (provided)
└── tests/
    ├── __init__.py
    ├── test_coeff_search.py     ← verify your coefficient search
    └── test_adaptive.py         ← verify your adaptive optimizer
```

## Task Checklist

### Task 1: Implement `is_stable()` in `muon/newton_schulz.py`

Implement the function `is_stable(a, b, c, n_pts=200)` that checks whether a
coefficient triple $(a,b,c)$ is stable under 5 NS iterations.

**Requirements:**
- For all $x \in [10^{-3}, 1]$, after 5 iterations: $0.5 \leq f^5(x) \leq 1.5$
- No intermediate orbit escape: $|f^k(x)| \leq 2$ for all $k = 1,\dots,5$
- Return `True` only if all test points pass both constraints

**Verify:**
```bash
pytest tests/test_coeff_search.py::TestIsStable -v
```

**Expected:** Standard NS fails (too slow for small $x$). Jordan is near the boundary.
A successful implementation correctly rejects divergent coefficients and
coefficients that map values outside the band.

---

### Task 2: Implement `search_coefficients()` in `muon/coeff_search.py`

Implement random search over $(a,b,c)$ space to find stable triples that
maximize $a$ (convergence speed).

**Requirements:**
- Sample $(a,b,c)$ uniformly from given ranges
- Check each with `is_stable()`
- Return the triple with the largest $a$ that passes
- Fall back to STANDARD_NS if no stable triple found (but your search should find one)

**Verify:**
```bash
pytest tests/test_coeff_search.py::TestCoeffSearch -v
```

**Expected:** Find $a > 1.5$ (beats standard NS) with 5000+ samples.

---

### Task 3: Implement `find_fast_coeffs()` in `muon/coeff_search.py`

Find the **fastest** stable coefficient triple — the "FastNS."

**Requirements:**
- The result should have $a > 3.4$ (faster than Jordan's $a = 3.44$)
- The result must pass `is_stable()`
- Try a large sample count, or a targeted search over individual $a$ values

**Verify:**
```bash
pytest tests/test_coeff_search.py::TestPolynomial::test_find_fast_coeffs_returns_stable -v
pytest tests/test_coeff_search.py::TestPolynomial::test_find_fast_coeffs_beats_jordan -v
```

**Expected:** $a \approx 3.8$--$3.9$ with appropriate $(b,c)$.

---

### Task 4: Build `build_coeff_table()` in `muon/coeff_search.py`

Build a lookup table mapping target $a$ values to their best stable $(a,b,c)$ triples.
This table is used by the adaptive optimizer to select coefficients.

**Requirements:**
- For each target $a$ in the input list, find the best stable $(b,c)$ near that $a$
- Return a dictionary `{a_target: (a, b, c)}`
- Use a narrow $a$-range in your search around each target

**Verify:** The table entries should all pass `is_stable()`.

---

### Task 5: Implement adaptivity in `muon/adaptive_optimizer.py`

Implement `AdaptiveMuon.step(loss_val)` — the loss-based coefficient scheduling.

**The algorithm:**
1. Maintain an exponential moving average of the loss: `ema = 0.99 * ema + 0.01 * loss_val`
2. If `loss_val > 3.0 * ema`: **spike detected** — decrease $a$ by 0.2 (more conservative)
3. Otherwise: increment `stable_count`. If `stable_count > 100`: increase $a$ by 0.05
4. Clamp $a$ to $[1.5, 4.0]$
5. Look up $(a,b,c)$ via the coefficient table

**Verify:**
```bash
pytest tests/test_adaptive.py -v
```

**Expected:**
- `a` increases during stable training
- `a` decreases after a loss spike
- `a` stays within bounds $[1.5, 4.0]$

---

### Task 6: Run experiments

```bash
python run.py
```

This generates `adaptive_ns_comparison.png` comparing:
- Standard NS (fixed a=1.5)
- Jordan (fixed a=3.44)
- Your FastNS (from Task 3)
- Your Adaptive optimizer (from Task 5)

**Expected observations:**
- Standard NS converges slowly but steadily
- Jordan converges fast with some variance
- FastNS converges fastest but with the most variance
- Adaptive starts slow (like standard NS) and becomes fast, combining both benefits

## Background

### The 5-iteration band constraint

After $K=5$ NS iterations with Frobenius normalization, the singular values
should land in $[0.5, 1.5]$. Values below 0.5 mean the orthogonalization is
too weak; values above 1.5 risk orbit escape (divergence).

### Orbit boundedness

The NS polynomial must not let any intermediate iterate escape. The constraint:

$$\max_{k=1,\dots,5} |f^{(k)}(x)| \leq 2 \quad \forall x \in (0, 1]$$

### Why adapt the coefficients?

The optimal NS coefficient depends on the training stage:
- **Early training**: gradients are noisy and high-rank → conservative $a$ is safer
- **Late training**: gradients are low-rank → aggressive $a$ accelerates convergence

An adaptive schedule captures both benefits without hand-tuning.

## Hints

- Start with the fixed-point constraint $a+b+c \approx 1$ (relaxed) to narrow the search
- The free parameter $a = f'(0)$ controls convergence speed for small singular values
- Jordan's coefficients ($a=3.44$) relax the fixed-point constraint to $f(1) \approx 0.7$
- The Pareto frontier spans $a \in [1.5, 3.9]$ — trade off speed vs stability
- For `search_coefficients`, try at least 50,000 random samples
- For `find_fast_coeffs`, try focusing the search: for each target $a$, optimize only $(b,c)$
- The `orbit_max` function (provided) is useful for debugging unstable coefficients

## References

- Jordan (2024): "Muon: An optimizer for hidden layers" — modded-nanogpt
- Amsel et al. (2025): "The Polar Express" — optimal NS coefficients (arXiv:2505.16932)
