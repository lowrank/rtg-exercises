"""Coefficient search: find stable (a,b,c) triples for Newton-Schulz.

Students implement:
  1. search_coefficients  — random search maximizing 'a'
  2. find_fast_coeffs     — find the fastest stable triple (the "FastNS")
  3. build_coeff_table    — build a lookup table for adaptive scheduling

Reference: only STANDARD_NS and JORDAN_NS are provided as known coefficient sets.
"""

import numpy as np
from .newton_schulz import is_stable, orbit_max


# ── Known reference coefficient sets (provided) ──

STANDARD_NS = (1.500, -0.500, 0.000)
JORDAN_NS   = (3.445, -4.775, 2.032)


# ── Tasks ──

def search_coefficients(n_samples=50000, a_range=(1.5, 5.0),
                        b_range=(-7.5, 0.0), c_range=(0.1, 4.0)):
    """Task 1: Random search for stable (a,b,c) that maximizes 'a'.

    For each randomly sampled triple, call is_stable(a,b,c).
    Return the one with the largest 'a' that passes.

    Parameters
    ----------
    n_samples : int        Number of random triples to test.
    a_range, b_range, c_range : tuple   Search ranges.

    Returns
    -------
    best : tuple (a, b, c)
        The stable triple with the largest a found.
    """
    # TODO: Implement random search.
    # 1. Initialize best = STANDARD_NS as fallback
    # 2. Loop n_samples times: sample a,b,c uniformly, check is_stable, keep best
    pass


def find_fast_coeffs(n_samples=200000):
    """Task 2: Find the "FastNS" — the fastest stable coefficient triple.

    Uses a wider search than search_coefficients to find the maximum
    stable 'a'.  The result should have a > 3.4 (faster than Jordan).

    Hint: search_coefficients with enough samples should find it.
    But you can also try a targeted search: fix a, optimize (b,c).

    Returns
    -------
    fast : tuple (a, b, c)
        The fastest stable triple found.
    """
    # TODO: Find the fastest stable coefficients.
    # Try: search_coefficients with large n_samples, OR
    #      for each a in [3.0, 3.2, ..., 4.5], search best (b,c)
    pass


def build_coeff_table(a_values):
    """Task 3: Build a lookup table mapping target-a to best stable (a,b,c).

    For each target a in a_values, find the best stable (b,c) near that a
    using a focused search.

    Parameters
    ----------
    a_values : list of float
        Target a values (e.g., [1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

    Returns
    -------
    table : dict
        {a_target: (a, b, c)} mapping each target to its best stable triple.
    """
    # TODO: Build a lookup table.
    # For each target a, call search_coefficients with a narrow a_range.
    pass
