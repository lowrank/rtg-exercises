"""
p-Spin Spherical Spin Glass Model (ABC Normalization)
=======================================================
Auffinger, Ben Arous & Černý (2013), "Random Matrices and Complexity of
Spin Glasses", Comm. Pure Appl. Math. 66(2), 165-201.

Hamiltonian (Equation 2.2):
  H_{N,p}(sigma) = N^{-(p-1)/2} * sum_{i1,...,ip=1}^N J_{i1...ip} sigma_{i1} ... sigma_{ip}

  - J_{i1...ip} ~ N(0, 1) i.i.d. standard Gaussian
  - Sum over ALL ordered p-tuples (N^p terms)
  - Constraint: (1/N) sum sigma_i^2 = 1, i.e. ||sigma||^2 = N
  - NO minus sign and NO 1/p! factor

Key quantities (p=3):
  E_inf = -2 * sqrt((p-1)/p) = -2*sqrt(2/3) ≈ -1.6330  (threshold)
  E_0   = root of Theta_{0,p}(u) = 0  (ground state, numerical)

Usage:
    model = SpinGlass(N=200, p=3, seed=42)
    e = model.energy(sigma)   # H/N (energy per spin, O(1))
    g = model.gradient(sigma) # tangent-projected gradient
    H = model.hessian(sigma)  # full Hessian matrix
"""
import numpy as np
import math


class SpinGlass:
    def __init__(self, N=200, p=3, seed=42):
        self.N = N
        self.p = p
        rng = np.random.RandomState(seed)
        # J ~ N(0, 1) standard Gaussian, full N^p tensor
        self.J = rng.randn(*([N]*p))
        self._scale = N**(-(p-1)/2)  # N^{-(p-1)/2} prefactor

    def energy(self, sigma):
        """H(sigma) / N — energy per spin."""
        p = self.p
        if p == 2:
            raw = self._scale * (sigma @ self.J @ sigma)
        elif p == 3:
            raw = self._scale * np.einsum('ijk,i,j,k', self.J, sigma, sigma, sigma)
        else:
            raw = self._scale * np.einsum(
                ''.join(chr(105+k) for k in range(p)),
                self.J,
                *([sigma]*p)
            )
        return raw / self.N

    def gradient(self, sigma):
        """Gradient of H, projected to tangent space of sphere.

        J has independent entries for each ordered tuple. Derivative w.r.t.
        sigma_i picks up contributions from all p positions where sigma_i
        appears in the product, using different J entries each time.
        """
        N, p, sc = self.N, self.p, self._scale
        if p == 2:
            g_raw = sc * (self.J @ sigma + self.J.T @ sigma)
        elif p == 3:
            g_raw = sc * (
                np.einsum('ijk,j,k->i', self.J, sigma, sigma) +   # i = 1st index
                np.einsum('ijk,i,k->j', self.J, sigma, sigma) +   # i = 2nd index
                np.einsum('ijk,i,j->k', self.J, sigma, sigma)     # i = 3rd index
            )
        else:
            # General p: sum over which position sigma_i occupies
            idx_all = ''.join(chr(105+k) for k in range(p))
            g_raw = np.zeros(N)
            for pos in range(p):
                sum_idx = list(idx_all)
                sum_idx.pop(pos)
                out_idx = idx_all[pos]
                g_raw += sc * np.einsum(
                    f'{idx_all},{",".join(sum_idx)}->{out_idx}',
                    self.J, *([sigma]*(p-1))
                )
        return g_raw - np.dot(g_raw, sigma) / N * sigma

    def hessian(self, sigma):
        """Full Hessian matrix at sigma.

        Second derivative picks up terms where sigma appears at two
        different positions (i,j with i != j) in the product.
        """
        N, p, sc = self.N, self.p, self._scale
        if p == 3:
            H = sc * (
                np.einsum('ijk,k->ij', self.J, sigma) +   # ∂_i ∂_j: 1st,2nd indices
                np.einsum('ijk,k->ji', self.J, sigma) +   # symmetric: transpose
                np.einsum('ijk,j->ik', self.J, sigma) +   # ∂_i ∂_k: 1st,3rd
                np.einsum('ijk,j->ki', self.J, sigma) +   # transpose
                np.einsum('ijk,i->jk', self.J, sigma) +   # ∂_j ∂_k: 2nd,3rd
                np.einsum('ijk,i->kj', self.J, sigma)     # transpose
            )
        elif p == 2:
            H = sc * (self.J + self.J.T)
        else:
            H = np.zeros((N, N))
            idx_all = ''.join(chr(105+k) for k in range(p))
            for pos_i in range(p):
                for pos_j in range(p):
                    if pos_i == pos_j:
                        continue
                    remaining = list(idx_all)
                    remaining.pop(max(pos_i, pos_j))
                    remaining.pop(min(pos_i, pos_j))
                    # The Hessian entry H_{a,b} comes from terms where
                    # sigma appears at positions pos_i and pos_j
                    out_i, out_j = idx_all[pos_i], idx_all[pos_j]
                    sum_idx = ''.join(remaining)
                    H += sc * np.einsum(
                        f'{idx_all},{sum_idx}->{out_i}{out_j}',
                        self.J, *([sigma]*(p-2))
                    )
        return H

    def random_init(self, seed=None):
        """Random point on sphere ||sigma||^2 = N."""
        rng = np.random.RandomState(seed) if seed is not None else np.random
        s = rng.randn(self.N)
        return s / np.linalg.norm(s) * np.sqrt(self.N)

    def project(self, sigma):
        """Project onto sphere ||sigma||^2 = N."""
        return sigma / np.linalg.norm(sigma) * np.sqrt(self.N)

    def hessian_spectrum(self, sigma, tol=1e-8):
        """Covariant Hessian eigenvalues on the sphere. Returns (evals, index).

        H_cov = P H P - lambda * P, where lambda = sigma^T grad_raw(H) / N
        is the Lagrange multiplier at the critical point.
        Drops the zero eigenvalue (radial direction), returns N-1 eigenvalues.
        """
        H = self.hessian(sigma)
        N = self.N
        # Raw gradient (not tangent-projected) for Lagrange multiplier
        p, sc = self.p, self._scale
        if p == 3:
            g_raw = sc * (
                np.einsum('ijk,j,k->i', self.J, sigma, sigma) +
                np.einsum('ijk,i,k->j', self.J, sigma, sigma) +
                np.einsum('ijk,i,j->k', self.J, sigma, sigma)
            )
        elif p == 2:
            g_raw = sc * (self.J @ sigma + self.J.T @ sigma)
        else:
            g_raw = self.gradient(sigma) + np.dot(self.gradient(sigma), sigma) * sigma / N

        lam = np.dot(g_raw, sigma) / N  # Lagrange multiplier
        P = np.eye(N) - np.outer(sigma, sigma) / N
        H_cov = P @ H @ P - lam * P

        evals = np.sort(np.linalg.eigvalsh(H_cov))
        # Drop the eigenvalue closest to 0 (radial direction)
        evals_by_mag = sorted(evals, key=abs)
        evals_tan = np.sort(evals_by_mag[1:])
        return evals_tan, np.sum(evals_tan < -tol)
