"""Nelson-Siegel zero curve.

Rates are continuously compounded decimals (0.02 = 2%) and maturities are in
years. The curve is

    z(t) = b0 + b1 * (1 - e^{-t/tau}) / (t/tau) + b2 * ((1 - e^{-t/tau}) / (t/tau) - e^{-t/tau})

so b0 is the long end, b0 + b1 the short end, and b2 the hump.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def ns_loadings(t, tau: float) -> np.ndarray:
    """Factor loadings of the zero rate at maturities ``t``; shape ``t.shape + (3,)``."""
    t = np.asarray(t, dtype=float)
    x = np.maximum(t, 1e-10) / tau
    slope = -np.expm1(-x) / x  # (1 - e^{-x}) / x, accurate near 0
    curv = slope - np.exp(-x)
    return np.stack([np.ones_like(x), slope, curv], axis=-1)


@dataclass(frozen=True)
class NelsonSiegel:
    b0: float
    b1: float
    b2: float
    tau: float

    @property
    def betas(self) -> np.ndarray:
        return np.array([self.b0, self.b1, self.b2])

    def zero(self, t):
        """Zero rate at maturity ``t`` (years)."""
        return ns_loadings(t, self.tau) @ self.betas

    def discount(self, t):
        t = np.asarray(t, dtype=float)
        return np.exp(-self.zero(t) * t)

    def forward(self, t):
        """Instantaneous forward rate."""
        x = np.asarray(t, dtype=float) / self.tau
        e = np.exp(-x)
        return self.b0 + self.b1 * e + self.b2 * x * e


def zero_matrix(betas: np.ndarray, tau: float, t: np.ndarray) -> np.ndarray:
    """Zero rates for many curves at once.

    ``betas`` has shape (n, 3) and ``t`` shape (n, m): row i of ``t`` is priced
    on curve i. Returns shape (n, m).
    """
    return np.einsum("nmk,nk->nm", ns_loadings(t, tau), betas)
