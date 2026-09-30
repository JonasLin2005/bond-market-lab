"""Refit a Nelson-Siegel curve to bond quotes and score it against the true curve.

The fit minimises price errors divided by each bond's dollar duration, which
makes each residual approximately a yield error, in bp. The decay ``tau`` is
chosen by a grid search with the betas solved by least squares (the Diebold-Li
shortcut), and that answer seeds a full nonlinear fit of all four parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

from .bonds import Bond
from .curve import NelsonSiegel, ns_loadings

TAU_GRID = np.linspace(0.3, 8.0, 40)
REPORT_TENORS = np.linspace(0.5, 30.0, 60)


@dataclass
class FitResult:
    curve: NelsonSiegel
    residual_bp: np.ndarray      # model yield minus quoted yield, per bond
    codes: list

    @property
    def rmse_bp(self) -> float:
        return float(np.sqrt(np.mean(self.residual_bp**2)))


def fit_curve(bonds: Sequence[Bond], t: float, yields: np.ndarray,
              spread: Optional[Callable] = None, tau_grid=TAU_GRID,
              tau: Optional[float] = None) -> FitResult:
    """Fit to yields to maturity (decimals) of ``bonds`` observed at time ``t`` (years).

    With ``tau`` given, only the three betas are fitted.
    """
    bonds = list(bonds)
    yields = np.asarray(yields, float)
    if len(bonds) < 4:
        raise ValueError("need at least 4 bonds to fit 4 parameters")
    dirty = np.array([b.dirty_from_yield(t, y) for b, y in zip(bonds, yields)])
    dd = np.array([b.modified_duration(t, y) for b, y in zip(bonds, yields)]) * dirty
    ttm = np.array([float(b.remaining_years(t)) for b in bonds])

    # Stage 1: tau grid, betas by OLS on yields at remaining maturity.
    best = None
    for tg in ([tau] if tau is not None else tau_grid):
        X = ns_loadings(ttm, tg)
        beta, *_ = np.linalg.lstsq(X, yields, rcond=None)
        sse = float(np.sum((X @ beta - yields) ** 2))
        if best is None or sse < best[0]:
            best = (sse, beta, tg)
    _, beta0, tau0 = best

    def resid(p):
        betas = p[:3]
        tau_ = np.exp(p[3]) if tau is None else tau
        model = np.array([b.dirty_from_curve(t, betas, tau_, spread) for b in bonds])
        return (model - dirty) / dd * 1e4

    if tau is None:
        x0, lo, hi, xs = np.r_[beta0, np.log(tau0)], [-1, -1, -1, np.log(0.05)], [1, 1, 1, np.log(30.0)], [0.01] * 3 + [1.0]
    else:
        x0, lo, hi, xs = beta0, [-1, -1, -1], [1, 1, 1], [0.01] * 3
    sol = least_squares(resid, x0, bounds=(lo, hi), x_scale=xs, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    fitted_tau = float(np.exp(sol.x[3])) if tau is None else float(tau)
    return FitResult(NelsonSiegel(*sol.x[:3], fitted_tau), sol.fun.copy(), [b.code for b in bonds])


def curve_error_bp(fitted: NelsonSiegel, true: NelsonSiegel, tenors=REPORT_TENORS) -> np.ndarray:
    """Fitted minus true zero rate at ``tenors``, in bp."""
    return (fitted.zero(tenors) - true.zero(tenors)) * 1e4
