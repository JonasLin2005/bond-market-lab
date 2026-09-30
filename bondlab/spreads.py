"""Z-spread: the constant spread over a zero curve that reprices a bond."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .bonds import Bond
from .curve import NelsonSiegel


def price_on_curve(bond: Bond, t: float, curve: NelsonSiegel, z_spread: float = 0.0) -> float:
    return bond.dirty_from_curve(t, curve.betas, curve.tau, lambda ttm: np.full_like(ttm, z_spread))


def z_spread(bond: Bond, t: float, dirty: float, curve: NelsonSiegel) -> float:
    """Spread (decimal, continuously compounded) that makes the curve price equal ``dirty``."""
    return brentq(lambda s: price_on_curve(bond, t, curve, s) - dirty, -0.05, 0.30, xtol=1e-13)
