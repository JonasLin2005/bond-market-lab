"""Fixed-coupon bonds: cash flows, accrued interest, price <-> yield, risk.

Conventions
-----------
* Time is measured in years since ``EPOCH`` on an ACT/365F basis; ``to_years``
  converts timestamps. Curve discounting uses this clock.
* Prices are per 100 face. Clean = dirty - accrued.
* Accrued interest is linear within the coupon period (actual/actual by period).
* Yield to maturity is compounded at the coupon frequency, with the first
  discount exponent equal to the fraction of the current period left
  (the usual street convention). Yields are decimals; the API layer converts
  to percent, which is how the market quotes them.

Every pricing function is vectorised over the valuation time ``t`` so a whole
day of quote events can be priced in one call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd

from .curve import zero_matrix

EPOCH = pd.Timestamp("2000-01-01")
_YEAR = pd.Timedelta(days=365)


def to_years(ts) -> np.ndarray:
    """Timestamp(s) -> years since EPOCH (ACT/365F)."""
    ts = pd.to_datetime(ts)
    if isinstance(ts, pd.Timestamp):
        return np.float64((ts - EPOCH) / _YEAR)
    return np.asarray((pd.DatetimeIndex(ts) - EPOCH) / _YEAR, dtype=float)


def _out(x, scalar: bool):
    return float(x[0]) if scalar else x


@dataclass(frozen=True)
class Bond:
    code: str
    sector: str          # treasury, policy_bank, local_gov, urban_invest, bank_capital
    coupon: float        # annual coupon rate, decimal
    issue: pd.Timestamp
    maturity: pd.Timestamp
    freq: int = 1        # coupons per year
    name: str = field(default="", compare=False)
    issuer: str = field(default="", compare=False)

    def __post_init__(self):
        object.__setattr__(self, "issue", pd.Timestamp(self.issue))
        object.__setattr__(self, "maturity", pd.Timestamp(self.maturity))
        if self.freq not in (1, 2, 4):
            raise ValueError("freq must be 1, 2 or 4")
        if self.maturity <= self.issue:
            raise ValueError("maturity must be after issue")

    # ---- schedule -------------------------------------------------------
    @cached_property
    def schedule(self) -> pd.DatetimeIndex:
        """Period boundaries from the start of the first period to maturity.

        Rolled back from maturity to the last roll date on or before issue.
        The simulated book issues on roll dates, so every period is regular;
        odd first coupons are out of scope.
        """
        step = pd.DateOffset(months=12 // self.freq)
        dates = [self.maturity]
        k = 1
        while dates[-1] > self.issue:
            dates.append(self.maturity - step * k)
            k += 1
        return pd.DatetimeIndex(dates[::-1])

    @cached_property
    def _sched_t(self) -> np.ndarray:
        return to_years(self.schedule)

    @cached_property
    def cashflows(self) -> np.ndarray:
        """Amount paid at each schedule date after the first (per 100 face)."""
        cf = np.full(len(self.schedule) - 1, 100.0 * self.coupon / self.freq)
        cf[-1] += 100.0
        return cf

    @property
    def payment_dates(self) -> pd.DatetimeIndex:
        return self.schedule[1:]

    def remaining_years(self, t) -> np.ndarray:
        return self._sched_t[-1] - np.asarray(t, dtype=float)

    # ---- period arithmetic ---------------------------------------------
    def _state(self, t: np.ndarray):
        s = self._sched_t
        if np.any(t < s[0]) or np.any(t >= s[-1]):
            raise ValueError(f"{self.code}: valuation time outside the bond's life")
        nxt = np.searchsorted(s, t, side="right")          # index of next payment
        w = (s[nxt] - t) / (s[nxt] - s[nxt - 1])          # fraction of period left
        j = np.arange(1, len(s))                           # payment indices
        live = j[None, :] >= nxt[:, None]
        expo = np.where(live, w[:, None] + (j[None, :] - nxt[:, None]), 0.0)
        return w, live, expo

    def accrued(self, t):
        scalar = np.ndim(t) == 0
        t = np.atleast_1d(np.asarray(t, dtype=float))
        w, _, _ = self._state(t)
        return _out(100.0 * self.coupon / self.freq * (1.0 - w), scalar)

    # ---- yield space ----------------------------------------------------
    def _pv(self, t, y, order=0):
        _, live, expo = self._state(t)
        f = self.freq
        base = 1.0 + y[:, None] / f
        if order == 0:
            terms = self.cashflows * base ** (-expo)
        elif order == 1:
            terms = -expo / f * self.cashflows * base ** (-expo - 1)
        else:
            terms = expo * (expo + 1) / f**2 * self.cashflows * base ** (-expo - 2)
        return np.where(live, terms, 0.0).sum(axis=1)

    def dirty_from_yield(self, t, y):
        scalar = np.ndim(t) == 0 and np.ndim(y) == 0
        t, y = np.broadcast_arrays(np.atleast_1d(np.asarray(t, float)), np.atleast_1d(np.asarray(y, float)))
        return _out(self._pv(t, y), scalar)

    def clean_from_yield(self, t, y):
        scalar = np.ndim(t) == 0 and np.ndim(y) == 0
        t, y = np.broadcast_arrays(np.atleast_1d(np.asarray(t, float)), np.atleast_1d(np.asarray(y, float)))
        return _out(self._pv(t, y) - self.accrued(t), scalar)

    def yield_from_dirty(self, t, dirty, tol=1e-13, max_iter=50):
        """Yield to maturity by vectorised Newton iteration."""
        scalar = np.ndim(t) == 0 and np.ndim(dirty) == 0
        t, dirty = np.broadcast_arrays(np.atleast_1d(np.asarray(t, float)),
                                       np.atleast_1d(np.asarray(dirty, float)))
        y = np.full(t.shape, self.coupon, dtype=float)
        for _ in range(max_iter):
            step = (self._pv(t, y) - dirty) / self._pv(t, y, order=1)
            y = np.maximum(y - step, -0.5 * self.freq)
            if np.all(np.abs(step) < tol):
                break
        else:  # pragma: no cover - never observed with market-like inputs
            raise RuntimeError(f"{self.code}: yield did not converge")
        return _out(y, scalar)

    def yield_from_clean(self, t, clean):
        t = np.asarray(t, float)
        return self.yield_from_dirty(t, np.asarray(clean, float) + self.accrued(t))

    def modified_duration(self, t, y):
        scalar = np.ndim(t) == 0 and np.ndim(y) == 0
        t, y = np.broadcast_arrays(np.atleast_1d(np.asarray(t, float)), np.atleast_1d(np.asarray(y, float)))
        return _out(-self._pv(t, y, 1) / self._pv(t, y), scalar)

    def convexity(self, t, y):
        scalar = np.ndim(t) == 0 and np.ndim(y) == 0
        t, y = np.broadcast_arrays(np.atleast_1d(np.asarray(t, float)), np.atleast_1d(np.asarray(y, float)))
        return _out(self._pv(t, y, 2) / self._pv(t, y), scalar)

    def dv01(self, t, y):
        """Price change per 100 face for a 1bp fall in yield."""
        return self.modified_duration(t, y) * self.dirty_from_yield(t, y) * 1e-4

    # ---- curve space ----------------------------------------------------
    def dirty_from_curve(self, t, betas, tau, spread=None):
        """Dirty price discounting each cash flow on a Nelson-Siegel zero curve.

        ``betas`` is (3,) for one curve or (n, 3) for one curve per valuation
        time. ``spread`` is an optional function of time-to-cash-flow (years)
        returning a continuously compounded spread added to the zero rate.
        """
        scalar = np.ndim(t) == 0 and np.ndim(betas) == 1
        t = np.atleast_1d(np.asarray(t, float))
        betas = np.atleast_2d(np.asarray(betas, float))
        t, _ = np.broadcast_arrays(t, betas[:, 0])
        betas = np.broadcast_to(betas, (len(t), 3))
        _, live, _ = self._state(t)
        ttm = np.maximum(self._sched_t[1:][None, :] - t[:, None], 1e-10)
        z = zero_matrix(betas, tau, ttm)
        if spread is not None:
            z = z + spread(ttm)
        pv = np.where(live, self.cashflows * np.exp(-z * ttm), 0.0).sum(axis=1)
        return _out(pv, scalar)
