"""Price a new bond the way a syndicate desk sets guidance.

1. Fit the treasury curve to the latest logged quotes.
2. Turn every quoted bond in the same sector into a z-spread over that curve.
3. Fit a spread term structure across those comparables, a + b(1 - e^{-T/5}),
   and shift it by the issuer's average residual when the issuer has bonds
   outstanding.
4. Fair yield = par coupon on the curve plus that spread. Add a new-issue
   concession and set a bookbuilding range around it.

Because the market is simulated, ``true_fair_yield`` gives the answer the
estimate should reach, so the method can be scored.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .bonds import to_years
from .curve import NelsonSiegel
from .experiment import fit_at
from .market import Market
from .spreads import z_spread
from .store import run_query

CONCESSION_BP = {"treasury": 0.0, "policy_bank": 1.0, "local_gov": 2.0,
                 "urban_invest": 10.0, "bank_capital": 5.0}


def _term(tenor):
    return 1.0 - np.exp(-np.asarray(tenor, float) / 5.0)


def par_yield(curve: NelsonSiegel, tenor: float, spread_fn=None, freq: int = 1) -> float:
    """Coupon (decimal) at which a new bond prices at 100 on ``curve`` plus spread."""
    k = np.arange(1, int(round(tenor * freq)) + 1) / freq
    z = curve.zero(k) + (spread_fn(k) if spread_fn is not None else 0.0)
    df = np.exp(-z * k)
    return float(freq * (1 - df[-1]) / df.sum())


@dataclass
class NewIssue:
    sector: str
    issuer: Optional[str]
    tenor: float
    at: pd.Timestamp
    curve: NelsonSiegel
    spread_bp: float            # fair z-spread for the new bond
    issuer_adj_bp: float
    fair_yield: float           # percent
    concession_bp: float
    guidance: tuple             # (low, high) percent
    comparables: pd.DataFrame = field(repr=False)

    @property
    def guidance_mid(self) -> float:
        return self.fair_yield + self.concession_bp / 100


def comparables(con: sqlite3.Connection, market: Market, at, sector: str,
                curve: NelsonSiegel, max_age_s: float = 3600) -> pd.DataFrame:
    at = pd.Timestamp(at)
    t = to_years(at)
    snap = run_query(con, "curve_snapshot", at=at, sector=sector, max_age_s=max_age_s)
    rows = []
    for r in snap.itertuples():
        b = market.by_code[r.code]
        dirty = b.dirty_from_yield(t, r.mid_yield / 100)
        rows.append({"code": b.code, "issuer": b.issuer, "years_left": float(b.remaining_years(t)),
                     "mid_yield": r.mid_yield, "z_spread_bp": z_spread(b, t, dirty, curve) * 1e4})
    return pd.DataFrame(rows)


def price_new_issue(con: sqlite3.Connection, market: Market, at, sector: str, tenor: float,
                    issuer: Optional[str] = None, concession_bp: Optional[float] = None,
                    range_bp: float = 10.0, max_age_s: float = 3600) -> NewIssue:
    at = pd.Timestamp(at)
    fit, _ = fit_at(con, market, at, "quotes")
    curve = fit.curve
    if sector == "treasury":
        comps = pd.DataFrame(columns=["code", "issuer", "years_left", "mid_yield", "z_spread_bp"])
        spread_bp, adj = 0.0, 0.0
    else:
        comps = comparables(con, market, at, sector, curve, max_age_s)
        if comps.empty:
            raise ValueError(f"no live {sector} quotes at {at}")
        z = comps["z_spread_bp"].to_numpy()
        x = _term(comps["years_left"].to_numpy())
        if len(comps) >= 3 and np.ptp(x) > 0.05:
            X = np.column_stack([np.ones_like(x), x])
            coef, *_ = np.linalg.lstsq(X, z, rcond=None)
        else:
            coef = np.array([z.mean(), 0.0])
        comps["fitted_bp"] = coef[0] + coef[1] * x
        comps["residual_bp"] = z - comps["fitted_bp"]
        own = comps[comps["issuer"] == issuer] if issuer else comps.iloc[0:0]
        adj = float(own["residual_bp"].mean()) if len(own) else 0.0
        spread_bp = float(coef[0] + coef[1] * _term(tenor) + adj)
    s = spread_bp * 1e-4
    fair = par_yield(curve, tenor, lambda k: np.full_like(k, s)) * 100
    conc = CONCESSION_BP[sector] if concession_bp is None else concession_bp
    mid = fair + conc / 100
    return NewIssue(sector, issuer, tenor, at, curve, spread_bp, adj, fair, conc,
                    (round(mid - range_bp / 200, 4), round(mid + range_bp / 200, 4)), comps)


def true_fair_yield(market: Market, at, sector: str, tenor: float, issuer: Optional[str] = None) -> float:
    """Par yield (percent) of a new bond on the true curve and the issuer's true spread."""
    at = pd.Timestamp(at)
    base = 0.0
    if issuer is not None:
        codes = [b.code for b in market.bonds if b.issuer == issuer]
        base = market.base_bp[codes[0]] if codes else 0.0
    else:
        codes = [b.code for b in market.bonds if b.sector == sector]
        base = float(np.mean([market.base_bp[c] for c in codes])) if codes else 0.0
    spread = lambda k: market.true_sector_spread(sector, base, at, k)  # noqa: E731
    return par_yield(market.true_curve(at), tenor, spread) * 100
