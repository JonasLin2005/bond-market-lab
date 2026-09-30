"""The validation experiment: refit the curve from logged quotes and compare with the truth.

Because the market is simulated, the curve that generated every quote is known.
At each check time the treasury quotes are read back out of SQLite
(``curve_snapshot.sql``), a curve is fitted, and its zero rates are compared
with the true curve at that minute. Three fits per check time:

* ``truth``   fair yields straight from the true curve: tests the fitter itself
* ``quotes``  mid yields as logged, with noise, bid-offer and staleness
* ``no_long`` the same quotes without the longest bond: how far the long end rests on it
"""
from __future__ import annotations

import sqlite3
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .bonds import to_years
from .fit import REPORT_TENORS, curve_error_bp, fit_curve
from .market import SESSIONS, Market
from .store import run_query

SEGMENTS = {"short (<2y)": (0, 2), "belly (2-10y)": (2, 10), "long (10-30y)": (10, 30.01)}


def fit_at(con: sqlite3.Connection, market: Market, at, variant: str = "quotes",
           max_age_s: float = 1800, exclude: Iterable[str] = ()):
    at = pd.Timestamp(at)
    snap = run_query(con, "curve_snapshot", at=at, sector="treasury", max_age_s=max_age_s)
    snap = snap[~snap["code"].isin(list(exclude))]
    if variant == "no_long":
        snap = snap.iloc[:-1]
    bonds = [market.by_code[c] for c in snap["code"]]
    t = to_years(at)
    if variant == "truth":
        y = np.array([market.fair_yield(b.code, [at])[0] for b in bonds])
    else:
        y = snap["mid_yield"].to_numpy() / 100.0
    res = fit_curve(bonds, t, y)
    err = curve_error_bp(res.curve, market.true_curve(at))
    return res, err


def run_experiment(con: sqlite3.Connection, market: Market, times: Iterable,
                   variants=("truth", "quotes", "no_long")) -> pd.DataFrame:
    """One row per (time, variant, tenor) with the zero-rate error in bp."""
    rows = []
    for at in times:
        for v in variants:
            res, err = fit_at(con, market, at, v)
            for tenor, e in zip(REPORT_TENORS, err):
                rows.append((pd.Timestamp(at), v, tenor, e, res.rmse_bp, len(res.codes)))
    return pd.DataFrame(rows, columns=["time", "variant", "tenor", "error_bp", "fit_rmse_bp", "n_bonds"])


def summarise(errors: pd.DataFrame) -> pd.DataFrame:
    """RMSE of the zero-rate error by variant and curve segment, plus the fit residual."""
    out = []
    for v, g in errors.groupby("variant", sort=False):
        row = {"variant": v, "fits": g["time"].nunique(),
               "bonds": int(g["n_bonds"].median()),
               "fit_residual_bp": round(float(g.groupby("time")["fit_rmse_bp"].first().median()), 2)}
        for name, (lo, hi) in SEGMENTS.items():
            s = g[(g["tenor"] >= lo) & (g["tenor"] < hi)]["error_bp"]
            row[name] = round(float(np.sqrt(np.mean(s**2))), 2)
        row["worst_bp"] = round(float(g["error_bp"].abs().max()), 2)
        out.append(row)
    return pd.DataFrame(out)


def check_times(con: sqlite3.Connection, every_minutes: int = 30) -> list:
    """Session times on the logged grid, every ``every_minutes``."""
    polls = pd.to_datetime(pd.read_sql_query(
        "SELECT DISTINCT poll_time FROM quotes_long ORDER BY poll_time", con)["poll_time"])
    if polls.empty:
        return []
    keep = polls[(polls.dt.minute % every_minutes == 0) & (polls.dt.second == 0)]
    # skip the first half hour of each session: until bonds have quoted again,
    # the latest quotes are from before the break and too stale to fit
    tod = keep - keep.dt.normalize()
    ok = pd.Series(False, index=keep.index)
    for hhmm, n in SESSIONS:
        start = pd.Timedelta(hhmm + ":00")
        ok |= (tod >= start + pd.Timedelta(minutes=30)) & (tod < start + pd.Timedelta(minutes=n))
    return list(keep[ok])
