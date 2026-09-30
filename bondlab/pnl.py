"""Daily position and P&L sheet with attribution.

Each session every bond is marked at its closing yield (``eod_marks.sql``,
carried forward when it did not trade), the treasury curve is refitted to the
fresh treasury marks, and each bond's z-spread over that curve is solved. The
day's P&L for a position is then split by revaluing it one factor at a time:

* carry      the bond at yesterday's yield, one day older, plus any coupon paid
* roll-down  the move along an unchanged curve and spread as maturity shortens
* level / slope / curvature   the curve's three Nelson-Siegel factors, in that order
                               (a change in the decay parameter is booked to curvature)
* spread     the bond's z-spread move over the curve
* unexplained  what is left against the mark-to-market change; zero up to
               solver tolerance, so it doubles as a reconciliation check

Units: face in millions, P&L and DV01 in thousands.
"""
from __future__ import annotations

import sqlite3
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .bonds import to_years
from .curve import NelsonSiegel
from .fit import fit_curve
from .market import Market
from .spreads import price_on_curve, z_spread
from .store import run_query

MARK_TIME = pd.Timedelta(hours=17)
COMPONENTS = ["carry_k", "rolldown_k", "level_k", "slope_k", "curvature_k", "spread_k", "unexplained_k"]

# A small mock book: long credit and rates, partly hedged with a short in the 10-year treasury.
DEFAULT_POSITIONS = {
    "treasury_10y_otr": -150.0,
    "treasury_30y": 50.0,
    "policy_bank_10y": 100.0,
    "local_gov_10y": 200.0,
    "urban_invest_5y": 100.0,
    "urban_invest_high": 50.0,
    "bank_capital": 100.0,
}


def default_positions(market: Market) -> Dict[str, float]:
    """Map the mock book's roles onto this market's bond codes."""
    t_end = to_years(market.minutes[-1])
    def pick(sector, target, key=None):
        cands = [b for b in market.bonds if b.sector == sector]
        if key is not None:
            return max(cands, key=key).code
        return min(cands, key=lambda b: abs(float(b.remaining_years(t_end)) - target)).code
    roles = {
        "treasury_10y_otr": pick("treasury", 9.6),
        "treasury_30y": pick("treasury", 30),
        "policy_bank_10y": pick("policy_bank", 9.5),
        "local_gov_10y": pick("local_gov", 8.8),
        "urban_invest_5y": pick("urban_invest", 3.8),
        "urban_invest_high": pick("urban_invest", 0, key=lambda b: market.base_bp[b.code]),
        "bank_capital": pick("bank_capital", 4.1),
    }
    out: Dict[str, float] = {}
    for role, face in DEFAULT_POSITIONS.items():
        out[roles[role]] = out.get(roles[role], 0.0) + face
    return out


def daily_curves(marks: pd.DataFrame, market: Market) -> Dict[str, NelsonSiegel]:
    """Treasury curve per session, fitted to that day's fresh treasury marks.

    The decay parameter is estimated on the first day and then held fixed, as in
    Diebold and Li (2006). With it free, day-to-day noise in the decay moves
    P&L between level, slope and curvature in large offsetting amounts.
    """
    curves, prev, tau = {}, None, None
    for d, g in marks[marks["sector"] == "treasury"].groupby("date"):
        g = g[(g["stale"] == 0) & g["close_yield"].notna()]
        t = to_years(pd.Timestamp(d) + MARK_TIME)
        if len(g) >= 5:
            bonds = [market.by_code[c] for c in g["code"]]
            prev = fit_curve(bonds, t, g["close_yield"].to_numpy() / 100, tau=tau).curve
            tau = prev.tau
        if prev is not None:
            curves[d] = prev
    return curves


def _coupon_paid(bond, d0: pd.Timestamp, d1: pd.Timestamp) -> float:
    paid = (bond.payment_dates > d0) & (bond.payment_dates <= d1)
    return float(bond.cashflows[np.asarray(paid)].sum()) if paid.any() else 0.0


def build_sheet(con: sqlite3.Connection, market: Market, start, end,
                positions: Optional[Dict[str, float]] = None,
                true_curves: bool = False) -> pd.DataFrame:
    """The position-level sheet. ``true_curves`` swaps the fitted daily curves
    for the simulation's true ones, which gives the attribution a benchmark."""
    positions = positions or default_positions(market)
    marks = run_query(con, "eod_marks", start=f"{pd.Timestamp(start):%Y-%m-%d}",
                      end=f"{pd.Timestamp(end):%Y-%m-%d}")
    curves = daily_curves(marks, market)
    if true_curves:
        curves = {d: market.true_curve(pd.Timestamp(d) + MARK_TIME - pd.Timedelta(minutes=1))
                  for d in curves}
    marks = marks[marks["code"].isin(positions) & marks["date"].isin(curves)]
    rows = []
    for code, g in marks.groupby("code"):
        bond = market.by_code[code]
        face = positions[code]
        scale = face * 10.0   # per-100 price change -> thousands for `face` million
        prev = None
        for r in g.sort_values("date").itertuples():
            if np.isnan(r.close_yield):
                continue
            d = pd.Timestamp(r.date)
            ts = d + MARK_TIME
            if ts >= bond.maturity:
                continue
            t = to_years(ts)
            y = r.close_yield / 100
            curve = curves[r.date]
            dirty = bond.dirty_from_yield(t, y)
            z = z_spread(bond, t, dirty, curve)
            row = {"date": r.date, "code": code, "face_mm": face,
                   "clean": dirty - bond.accrued(t), "yield_pct": r.close_yield,
                   "z_spread_bp": z * 1e4, "stale": int(r.stale),
                   "dv01_k": bond.dv01(t, y) * scale}
            if prev is None:
                row.update({c: 0.0 for c in COMPONENTS}, total_k=0.0)
            else:
                t0, y0, c0, z0, dirty0, d0 = prev
                cpn = _coupon_paid(bond, d0 + MARK_TIME, ts)
                carry = bond.dirty_from_yield(t, y0) + cpn - dirty0
                b_ = price_on_curve(bond, t, c0, z0)
                roll = b_ - bond.dirty_from_yield(t, y0)
                lvl_c = NelsonSiegel(curve.b0, c0.b1, c0.b2, c0.tau)
                slp_c = NelsonSiegel(curve.b0, curve.b1, c0.b2, c0.tau)
                p1 = price_on_curve(bond, t, lvl_c, z0)
                p2 = price_on_curve(bond, t, slp_c, z0)
                p3 = price_on_curve(bond, t, curve, z0)
                total = dirty + cpn - dirty0
                p4 = price_on_curve(bond, t, curve, z)
                parts = dict(carry_k=carry, rolldown_k=roll, level_k=p1 - b_, slope_k=p2 - p1,
                             curvature_k=p3 - p2, spread_k=p4 - p3)
                parts["unexplained_k"] = total - sum(parts.values())
                row.update({k: v * scale for k, v in parts.items()}, total_k=total * scale)
            rows.append(row)
            prev = (t, y, curve, z, dirty, d)
    cols = ["date", "code", "face_mm", "clean", "yield_pct", "z_spread_bp", "stale", "dv01_k",
            *COMPONENTS, "total_k"]
    return pd.DataFrame(rows, columns=cols).sort_values(["date", "code"]).reset_index(drop=True)


def write_sheet(con: sqlite3.Connection, sheet: pd.DataFrame) -> None:
    con.execute("DELETE FROM pnl_daily")
    con.executemany(f"INSERT INTO pnl_daily VALUES ({', '.join('?' * 16)})",
                    sheet.itertuples(index=False, name=None))
    con.commit()


def daily_totals(sheet: pd.DataFrame) -> pd.DataFrame:
    cols = ["dv01_k", *COMPONENTS, "total_k"]
    out = sheet.groupby("date")[cols].sum()
    out["cumulative_k"] = out["total_k"].cumsum()
    return out.reset_index()


def export_excel(sheet: pd.DataFrame, by_sector: pd.DataFrame, path: str) -> None:
    """Three tabs: daily totals, the position-level sheet, and the sector summary."""
    from openpyxl.utils import get_column_letter

    totals = daily_totals(sheet)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in (("Daily P&L", totals), ("Positions", sheet), ("By sector", by_sector)):
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes = "B2" if name != "Positions" else "C2"
            for i, col in enumerate(df.columns, start=1):
                width = max(len(str(col)), 10) + 2
                ws.column_dimensions[get_column_letter(i)].width = width
                if df[col].dtype.kind == "f":
                    for cell in ws.iter_rows(min_row=2, min_col=i, max_col=i):
                        cell[0].number_format = "#,##0.00"
