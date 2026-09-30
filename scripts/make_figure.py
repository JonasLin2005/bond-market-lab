"""Rebuild the README figure and print the README's result tables.

    python scripts/make_figure.py            # uses or creates lab.sqlite (default demo)
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bondlab import store  # noqa: E402
from bondlab.cli import _market_from_db, _sessions, main  # noqa: E402
from bondlab.experiment import check_times, run_experiment, summarise  # noqa: E402
from bondlab.newissue import price_new_issue, true_fair_yield  # noqa: E402
from bondlab.pnl import build_sheet, daily_totals  # noqa: E402

DB = sys.argv[1] if len(sys.argv) > 1 else "lab.sqlite"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

if not os.path.exists(DB):
    main(["demo", "--db", DB])
con = store.connect(DB)
market = _market_from_db(con)
sessions = _sessions(market)

# 1. refit experiment
times = check_times(con, 30)
err = run_experiment(con, market, times)
summary = summarise(err)
print(f"## Refit: {len(times)} check times\n")
print(summary.to_markdown(index=False))

# 2. new issues
cases = [("treasury", None, 10), ("policy_bank", "Policy Bank A", 5), ("local_gov", "Province A", 7),
         ("local_gov", "Province B", 15), ("urban_invest", "City E Urban Development Group", 3),
         ("urban_invest", "County F Construction Investment", 5), ("bank_capital", "Bank H", 5),
         ("urban_invest", None, 5)]
at = sessions[-1] + pd.Timedelta(hours=15)
rows = []
for sector, issuer, tenor in cases:
    ni = price_new_issue(con, market, at, sector, tenor, issuer)
    truth = true_fair_yield(market, at, sector, tenor, issuer)
    rows.append({"sector": sector, "issuer": issuer or ("Treasury" if sector == "treasury" else "(new issuer)"), "tenor": tenor,
                 "comparables": len(ni.comparables), "fair yield %": round(ni.fair_yield, 3),
                 "true %": round(truth, 3), "error bp": round((ni.fair_yield - truth) * 100, 1),
                 "range %": f"{ni.guidance[0]:.2f}-{ni.guidance[1]:.2f}"})
print(f"\n## New issues at {at:%Y-%m-%d %H:%M}\n")
print(pd.DataFrame(rows).to_markdown(index=False))

# 3. P&L attribution, fitted vs true curves
start, end = sessions[-60], sessions[-1]
fitted = daily_totals(build_sheet(con, market, start, end))
truth = daily_totals(build_sheet(con, market, start, end, true_curves=True))
curve_cols = ["level_k", "slope_k", "curvature_k"]
corr = np.corrcoef(fitted[curve_cols].sum(axis=1), truth[curve_cols].sum(axis=1))[0, 1]
print(f"\n## P&L {start:%Y-%m-%d} to {end:%Y-%m-%d}\n")
tot = fitted[["carry_k", "rolldown_k", *curve_cols, "spread_k", "unexplained_k", "total_k"]].sum()
print(tot.round(1).to_frame("fitted curves").join(
    truth[["carry_k", "rolldown_k", *curve_cols, "spread_k", "unexplained_k", "total_k"]].sum()
    .round(1).to_frame("true curves")).to_markdown())
print(f"\ncorrelation of daily curve P&L, fitted vs true: {corr:.3f}; "
      f"max |unexplained| {fitted['unexplained_k'].abs().max():.2e}")

# ---- figure ------------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=160)

for variant, color, label in (("quotes", BLUE, "all 10 treasuries"), ("no_long", ORANGE, "without the 30-year")):
    g = err[err["variant"] == variant].groupby("tenor")["error_bp"]
    lo, mid, hi = g.quantile(0.05), g.median(), g.quantile(0.95)
    ax1.fill_between(mid.index, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax1.plot(mid.index, mid, color=color, linewidth=2)
    ax1.plot([], [], color=color, linewidth=2, label=label)
ax1.axhline(0, color=INK2, linewidth=0.8)
ax1.legend(frameon=False, loc="lower left", fontsize=8.5, labelcolor=INK)
ax1.set_xlabel("maturity (years)")
ax1.set_ylabel("fitted minus true zero rate (bp)")
ax1.set_title("Curve refit from logged quotes vs the true curve\nmedian and 5-95% band over "
              f"{len(times)} fits", loc="left", color=INK, fontsize=9.5)
ax1.grid(axis="y", color=GRID, linewidth=0.6)

dates = pd.to_datetime(fitted["date"])
series = [("carry + roll-down", fitted["carry_k"] + fitted["rolldown_k"], AQUA),
          ("curve (level, slope, curvature)", fitted[curve_cols].sum(axis=1), BLUE),
          ("spread", fitted["spread_k"], ORANGE),
          ("total", fitted["total_k"], INK)]
for label, s, color in series:
    c = s.cumsum() / 1000
    ax2.plot(dates, c, color=color, linewidth=2 if label != "total" else 1.6,
             linestyle="-" if label != "total" else "--")
    ax2.annotate(label, (dates.iloc[-1], c.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                 va="center", color=INK, fontsize=8.5)
ax2.axhline(0, color=INK2, linewidth=0.8)
ax2.set_ylabel("cumulative P&L (millions)")
ax2.set_title("Mock book: daily P&L attribution, last 60 sessions", loc="left", color=INK, fontsize=9.5)
ax2.grid(axis="y", color=GRID, linewidth=0.6)
ax2.tick_params(axis="x", labelrotation=0)
ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
ax2.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
fig.subplots_adjust(right=0.8, wspace=0.28)
os.makedirs("docs", exist_ok=True)
fig.savefig("docs/figure.png", bbox_inches="tight", facecolor="white")
print("\nwrote docs/figure.png")
