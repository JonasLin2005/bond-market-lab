"""Command line: ``bondlab demo`` builds a database; the other commands read it."""
from __future__ import annotations

import argparse
import sqlite3
import sys

import pandas as pd

from . import store
from .api import Limits, MockQuoteAPI, SimClock
from .client import QuoteClient
from .experiment import check_times, run_experiment, summarise
from .logger import QuoteLogger
from .market import Market
from .newissue import price_new_issue, true_fair_yield
from .pnl import build_sheet, export_excel, write_sheet

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def _market_from_db(con: sqlite3.Connection) -> Market:
    meta = dict(con.execute("SELECT key, value FROM meta").fetchall())
    if not meta:
        sys.exit("no meta table: build the database with `bondlab demo` first")
    return Market(start=meta["start"], days=int(meta["days"]), seed=int(meta["seed"]))


def _sessions(market: Market) -> pd.DatetimeIndex:
    return pd.bdate_range(market.start.normalize(), market.minutes[-1].normalize())


def cmd_demo(a):
    market = Market(start=a.start, days=a.days, seed=a.seed)
    sessions = _sessions(market)
    log_from = sessions[-a.log_days]
    clock = SimClock(log_from + pd.Timedelta(hours=8))
    api = MockQuoteAPI(market, clock, Limits(transient_error_rate=a.error_rate), seed=a.seed + 1)
    client = QuoteClient(api, cache_dir=a.cache_dir, seed=a.seed + 2)
    con = store.connect(a.db)
    con.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                    [("start", a.start), ("days", str(a.days)), ("seed", str(a.seed))])
    store.write_securities(con, market.bonds)
    codes = [b.code for b in market.bonds]

    daily = client.daily_bars(codes, market.start, clock.now())
    n_daily = store.write_bars(con, daily, "bars_daily")
    minute = client.minute_bars(codes, sessions[-a.log_days - 1], sessions[-a.log_days - 1])
    n_min = store.write_bars(con, minute, "bars_minute")
    logger = QuoteLogger(client, con)
    logger.run(codes, until=market.end, every_s=a.every)

    print(f"market: {len(codes)} bonds, {a.days} sessions from {market.start:%Y-%m-%d}")
    print(f"daily bars {n_daily:,} rows, minute bars {n_min:,} rows, "
          f"quote log {logger.stats.polls:,} polls / {logger.stats.rows:,} rows -> {a.db}")
    s = client.stats
    print(f"client: {s.requests} requests, {s.attempts} attempts, {s.retries} retries "
          f"({s.rate_limited} rate-limited, {s.transient} transient), {s.cache_hits} cache hits")
    print("\nbid-offer by sector and tenor (bp):")
    print(store.run_query(con, "spread_by_tenor").to_string(index=False))


def cmd_query(a):
    con = store.connect(a.db)
    params = {k: v for k, v in (("code", a.code), ("at", a.at), ("sector", a.sector),
                                ("start", a.start), ("end", a.end)) if v is not None}
    if a.max_age is not None:
        params["max_age_s"] = a.max_age
    print(store.run_query(con, a.name, **params).to_string(index=False))


def cmd_refit(a):
    con = store.connect(a.db)
    market = _market_from_db(con)
    times = check_times(con, a.every)
    errors = run_experiment(con, market, times)
    print(f"{len(times)} check times; zero-rate error against the true curve, RMSE in bp:")
    print(summarise(errors).to_string(index=False))


def cmd_newissue(a):
    con = store.connect(a.db)
    market = _market_from_db(con)
    at = pd.Timestamp(a.at) if a.at else pd.Timestamp(
        con.execute("SELECT MAX(poll_time) FROM quotes_long").fetchone()[0])
    ni = price_new_issue(con, market, at, a.sector, a.tenor, a.issuer, range_bp=a.range)
    truth = true_fair_yield(market, at, a.sector, a.tenor, a.issuer)
    who = a.issuer or f"a {a.sector} issuer"
    print(f"{a.tenor:g}-year new issue for {who}, {at:%Y-%m-%d %H:%M}")
    if len(ni.comparables):
        print(ni.comparables.round(3).to_string(index=False))
    print(f"fair spread {ni.spread_bp:.1f} bp (issuer adjustment {ni.issuer_adj_bp:+.1f})")
    print(f"fair yield {ni.fair_yield:.4f}%   true {truth:.4f}%   error {(ni.fair_yield - truth) * 100:+.2f} bp")
    print(f"concession {ni.concession_bp:.0f} bp -> bookbuilding range {ni.guidance[0]:.2f}%-{ni.guidance[1]:.2f}%")


def cmd_pnl(a):
    con = store.connect(a.db)
    market = _market_from_db(con)
    days = _sessions(market)
    start, end = days[-a.sessions], days[-1]
    sheet = build_sheet(con, market, start, end)
    write_sheet(con, sheet)
    by_sector = store.run_query(con, "pnl_by_sector")
    print(f"P&L attribution, {start:%Y-%m-%d} to {end:%Y-%m-%d} (thousands):")
    print(by_sector.to_string(index=False))
    if a.xlsx:
        export_excel(sheet, by_sector, a.xlsx)
        print(f"sheet written to {a.xlsx}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="bondlab", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="simulate a market, pull history, log quotes into SQLite")
    d.add_argument("--db", default="lab.sqlite")
    d.add_argument("--start", default="2024-07-01")
    d.add_argument("--days", type=int, default=540)
    d.add_argument("--log-days", type=int, default=5)
    d.add_argument("--every", type=float, default=60.0, help="seconds between polls")
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--error-rate", type=float, default=0.02)
    d.add_argument("--cache-dir", default=None)
    d.set_defaults(fn=cmd_demo)

    q = sub.add_parser("query", help="run a named SQL query: " + ", ".join(store.query_names()))
    q.add_argument("name", choices=store.query_names())
    q.add_argument("--db", default="lab.sqlite")
    for opt in ("code", "at", "sector", "start", "end"):
        q.add_argument(f"--{opt}")
    q.add_argument("--max-age", type=float)
    q.set_defaults(fn=cmd_query)

    r = sub.add_parser("refit", help="refit the curve through the log and score it against the truth")
    r.add_argument("--db", default="lab.sqlite")
    r.add_argument("--every", type=int, default=30, help="minutes between check times")
    r.set_defaults(fn=cmd_refit)

    n = sub.add_parser("newissue", help="price a new bond off the curve and comparables")
    n.add_argument("--db", default="lab.sqlite")
    n.add_argument("--sector", required=True,
                   choices=["treasury", "policy_bank", "local_gov", "urban_invest", "bank_capital"])
    n.add_argument("--tenor", type=float, required=True)
    n.add_argument("--issuer")
    n.add_argument("--at")
    n.add_argument("--range", type=float, default=10.0, help="width of the bookbuilding range, bp")
    n.set_defaults(fn=cmd_newissue)

    s = sub.add_parser("pnl", help="daily position and P&L sheet with attribution")
    s.add_argument("--db", default="lab.sqlite")
    s.add_argument("--sessions", type=int, default=60)
    s.add_argument("--xlsx")
    s.set_defaults(fn=cmd_pnl)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":  # pragma: no cover
    main()
