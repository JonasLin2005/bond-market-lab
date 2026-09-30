"""SQLite store: schema, writers, and the named analytics queries in ``sql/``."""
from __future__ import annotations

import sqlite3
from importlib import resources
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .market import QUOTE_FIELDS

TIME_FIELDS = {"quote_time", "last_trade_time"}


def sql_text(name: str) -> str:
    return resources.files("bondlab").joinpath("sql").joinpath(f"{name}.sql").read_text(encoding="utf-8")


def query_names() -> list:
    files = resources.files("bondlab").joinpath("sql").iterdir()
    return sorted(p.name[:-4] for p in files if p.name.endswith(".sql") and p.name != "schema.sql")


def connect(path: str = ":memory:") -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(sql_text("schema"))
    return con


def write_securities(con: sqlite3.Connection, bonds: Iterable) -> None:
    rows = [(b.code, b.name, b.sector, b.issuer, b.coupon * 100, b.freq,
             f"{b.issue:%Y-%m-%d}", f"{b.maturity:%Y-%m-%d}") for b in bonds]
    con.executemany("INSERT OR REPLACE INTO securities VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    con.commit()


def _unix(ts) -> Optional[float]:
    if ts is None or pd.isna(ts):
        return None
    return (pd.Timestamp(ts) - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)


def snapshot_to_long(snap: pd.DataFrame, poll_time) -> list:
    """Melt a snapshot frame into ``(poll_time, code, field, value)`` rows."""
    stamp = f"{pd.Timestamp(poll_time):%Y-%m-%d %H:%M:%S}"
    rows = []
    for rec in snap.to_dict("records"):
        for field in QUOTE_FIELDS:
            if field not in rec:
                continue
            v = rec[field]
            v = _unix(v) if field in TIME_FIELDS else (None if pd.isna(v) else float(v))
            rows.append((stamp, rec["code"], field, v))
    return rows


def write_quotes(con: sqlite3.Connection, rows: list) -> int:
    """Insert long rows; repeats of the same poll are ignored. Returns rows added."""
    before = con.total_changes
    con.executemany("INSERT OR IGNORE INTO quotes_long VALUES (?, ?, ?, ?)", rows)
    con.commit()
    return con.total_changes - before


def write_bars(con: sqlite3.Connection, bars: pd.DataFrame, table: str) -> int:
    if table not in ("bars_daily", "bars_minute"):
        raise ValueError(table)
    if bars.empty:
        return 0
    cols = ["code", "time", "open", "high", "low", "close", "open_yield", "high_yield",
            "low_yield", "close_yield", "close_mid_yield", "volume", "n_trades"]
    df = bars[cols].copy()
    df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["n_trades"] = df["n_trades"].astype(int)
    before = con.total_changes
    rows = [tuple(None if (isinstance(v, float) and np.isnan(v)) else v for v in r)
            for r in df.itertuples(index=False, name=None)]
    con.executemany(f"INSERT OR REPLACE INTO {table} VALUES ({', '.join('?' * len(cols))})", rows)
    con.commit()
    return con.total_changes - before


def run_query(con: sqlite3.Connection, name: str, **params) -> pd.DataFrame:
    """Run ``sql/<name>.sql`` with named parameters."""
    defaults = {"code": None, "max_age_s": 1800, "n": 10, "sector": "treasury"}
    sql = sql_text(name)
    used = {k: v for k, v in {**defaults, **params}.items() if f":{k}" in sql}
    for k, v in list(used.items()):
        if isinstance(v, pd.Timestamp):
            used[k] = f"{v:%Y-%m-%d %H:%M:%S}"
    return pd.read_sql_query(sql, con, params=used)
