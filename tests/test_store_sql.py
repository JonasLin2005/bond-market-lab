import numpy as np
import pandas as pd
import pytest

from bondlab import store
from bondlab.logger import in_session, next_session_start
from bondlab.market import QUOTE_FIELDS


def wide(con):
    df = pd.read_sql_query("SELECT * FROM quotes_wide", con)
    df["poll_time"] = pd.to_datetime(df["poll_time"])
    df["quote_time"] = pd.to_datetime(df["quote_time"])
    return df


def test_tables_and_queries_exist(lab):
    names = {r[0] for r in lab["con"].execute("SELECT name FROM sqlite_master")}
    assert {"securities", "quotes_long", "bars_daily", "bars_minute", "pnl_daily", "meta",
            "quotes_wide"} <= names
    assert {"spread_by_tenor", "ohlc_1min", "curve_snapshot", "eod_marks"} <= set(store.query_names())


def test_long_format_and_idempotent_insert(market):
    con = store.connect()
    store.write_securities(con, market.bonds)
    snap = pd.DataFrame([market.snapshot(b.code, market.minutes[-1]) for b in market.bonds[:3]])
    rows = store.snapshot_to_long(snap, market.minutes[-1])
    assert len(rows) == 3 * len(QUOTE_FIELDS)
    assert store.write_quotes(con, rows) == len(rows)
    assert store.write_quotes(con, rows) == 0


def test_wide_view_round_trips_a_snapshot(market):
    con = store.connect()
    store.write_securities(con, market.bonds)
    at = market.minutes[-1]
    s = market.snapshot(market.bonds[7].code, at)
    store.write_quotes(con, store.snapshot_to_long(pd.DataFrame([s]), at))
    w = pd.read_sql_query("SELECT * FROM quotes_wide", con).iloc[0]
    for f in ("bid_yield", "offer_yield", "bid_clean", "accrued", "last_yield"):
        assert w[f] == pytest.approx(s[f])
    assert pd.Timestamp(w["quote_time"]) == s["quote_time"].floor("s")


def test_logger_polls_only_in_session_on_a_fixed_grid(lab):
    polls = pd.to_datetime(pd.read_sql_query(
        "SELECT DISTINCT poll_time FROM quotes_long ORDER BY 1", lab["con"])["poll_time"])
    assert all(in_session(p) for p in polls)
    assert len(polls) == 2 * 390
    assert set(polls.dt.second) == {0}
    assert next_session_start(pd.Timestamp("2024-07-05 12:30")) == pd.Timestamp("2024-07-05 13:30")
    assert next_session_start(pd.Timestamp("2024-07-05 17:00")) == pd.Timestamp("2024-07-08 09:00")


def test_ohlc_matches_pandas(lab):
    con = lab["con"]
    code = lab["codes"][7]
    got = store.run_query(con, "ohlc_1min", code=code)
    w = wide(con)
    w = w[(w["code"] == code) & w["mid_yield"].notna()].set_index("poll_time")["mid_yield"]
    want = w.resample("min").ohlc().dropna()
    assert len(got) == len(want)
    assert np.allclose(got[["open", "high", "low", "close"]].to_numpy(), want.to_numpy())


def test_spread_by_tenor_matches_pandas(lab, market):
    con = lab["con"]
    got = store.run_query(con, "spread_by_tenor")
    w = wide(con).dropna(subset=["bid_yield", "offer_yield"])
    w = w.drop_duplicates(["code", "quote_time", "bid_yield", "offer_yield"])
    sec = market.describe_book().set_index("code")
    w["sector"] = w["code"].map(sec["sector"])
    tr = got[got["sector"] == "treasury"]["n_quotes"].sum()
    assert tr == (w["sector"] == "treasury").sum()
    avg = ((w["bid_yield"] - w["offer_yield"]) * 100).groupby(w["sector"]).mean()
    for s, g in got.groupby("sector"):
        assert np.average(g["avg_spread_bp"], weights=g["n_quotes"]) == pytest.approx(avg[s], abs=0.01)


def test_staleness_and_largest_moves(lab):
    con = lab["con"]
    w = wide(con)
    age = (w["poll_time"] - w["quote_time"]).dt.total_seconds()
    pct = (age > 600).groupby(w["code"]).mean() * 100
    got = store.run_query(con, "quote_staleness", max_age_s=600).set_index("code")
    assert np.allclose(got.loc[pct.index, "pct_stale"], pct.round(1), atol=0.051)

    moves = store.run_query(con, "largest_moves", n=3)
    m = w.dropna(subset=["mid_yield"]).sort_values(["code", "poll_time"])
    d = m.groupby("code")["mid_yield"].diff().abs().max() * 100
    assert abs(moves["change_bp"].iloc[0]) == pytest.approx(d, abs=0.006)


def test_curve_snapshot_takes_latest_fresh_two_sided_quote(lab, sessions):
    con = lab["con"]
    at = sessions[-1] + pd.Timedelta(hours=15, minutes=12)
    got = store.run_query(con, "curve_snapshot", at=at, sector="treasury", max_age_s=1800)
    w = wide(con)
    w = w[(w["poll_time"] <= at) & w["mid_yield"].notna()
          & ((at - w["quote_time"]).dt.total_seconds() <= 1800)
          & (w["poll_time"] >= at - pd.Timedelta(seconds=1800))]
    w = w[w["code"].str.startswith("TB")]
    want = w.sort_values("poll_time").groupby("code").last()
    assert set(got["code"]) == set(want.index)
    assert np.allclose(got.set_index("code").loc[want.index, "mid_yield"], want["mid_yield"])


def test_eod_marks_carry_forward_and_flag(lab, sessions):
    con = lab["con"]
    marks = store.run_query(con, "eod_marks", start=f"{sessions[-20]:%Y-%m-%d}",
                            end=f"{sessions[-1]:%Y-%m-%d}")
    assert marks.groupby("code").size().max() <= 20
    stale = marks[marks["stale"] == 1]
    assert len(stale) > 0                                  # illiquid bonds miss sessions
    assert (stale["mark_date"] < stale["date"]).all()
    fresh = marks[marks["stale"] == 0]
    bars = pd.read_sql_query("SELECT code, substr(time,1,10) AS d, close_mid_yield FROM bars_daily", con)
    j = fresh.merge(bars, left_on=["code", "date"], right_on=["code", "d"])
    assert len(j) == len(fresh) and np.allclose(j["close_yield"], j["close_mid_yield"])


def test_daily_moves(lab):
    con = lab["con"]
    got = store.run_query(con, "daily_moves").set_index("code")
    b = pd.read_sql_query("SELECT code, time, close_yield FROM bars_daily ORDER BY code, time", con)
    chg = b.groupby("code")["close_yield"].diff() * 100
    avg = chg.abs().groupby(b["code"]).mean()
    assert np.allclose(got.loc[avg.index, "avg_abs_move_bp"], avg.round(2), atol=0.006)
