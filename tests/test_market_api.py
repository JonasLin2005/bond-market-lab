import numpy as np
import pandas as pd
import pytest

from bondlab.api import (RateLimitError, SpanLimitError, TooManyCodesError, TransientError,
                         UnknownCodeError)
from bondlab.bonds import to_years
from bondlab.market import MINUTES_PER_DAY, Market, trading_minutes

from conftest import fresh_api


def test_trading_minutes_skip_lunch_and_weekends():
    m = trading_minutes("2024-07-05", 3)          # Fri, Mon, Tue
    assert len(m) == 3 * MINUTES_PER_DAY
    assert set(m.dayofweek) <= {0, 1, 2, 3, 4}
    hm = m.hour * 60 + m.minute
    assert not ((hm >= 12 * 60) & (hm < 13 * 60 + 30)).any()
    assert hm.min() == 9 * 60 and hm.max() == 16 * 60 + 59


def test_reproducible_by_seed():
    a, b, c = Market(days=5, seed=1), Market(days=5, seed=1), Market(days=5, seed=2)
    code = a.bonds[0].code
    pd.testing.assert_frame_equal(a.quotes[code], b.quotes[code])
    assert not np.allclose(a.factors, c.factors)


def test_quotes_are_sane(market):
    for code, q in market.quotes.items():
        both = q.dropna(subset=["bid_yield", "offer_yield"])
        assert (both["bid_yield"] > both["offer_yield"]).all()
        assert q["time"].is_monotonic_increasing
        assert (q["time"] >= market.by_code[code].issue).all()
        one_sided = q["bid_yield"].isna() | q["offer_yield"].isna()
        assert 0 < one_sided.mean() < 0.15


def test_fair_yield_reprices_on_the_true_curve(market):
    b = next(x for x in market.bonds if x.sector == "urban_invest")
    ts = pd.DatetimeIndex([market.minutes[1000], market.minutes[-5]])
    y = market.fair_yield(b.code, ts)
    t = to_years(ts)
    spread = market.spread_fn(b.sector, market.base_bp[b.code], ts)
    betas = market.factors[market._minute_index(ts)]
    assert np.allclose(b.dirty_from_yield(t, y), b.dirty_from_curve(t, betas, market.model.tau, spread))


def test_credit_spreads_are_ordered(market):
    at = market.minutes[-1]
    fy = {b.sector: [] for b in market.bonds}
    for b in market.bonds:
        fy[b.sector].append(market.fair_yield(b.code, [at])[0] - market.true_curve(at).zero(
            float(b.remaining_years(to_years(at)))))
    assert np.mean(fy["urban_invest"]) > np.mean(fy["bank_capital"]) > np.mean(fy["local_gov"])


def test_trades_print_at_bid_or_offer(market):
    code = market.bonds[7].code
    q, tr = market.quotes[code], market.trades[code]
    idx = q["time"].searchsorted(tr["time"], side="right") - 1
    quotes = q.iloc[idx]
    at_side = np.isclose(tr["yield"].to_numpy(), quotes["bid_yield"].to_numpy()) | \
        np.isclose(tr["yield"].to_numpy(), quotes["offer_yield"].to_numpy())
    assert at_side.all()


def test_daily_bars_match_trades(market):
    code = market.bonds[9].code
    bars = market.bars(code, market.start, market.end, "1d")
    tr = market.trades[code]
    g = tr.groupby(tr["time"].dt.floor("D"))
    assert np.allclose(bars["close"], g["clean"].last())
    assert np.allclose(bars["volume"], g["volume"].sum())
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1)).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1)).all()
    assert bars["close_mid_yield"].notna().all()


def test_snapshot(market):
    code = market.bonds[0].code
    first = market.quotes[code]["time"].iloc[0]
    assert market.snapshot(code, first - pd.Timedelta(seconds=1)) is None
    s = market.snapshot(code, market.minutes[-1])
    assert s["quote_time"] <= market.minutes[-1]
    if not np.isnan(s["mid_yield"]):
        assert s["mid_yield"] == pytest.approx((s["bid_yield"] + s["offer_yield"]) / 2)


# ---- API limits ----------------------------------------------------------

def test_span_and_code_limits(market):
    api, _ = fresh_api(market, market.end, max_daily_span_days=30, max_codes=2)
    codes = [b.code for b in market.bonds]
    with pytest.raises(SpanLimitError):
        api.daily_bars(codes[:1], "2024-07-01", "2024-07-31")    # 31 days
    api.daily_bars(codes[:1], "2024-07-01", "2024-07-30")        # 30 days is fine
    with pytest.raises(TooManyCodesError):
        api.daily_bars(codes[:3], "2024-07-01", "2024-07-02")
    with pytest.raises(UnknownCodeError):
        api.snapshot(["NOPE"])
    with pytest.raises(SpanLimitError):
        api.minute_bars(codes[:1], "2024-07-01", "2024-07-02")


def test_rate_limit_and_refill(market):
    api, clock = fresh_api(market, market.end, calls_per_second=2, burst=2)
    code = [market.bonds[0].code]
    api.snapshot(code)
    api.snapshot(code)
    with pytest.raises(RateLimitError) as err:
        api.snapshot(code)
    assert err.value.retry_after == pytest.approx(0.5)
    clock.sleep(err.value.retry_after)
    api.snapshot(code)


def test_transient_errors(market):
    api, _ = fresh_api(market, market.end, transient_error_rate=1.0, calls_per_second=1e9, burst=10**9)
    with pytest.raises(TransientError):
        api.snapshot([market.bonds[0].code])


def test_snapshot_only_and_no_future_bars(market, sessions):
    now = sessions[-10] + pd.Timedelta(hours=10)
    api, clock = fresh_api(market, now, calls_per_second=1e9, burst=10**9, transient_error_rate=0)
    code = market.bonds[7].code
    snap = api.snapshot([code])
    assert (snap["quote_time"] <= now).all()
    bars = api.daily_bars([code], sessions[-12], sessions[-1])
    assert pd.to_datetime(bars["time"]).max() <= now
