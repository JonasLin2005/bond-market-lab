import numpy as np
import pandas as pd
import pytest

from bondlab.client import GaveUp, QuoteClient, batches, date_windows

from conftest import fresh_api


@pytest.mark.parametrize("span", [1, 7, 30, 365])
def test_windows_cover_the_range_exactly(span):
    rng = np.random.default_rng(span)
    for _ in range(50):
        s = pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(rng.integers(0, 1000)))
        e = s + pd.Timedelta(days=int(rng.integers(0, 900)))
        w = date_windows(s, e, span)
        assert w[0][0] == s and w[-1][1] == e
        for (a, b), (c, _) in zip(w, w[1:]):
            assert c == b + pd.Timedelta(days=1)
        assert all((b - a).days + 1 <= span for a, b in w)


def test_windows_edge_cases():
    assert date_windows("2024-01-02", "2024-01-01", 5) == []
    days = date_windows("2024-07-05", "2024-07-09", 1, business_days_only=True)
    assert [d.day for d, _ in days] == [5, 8, 9]
    assert batches(list("abcdefg"), 3) == [["a", "b", "c"], ["d", "e", "f"], ["g"]]


def direct_daily(market, codes, start, end):
    frames = [market.bars(c, start, end, "1d") for c in sorted(codes)]
    return pd.concat([f for f in frames if len(f)], ignore_index=True)


def test_chunked_pull_equals_one_big_pull(market):
    api, clock = fresh_api(market, market.end, max_daily_span_days=20, max_codes=4,
                           transient_error_rate=0, calls_per_second=1e9, burst=10**9)
    client = QuoteClient(api)
    codes = [b.code for b in market.bonds][:9]
    got = client.daily_bars(codes, market.start, market.minutes[-1])
    want = direct_daily(market, codes, market.start, market.end)
    pd.testing.assert_frame_equal(got.reset_index(drop=True), want, check_dtype=False)
    n_windows = len(date_windows(market.start, market.minutes[-1], 20))
    assert len(api.calls) == n_windows * 3


def test_retries_survive_errors_and_rate_limits(market):
    api, clock = fresh_api(market, market.end, max_daily_span_days=10, calls_per_second=1,
                           burst=1, transient_error_rate=0.3)
    client = QuoteClient(api, seed=1)
    codes = [b.code for b in market.bonds][:3]
    got = client.daily_bars(codes, market.start, market.minutes[-1])
    want = direct_daily(market, codes, market.start, market.end)
    assert len(got) == len(want)
    assert client.stats.transient > 0 and client.stats.rate_limited > 0
    assert clock.slept > 0
    ok = [c for c in api.calls if c.outcome == "ok"]
    gaps = np.diff([c.at.value for c in ok]) / 1e9
    assert (gaps >= 1.0 - 1e-9).all()          # never more than one call per second


def test_gives_up(market):
    api, _ = fresh_api(market, market.end, transient_error_rate=1.0, calls_per_second=1e9, burst=10**9)
    client = QuoteClient(api, max_retries=3)
    with pytest.raises(GaveUp):
        client.snapshot([market.bonds[0].code])
    assert len(api.calls) == 4


def test_cache_memory_and_disk(market, sessions, tmp_path):
    now = sessions[-1] + pd.Timedelta(hours=10)
    api, _ = fresh_api(market, now, max_daily_span_days=15, transient_error_rate=0,
                       calls_per_second=1e9, burst=10**9)
    codes = [market.bonds[0].code]
    c1 = QuoteClient(api, cache_dir=tmp_path)
    a = c1.daily_bars(codes, market.start, now)
    n = len(api.calls)
    b = c1.daily_bars(codes, market.start, now)
    pd.testing.assert_frame_equal(a, b)
    assert len(api.calls) == n + 1              # only the window reaching today is re-fetched
    c2 = QuoteClient(api, cache_dir=tmp_path)   # a new process reads the disk cache
    c2.daily_bars(codes, market.start, now)
    assert len(api.calls) == n + 2
    assert c2.stats.cache_hits == n - 1


def test_minute_bars_one_business_day_per_request(market, sessions):
    api, _ = fresh_api(market, market.end, transient_error_rate=0, calls_per_second=1e9, burst=10**9)
    client = QuoteClient(api)
    code = market.bonds[7].code
    start, end = sessions[-6], sessions[-1]
    got = client.minute_bars([code], start, end)
    assert len(api.calls) == len(pd.bdate_range(start, end))
    want = market.bars(code, start, end + pd.Timedelta(days=1), "1min")
    assert len(got) == len(want)
    assert pd.to_datetime(got["time"]).dt.date.nunique() == len(pd.bdate_range(start, end))
