"""A mock market-data API over a simulated market, with the limits real vendors impose.

Endpoints
---------
``daily_bars(codes, start, end)``   OHLC per session; at most ``max_daily_span_days`` per request
``minute_bars(codes, start, end)``  one-minute OHLC; at most ``max_minute_span_days`` per request
``snapshot(codes)``                 the live quote *now*; there is no way to ask for a past quote

Every call counts against a rate limit (a token bucket), may carry at most
``max_codes`` securities, and fails transiently with probability
``transient_error_rate``. Dates are inclusive calendar dates, as in most vendor
manuals. Time comes from a ``SimClock`` so tests and demos never really sleep.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

import numpy as np
import pandas as pd

from .market import Market


class APIError(Exception):
    """Base class for errors the API returns."""


class SpanLimitError(APIError):
    pass


class TooManyCodesError(APIError):
    pass


class UnknownCodeError(APIError):
    pass


class RateLimitError(APIError):
    def __init__(self, retry_after: float):
        super().__init__(f"rate limit exceeded; retry after {retry_after:.3f}s")
        self.retry_after = retry_after


class TransientError(APIError):
    """A timeout or 5xx: safe to retry."""


class SimClock:
    """Simulated wall clock. ``sleep`` advances time instantly."""

    def __init__(self, start):
        self._now = pd.Timestamp(start)
        self.slept = 0.0

    def now(self) -> pd.Timestamp:
        return self._now

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self._now += pd.Timedelta(seconds=seconds)
            self.slept += seconds

    def set(self, when) -> None:
        self._now = pd.Timestamp(when)


@dataclass(frozen=True)
class Limits:
    max_daily_span_days: int = 365
    max_minute_span_days: int = 1
    max_codes: int = 10
    calls_per_second: float = 5.0
    burst: int = 5
    transient_error_rate: float = 0.02


@dataclass
class CallRecord:
    endpoint: str
    codes: tuple
    start: object
    end: object
    at: pd.Timestamp
    outcome: str


class MockQuoteAPI:
    def __init__(self, market: Market, clock: SimClock, limits: Limits = Limits(), seed: int = 0):
        self.market = market
        self.clock = clock
        self.limits = limits
        self._rng = np.random.default_rng(seed)
        self._tokens = float(limits.burst)
        self._last = clock.now()
        self.calls: List[CallRecord] = []

    # ---- gatekeeping ------------------------------------------------------
    def _admit(self, endpoint, codes, start=None, end=None):
        now = self.clock.now()
        elapsed = max((now - self._last).total_seconds(), 0.0)
        self._tokens = min(self.limits.burst, self._tokens + elapsed * self.limits.calls_per_second)
        self._last = now
        rec = CallRecord(endpoint, tuple(codes), start, end, now, "ok")
        self.calls.append(rec)
        if self._tokens < 1.0:
            rec.outcome = "rate_limited"
            raise RateLimitError((1.0 - self._tokens) / self.limits.calls_per_second)
        self._tokens -= 1.0
        if len(codes) > self.limits.max_codes:
            rec.outcome = "too_many_codes"
            raise TooManyCodesError(f"{len(codes)} codes > {self.limits.max_codes} per call")
        unknown = [c for c in codes if c not in self.market.by_code]
        if unknown:
            rec.outcome = "unknown_code"
            raise UnknownCodeError(", ".join(unknown))
        if self._rng.uniform() < self.limits.transient_error_rate:
            rec.outcome = "transient"
            raise TransientError("upstream timeout")

    @staticmethod
    def _dates(start, end):
        s, e = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        if e < s:
            raise ValueError("end before start")
        return s, e, (e - s).days + 1

    # ---- endpoints ----------------------------------------------------------
    def daily_bars(self, codes: Iterable[str], start, end) -> pd.DataFrame:
        codes = list(codes)
        s, e, span = self._dates(start, end)
        self._admit("daily_bars", codes, s, e)
        if span > self.limits.max_daily_span_days:
            self.calls[-1].outcome = "span"
            raise SpanLimitError(f"{span} days > {self.limits.max_daily_span_days} per request")
        return self._bars(codes, s, e, "1d")

    def minute_bars(self, codes: Iterable[str], start, end) -> pd.DataFrame:
        codes = list(codes)
        s, e, span = self._dates(start, end)
        self._admit("minute_bars", codes, s, e)
        if span > self.limits.max_minute_span_days:
            self.calls[-1].outcome = "span"
            raise SpanLimitError(f"{span} days > {self.limits.max_minute_span_days} per request")
        return self._bars(codes, s, e, "1min")

    def snapshot(self, codes: Iterable[str]) -> pd.DataFrame:
        codes = list(codes)
        self._admit("snapshot", codes)
        now = self.clock.now()
        rows = [r for r in (self.market.snapshot(c, now) for c in codes) if r is not None]
        return pd.DataFrame(rows)

    def _bars(self, codes, s, e, interval):
        # Data exists only up to "now": a vendor cannot return bars from the future.
        end = min(e + pd.Timedelta(days=1), self.clock.now())
        frames = [self.market.bars(c, s, end, interval) for c in codes]
        frames = [f for f in frames if len(f)]
        if not frames:
            return pd.DataFrame(columns=["code", "time", "open", "high", "low", "close"])
        return pd.concat(frames, ignore_index=True)
