"""Client that works within the API's limits.

* Long date ranges are split into windows no longer than the span cap and
  requests for many bonds into batches no larger than the per-call cap.
* Rate-limit errors are retried after the delay the server asks for; transient
  errors with exponential backoff and jitter.
* Windows that lie wholly in the past are cached, so rerunning a pull costs no
  calls. A window that reaches today is not cached, since today's bars are
  still changing.
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .api import MockQuoteAPI, RateLimitError, TransientError


def date_windows(start, end, span_days: int, business_days_only: bool = False) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Split ``[start, end]`` (inclusive dates) into consecutive windows of at most ``span_days``."""
    s, e = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if e < s:
        return []
    if business_days_only and span_days == 1:
        return [(d, d) for d in pd.bdate_range(s, e)]
    out = []
    while s <= e:
        w_end = min(s + pd.Timedelta(days=span_days - 1), e)
        out.append((s, w_end))
        s = w_end + pd.Timedelta(days=1)
    return out


def batches(items: Iterable[str], size: int) -> List[List[str]]:
    items = list(items)
    return [items[i:i + size] for i in range(0, len(items), size)]


@dataclass
class ClientStats:
    requests: int = 0       # logical requests the client made
    attempts: int = 0       # calls that reached the API, retries included
    retries: int = 0
    rate_limited: int = 0
    transient: int = 0
    cache_hits: int = 0


class GaveUp(Exception):
    pass


class QuoteClient:
    def __init__(self, api: MockQuoteAPI, cache_dir: Optional[str] = None,
                 max_retries: int = 6, base_backoff: float = 0.5, seed: int = 0):
        self.api = api
        self.clock = api.clock
        self.limits = api.limits
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem = {}
        self._rng = np.random.default_rng(seed)
        self.stats = ClientStats()

    # ---- plumbing -------------------------------------------------------
    def _call(self, fn: Callable, *args):
        self.stats.requests += 1
        for attempt in range(self.max_retries + 1):
            self.stats.attempts += 1
            try:
                return fn(*args)
            except RateLimitError as err:
                self.stats.rate_limited += 1
                wait = err.retry_after
            except TransientError:
                self.stats.transient += 1
                wait = self.base_backoff * 2**attempt * (1 + self._rng.uniform())
            if attempt == self.max_retries:
                break
            self.stats.retries += 1
            self.clock.sleep(wait)
        raise GaveUp(f"{getattr(fn, '__name__', fn)} failed after {self.max_retries} retries")

    def _key(self, *parts) -> str:
        return hashlib.sha1(repr(parts).encode()).hexdigest()[:16]

    def _cache_get(self, key):
        if key in self._mem:
            return self._mem[key]
        if self.cache_dir and (self.cache_dir / f"{key}.pkl").exists():
            with open(self.cache_dir / f"{key}.pkl", "rb") as fh:
                self._mem[key] = pickle.load(fh)
            return self._mem[key]
        return None

    def _cache_put(self, key, df):
        self._mem[key] = df
        if self.cache_dir:
            with open(self.cache_dir / f"{key}.pkl", "wb") as fh:
                pickle.dump(df, fh)

    def _pull(self, endpoint: str, codes, start, end, span: int, business_only: bool) -> pd.DataFrame:
        fn = getattr(self.api, endpoint)
        today = self.clock.now().normalize()
        frames = []
        for batch in batches(sorted(set(codes)), self.limits.max_codes):
            for s, e in date_windows(start, end, span, business_only):
                closed = e < today
                key = self._key(endpoint, tuple(batch), s, e)
                df = self._cache_get(key) if closed else None
                if df is None:
                    df = self._call(fn, batch, s, e)
                    if closed:
                        self._cache_put(key, df)
                else:
                    self.stats.cache_hits += 1
                    self.stats.requests += 1
                if len(df):
                    frames.append(df)
        if not frames:
            return pd.DataFrame(columns=["code", "time"])
        out = pd.concat(frames, ignore_index=True)
        return out.drop_duplicates(["code", "time"]).sort_values(["code", "time"]).reset_index(drop=True)

    # ---- public ---------------------------------------------------------
    def daily_bars(self, codes, start, end) -> pd.DataFrame:
        """Daily bars for any date range and any number of bonds."""
        return self._pull("daily_bars", codes, start, end, self.limits.max_daily_span_days, False)

    def minute_bars(self, codes, start, end) -> pd.DataFrame:
        """One-minute bars, one request per business day per batch."""
        return self._pull("minute_bars", codes, start, end, self.limits.max_minute_span_days, True)

    def snapshot(self, codes) -> pd.DataFrame:
        frames = [self._call(self.api.snapshot, b) for b in batches(codes, self.limits.max_codes)]
        frames = [f for f in frames if len(f)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
