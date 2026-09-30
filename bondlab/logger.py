"""Poll the snapshot-only quote endpoint on a timer and keep every snapshot in SQLite.

The endpoint only ever returns the quote as it stands now, so an intraday
history exists only if something records it. The logger polls at a fixed
interval during trading sessions, skips the lunch break and nights, and
writes each snapshot in long format.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from .client import QuoteClient
from .market import SESSIONS
from .store import snapshot_to_long, write_quotes


def session_bounds(day) -> list:
    d = pd.Timestamp(day).normalize()
    out = []
    for hhmm, n in SESSIONS:
        s = d + pd.Timedelta(hhmm + ":00")
        out.append((s, s + pd.Timedelta(minutes=n)))
    return out


def in_session(ts) -> bool:
    ts = pd.Timestamp(ts)
    if ts.weekday() >= 5:
        return False
    return any(s <= ts < e for s, e in session_bounds(ts))


def next_session_start(ts) -> pd.Timestamp:
    """The next moment at or after ``ts`` that is inside a session."""
    ts = pd.Timestamp(ts)
    if in_session(ts):
        return ts
    day = ts.normalize()
    for _ in range(10):
        if day.weekday() < 5:
            for s, _e in session_bounds(day):
                if s >= ts:
                    return s
        day += pd.Timedelta(days=1)
        ts = day
    raise RuntimeError("no session found")  # pragma: no cover


@dataclass
class LogStats:
    polls: int = 0
    rows: int = 0


class QuoteLogger:
    def __init__(self, client: QuoteClient, con: sqlite3.Connection):
        self.client = client
        self.clock = client.clock
        self.con = con
        self.stats = LogStats()

    def poll(self, codes: Iterable[str]) -> int:
        """Take one snapshot of ``codes`` and store it. Returns rows written."""
        poll_time = self.clock.now().floor("s")
        snap = self.client.snapshot(list(codes))
        n = write_quotes(self.con, snapshot_to_long(snap, poll_time))
        self.stats.polls += 1
        self.stats.rows += n
        return n

    def run(self, codes: Iterable[str], until, every_s: float = 60.0,
            max_polls: Optional[int] = None) -> LogStats:
        """Poll every ``every_s`` seconds of session time until ``until``."""
        codes = list(codes)
        until = pd.Timestamp(until)
        while True:
            start = next_session_start(self.clock.now())
            if start >= until or (max_polls is not None and self.stats.polls >= max_polls):
                break
            if start > self.clock.now():
                self.clock.set(start)
            tick = self.clock.now()
            self.poll(codes)
            # keep a fixed cadence: retries inside the poll do not shift the grid
            self.clock.set(max(self.clock.now(), tick + pd.Timedelta(seconds=every_s)))
        return self.stats
