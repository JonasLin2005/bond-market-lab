"""A simulated interbank bond market.

The true treasury zero curve is Nelson-Siegel with factors that follow
independent Ornstein-Uhlenbeck processes on a one-minute trading clock (a
continuous-time version of the Diebold-Li dynamic model, with the decay fixed).
Policy-bank bonds are priced off the same curve plus a spread that widens with
maturity.

Brokers post two-sided quotes in yield at random times: the mid is the bond's
fair yield plus noise, the bid-offer width depends on the bond's liquidity, and
some quotes are one-sided. Trades arrive less often and print at the prevailing
bid or offer. Everything is driven by one seed, so a market is reproducible
from ``(start, days, seed)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .bonds import Bond, to_years
from .curve import NelsonSiegel

# Two sessions a day, as on most Asian bond platforms: 09:00-12:00, 13:30-17:00.
SESSIONS = (("09:00", 180), ("13:30", 210))
MINUTES_PER_DAY = sum(n for _, n in SESSIONS)
_MINUTE_YEARS = 1.0 / (252 * MINUTES_PER_DAY)   # OU time step, in trading years

QUOTE_FIELDS = (
    "quote_time", "bid_yield", "offer_yield", "bid_clean", "offer_clean",
    "bid_size", "offer_size", "mid_yield", "last_trade_time", "last_yield",
    "last_clean", "last_volume", "accrued",
)


@dataclass(frozen=True)
class FactorModel:
    mean: tuple = (0.0225, -0.0090, -0.0060)   # long end ~2.25%, short end ~1.35%
    kappa: tuple = (0.6, 0.9, 1.2)             # mean reversion, per year
    sigma: tuple = (0.0035, 0.0045, 0.0070)    # annualised volatility
    tau: float = 1.8                            # decay, years


SECTORS = ("treasury", "policy_bank", "local_gov", "urban_invest", "bank_capital")


@dataclass(frozen=True)
class Liquidity:
    quotes_per_min: float
    trades_per_min: float
    spread_bp: float  # typical bid-offer width in yield


@dataclass(frozen=True)
class SectorSpread:
    """Spread over the treasury zero curve, continuously compounded.

    spread(ttm, time) = issuer base + term * (1 - e^{-ttm/5}) + sector factor(time),
    where the sector factor is an Ornstein-Uhlenbeck process around zero.
    """
    term_bp: float
    factor_kappa: float = 2.0
    factor_sigma_bp: float = 0.0   # annualised


SECTOR_SPREADS = {
    "treasury": SectorSpread(0.0),
    "policy_bank": SectorSpread(10.0, factor_sigma_bp=4.0),
    "local_gov": SectorSpread(15.0, factor_sigma_bp=6.0),
    "urban_invest": SectorSpread(35.0, factor_sigma_bp=25.0),
    "bank_capital": SectorSpread(20.0, factor_sigma_bp=15.0),
}


@dataclass(frozen=True)
class BookEntry:
    sector: str
    tenor: int              # original term, years
    remaining: float        # years left at the end of the simulation
    liquidity: Liquidity
    issuer: str = ""
    base_bp: float = 0.0    # issuer-specific spread level


def _L(q, t, s):
    return Liquidity(q, t, s)


# Fictional issuers only. Bank capital bonds are Tier 2 notes treated as bullets
# to their first call date, which is how the market usually prices them.
DEFAULT_BOOK = (
    BookEntry("treasury", 1, 0.55, _L(0.25, 0.020, 1.5), "Treasury"),
    BookEntry("treasury", 2, 1.40, _L(0.25, 0.020, 1.2), "Treasury"),
    BookEntry("treasury", 3, 2.20, _L(0.30, 0.030, 1.0), "Treasury"),
    BookEntry("treasury", 5, 3.60, _L(0.25, 0.020, 1.2), "Treasury"),
    BookEntry("treasury", 5, 4.70, _L(0.40, 0.050, 0.8), "Treasury"),
    BookEntry("treasury", 7, 6.60, _L(0.30, 0.030, 1.0), "Treasury"),
    BookEntry("treasury", 10, 8.30, _L(0.25, 0.020, 1.2), "Treasury"),
    BookEntry("treasury", 10, 9.60, _L(0.80, 0.120, 0.5), "Treasury"),
    BookEntry("treasury", 20, 18.90, _L(0.25, 0.020, 1.8), "Treasury"),
    BookEntry("treasury", 30, 28.80, _L(0.60, 0.080, 0.8), "Treasury"),
    BookEntry("policy_bank", 1, 0.80, _L(0.20, 0.015, 2.0), "Policy Bank A", 2.0),
    BookEntry("policy_bank", 3, 2.60, _L(0.30, 0.030, 1.5), "Policy Bank A", 2.0),
    BookEntry("policy_bank", 5, 4.40, _L(0.30, 0.030, 1.5), "Policy Bank A", 2.0),
    BookEntry("policy_bank", 7, 6.80, _L(0.20, 0.015, 2.5), "Policy Bank B", 4.0),
    BookEntry("policy_bank", 10, 9.50, _L(0.60, 0.080, 0.8), "Policy Bank A", 2.0),
    BookEntry("local_gov", 5, 4.20, _L(0.12, 0.010, 3.0), "Province A", 12.0),
    BookEntry("local_gov", 10, 8.80, _L(0.15, 0.012, 3.0), "Province B", 16.0),
    BookEntry("local_gov", 10, 9.30, _L(0.12, 0.010, 3.5), "Province A", 12.0),
    BookEntry("local_gov", 20, 17.50, _L(0.08, 0.006, 4.0), "Province C", 20.0),
    BookEntry("urban_invest", 3, 1.60, _L(0.06, 0.004, 8.0), "City D Urban Construction Investment", 150.0),
    BookEntry("urban_invest", 3, 2.40, _L(0.08, 0.006, 6.0), "City E Urban Development Group", 95.0),
    BookEntry("urban_invest", 5, 3.80, _L(0.06, 0.004, 7.0), "City E Urban Development Group", 95.0),
    BookEntry("urban_invest", 5, 4.60, _L(0.05, 0.004, 9.0), "County F Construction Investment", 210.0),
    BookEntry("bank_capital", 5, 2.90, _L(0.10, 0.008, 4.0), "Bank G", 45.0),
    BookEntry("bank_capital", 5, 4.10, _L(0.10, 0.008, 4.0), "Bank H", 60.0),
)

_PREFIX = {"treasury": "TB", "policy_bank": "PB", "local_gov": "LG",
           "urban_invest": "UI", "bank_capital": "BC"}
_LABEL = {"treasury": "treasury", "policy_bank": "policy bank", "local_gov": "local government",
          "urban_invest": "urban investment", "bank_capital": "Tier 2"}


def trading_minutes(start, days: int) -> pd.DatetimeIndex:
    """Start of every trading minute over ``days`` business days from ``start``."""
    out = []
    for d in pd.bdate_range(pd.Timestamp(start).normalize(), periods=days):
        for hhmm, n in SESSIONS:
            out.append(pd.date_range(d + pd.Timedelta(hhmm + ":00"), periods=n, freq="min"))
    return out[0].append(out[1:]).as_unit("ns")


def _ou(rng, n: int, mean: float, kappa: float, sigma: float) -> np.ndarray:
    """Ornstein-Uhlenbeck path on the one-minute trading clock (exact AR(1) steps)."""
    if sigma == 0:
        return np.full(n, float(mean))
    phi = np.exp(-kappa * _MINUTE_YEARS)
    e = rng.standard_normal(n) * sigma * np.sqrt((1 - phi**2) / (2 * kappa))
    e[0] = 0.0
    return mean + lfilter([1.0], [1.0, -phi], e)


def _ns(times) -> np.ndarray:
    """Nanosecond integers, so comparisons never depend on pandas' inferred unit."""
    if isinstance(times, pd.Timestamp):
        return np.array([times.as_unit("ns").value])
    return pd.DatetimeIndex(times).as_unit("ns").asi8


def static_spread(sector: str, base_bp: float, ttm):
    """Time-invariant part of the spread, as a decimal."""
    sp = SECTOR_SPREADS[sector]
    return (base_bp + sp.term_bp * (1.0 - np.exp(-np.asarray(ttm, float) / 5.0))) * 1e-4


def _par_coupon(curve: NelsonSiegel, tenor: float, freq: int, spread=None) -> float:
    k = np.arange(1, int(round(tenor * freq)) + 1) / freq
    z = curve.zero(k) + (spread(k) if spread else 0.0)
    df = np.exp(-z * k)
    return float(freq * (1 - df[-1]) / df.sum())


def build_book(end, model: FactorModel, rng: np.random.Generator, spec=DEFAULT_BOOK):
    mean_curve = NelsonSiegel(*model.mean, model.tau)
    end = pd.Timestamp(end).normalize()
    bonds, liq, base, seq = [], {}, {}, {}
    for e in spec:
        maturity = (end + pd.DateOffset(months=int(round(e.remaining * 12)))).replace(day=15)
        issue = maturity - pd.DateOffset(years=e.tenor)
        freq = 2 if (e.sector == "treasury" and e.tenor >= 10) else 1
        spread = (lambda k, e=e: static_spread(e.sector, e.base_bp, k)) if e.sector != "treasury" else None
        coupon = _par_coupon(mean_curve, e.tenor, freq, spread) + rng.normal(0, 0.0015)
        coupon = round(max(coupon, 0.005), 4)
        prefix = _PREFIX[e.sector]
        n = seq[(prefix, issue.year)] = seq.get((prefix, issue.year), 0) + 1
        code = f"{prefix}{issue:%y}{n:02d}"
        name = f"{e.tenor}Y {_LABEL[e.sector]} {coupon * 100:.2f}% {maturity:%b %Y}"
        b = Bond(code, e.sector, coupon, issue, maturity, freq, name, e.issuer)
        bonds.append(b)
        liq[code] = e.liquidity
        base[code] = e.base_bp
    return bonds, liq, base


class Market:
    """One reproducible simulated market."""

    def __init__(self, start="2024-07-01", days: int = 540, seed: int = 0,
                 model: FactorModel = FactorModel(), quote_noise_bp: float = 0.3,
                 one_sided_prob: float = 0.05, spec=DEFAULT_BOOK):
        self.model = model
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.minutes = trading_minutes(start, days)
        self.minute_t = to_years(self.minutes)
        self.factors = self._simulate_factors(rng)
        self.sector_factors = {
            name: _ou(rng, len(self.minutes), 0.0, sp.factor_kappa, sp.factor_sigma_bp * 1e-4)
            for name, sp in SECTOR_SPREADS.items()
        }
        self.start, self.end = self.minutes[0], self.minutes[-1] + pd.Timedelta(minutes=1)
        self.bonds, self.liquidity, self.base_bp = build_book(self.minutes[-1], model, rng, spec)
        self.by_code: Dict[str, Bond] = {b.code: b for b in self.bonds}
        self.quote_noise = quote_noise_bp * 1e-4
        self.one_sided_prob = one_sided_prob
        self.quotes: Dict[str, pd.DataFrame] = {}
        self.trades: Dict[str, pd.DataFrame] = {}
        for b in self.bonds:
            self.quotes[b.code], self.trades[b.code] = self._simulate_bond(b, rng)

    # ---- truth ------------------------------------------------------------
    def _simulate_factors(self, rng) -> np.ndarray:
        n = len(self.minutes)
        return np.column_stack([
            _ou(rng, n, m, k, sg) for m, k, sg in zip(self.model.mean, self.model.kappa, self.model.sigma)
        ])

    def _minute_index(self, times) -> np.ndarray:
        """Index of the trading minute containing each time (last one at or before it)."""
        idx = np.searchsorted(self.minutes.asi8, _ns(times), side="right") - 1
        return np.clip(idx, 0, len(self.minutes) - 1)

    def true_curve(self, time) -> NelsonSiegel:
        i = int(self._minute_index([pd.Timestamp(time)])[0])
        return NelsonSiegel(*self.factors[i], self.model.tau)

    def spread_fn(self, sector: str, base_bp: float, times):
        """True spread as a function of time-to-cash-flow, one row per time in ``times``."""
        if sector == "treasury":
            return None
        f = self.sector_factors[sector][self._minute_index(times)]
        return lambda ttm: static_spread(sector, base_bp, ttm) + f[:, None]

    def true_sector_spread(self, sector: str, base_bp: float, time, ttm) -> np.ndarray:
        """True spread (decimal) of a ``sector`` issuer with ``base_bp`` at ``time`` for maturities ``ttm``."""
        if sector == "treasury":
            return np.zeros_like(np.asarray(ttm, float))
        f = self.sector_factors[sector][self._minute_index([pd.Timestamp(time)])[0]]
        return static_spread(sector, base_bp, ttm) + f

    def fair_yield(self, code: str, times) -> np.ndarray:
        """Yield to maturity implied by the true curve (plus spread) at ``times``."""
        b = self.by_code[code]
        times = pd.DatetimeIndex(times)
        t = to_years(times)
        betas = self.factors[self._minute_index(times)]
        spread = self.spread_fn(b.sector, self.base_bp[code], times)
        dirty = b.dirty_from_curve(t, betas, self.model.tau, spread)
        return b.yield_from_dirty(t, dirty)

    def alive_minutes(self, bond: Bond) -> np.ndarray:
        return np.flatnonzero((self.minutes >= bond.issue) & (self.minutes < bond.maturity))

    # ---- microstructure ------------------------------------------------
    def _event_times(self, rng, mins: np.ndarray, rate: float) -> pd.DatetimeIndex:
        counts = rng.poisson(rate, len(mins))
        idx = np.repeat(mins, counts)
        ms = rng.integers(0, 60_000, len(idx))
        times = self.minutes[idx] + pd.to_timedelta(ms, unit="ms")
        return times.sort_values().as_unit("ns")

    def _simulate_bond(self, b: Bond, rng):
        lq = self.liquidity[b.code]
        mins = self.alive_minutes(b)
        qt = self._event_times(rng, mins, lq.quotes_per_min)
        mid = self.fair_yield(b.code, qt) + rng.normal(0, self.quote_noise, len(qt))
        half = 0.5e-4 * lq.spread_bp * rng.lognormal(0, 0.35, len(qt))
        bid, offer = mid + half, mid - half          # bid yield is the higher one
        u = rng.uniform(size=len(qt))
        bid[u < self.one_sided_prob / 2] = np.nan
        offer[(u >= self.one_sided_prob / 2) & (u < self.one_sided_prob)] = np.nan
        sizes = np.array([10.0, 20.0, 30.0, 50.0, 100.0])  # millions of face
        quotes = pd.DataFrame({
            "time": qt, "bid_yield": bid, "offer_yield": offer,
            "bid_size": np.where(np.isnan(bid), np.nan, rng.choice(sizes, len(qt))),
            "offer_size": np.where(np.isnan(offer), np.nan, rng.choice(sizes, len(qt))),
        })

        tt = self._event_times(rng, mins, lq.trades_per_min)
        qi = np.searchsorted(_ns(quotes["time"]), _ns(tt), side="right") - 1
        keep = qi >= 0
        tt, qi = tt[keep], qi[keep]
        buy = rng.uniform(size=len(tt)) < 0.5       # buyer lifts the offer
        q_bid = quotes["bid_yield"].to_numpy()[qi]
        q_off = quotes["offer_yield"].to_numpy()[qi]
        y = np.where(buy, q_off, q_bid)
        y = np.where(np.isnan(y), np.where(buy, q_bid, q_off), y)
        vol = rng.choice(np.array([10.0, 20.0, 50.0, 100.0, 200.0]), len(tt))
        t_years = to_years(tt)
        clean = b.clean_from_yield(t_years, y) if len(tt) else np.array([])
        trades = pd.DataFrame({"time": tt, "yield": y, "clean": clean,
                               "volume": vol, "side": np.where(buy, "buy", "sell")})
        return quotes, trades

    # ---- views used by the API -------------------------------------------
    def _arrays(self, code):
        """Numpy views of one bond's quotes and trades, built once."""
        cache = self.__dict__.setdefault("_arr", {})
        if code not in cache:
            q, tr = self.quotes[code], self.trades[code]
            cache[code] = dict(
                qt=_ns(q["time"]), bid=q["bid_yield"].to_numpy(), offer=q["offer_yield"].to_numpy(),
                bs=q["bid_size"].to_numpy(), os=q["offer_size"].to_numpy(),
                tt=_ns(tr["time"]), ty=tr["yield"].to_numpy(), tc=tr["clean"].to_numpy(),
                tv=tr["volume"].to_numpy(),
            )
        return cache[code]

    def snapshot(self, code: str, now) -> Optional[dict]:
        """Latest quote and last trade at ``now``; None before the first quote."""
        b = self.by_code[code]
        now = pd.Timestamp(now)
        a = self._arrays(code)
        now_ns = _ns(now)[0]
        i = int(np.searchsorted(a["qt"], now_ns, side="right")) - 1
        if i < 0 or now >= b.maturity:
            return None
        t = to_years(now)
        bid, offer = a["bid"][i], a["offer"][i]
        ys = np.array([bid, offer])
        ok = ~np.isnan(ys)
        prices = np.full(2, np.nan)
        if ok.any():
            prices[ok] = b.clean_from_yield(np.full(ok.sum(), t), ys[ok])
        out = {
            "code": code, "quote_time": pd.Timestamp(a["qt"][i]),
            "bid_yield": bid * 100, "offer_yield": offer * 100,
            "bid_clean": prices[0], "offer_clean": prices[1],
            "bid_size": a["bs"][i], "offer_size": a["os"][i],
            "mid_yield": (bid + offer) * 50,     # NaN when one-sided
        }
        j = int(np.searchsorted(a["tt"], now_ns, side="right")) - 1
        if j >= 0:
            out.update(last_trade_time=pd.Timestamp(a["tt"][j]), last_yield=a["ty"][j] * 100,
                       last_clean=a["tc"][j], last_volume=a["tv"][j])
        else:
            out.update(last_trade_time=pd.NaT, last_yield=np.nan, last_clean=np.nan, last_volume=np.nan)
        out["accrued"] = b.accrued(t)
        return out

    def bars(self, code: str, start, end, interval: str) -> pd.DataFrame:
        """OHLC bars built from trades in ``[start, end)``; ``interval`` is "1d" or "1min".

        ``close_mid_yield`` is the mid of the last two-sided quote before the bar
        closes: a steadier close than the last print, which sits on the bid or
        the offer and may be hours old.
        """
        tr = self.trades[code]
        tr = tr[(tr["time"] >= pd.Timestamp(start)) & (tr["time"] < pd.Timestamp(end))]
        cols = ["open", "high", "low", "close", "open_yield", "high_yield",
                "low_yield", "close_yield", "close_mid_yield", "volume", "n_trades"]
        if tr.empty:
            return pd.DataFrame(columns=["code", "time"] + cols)
        key = tr["time"].dt.floor("D" if interval == "1d" else "min")
        g = tr.groupby(key)
        out = pd.DataFrame({
            "open": g["clean"].first(), "high": g["clean"].max(),
            "low": g["clean"].min(), "close": g["clean"].last(),
            "open_yield": g["yield"].first() * 100, "high_yield": g["yield"].max() * 100,
            "low_yield": g["yield"].min() * 100, "close_yield": g["yield"].last() * 100,
            "volume": g["volume"].sum(), "n_trades": g.size(),
        })
        out.index.name = "time"
        step = pd.Timedelta(days=1) if interval == "1d" else pd.Timedelta(minutes=1)
        q = self.quotes[code].dropna(subset=["bid_yield", "offer_yield"])
        qi = np.searchsorted(_ns(q["time"]), _ns(out.index + step), side="left") - 1
        mids = (q["bid_yield"].to_numpy() + q["offer_yield"].to_numpy()) * 50
        out.insert(8, "close_mid_yield", np.where(qi >= 0, mids[np.maximum(qi, 0)], np.nan))
        out = out.reset_index()
        out.insert(0, "code", code)
        return out

    def describe_book(self) -> pd.DataFrame:
        end = self.minutes[-1]
        return pd.DataFrame([{
            "code": b.code, "sector": b.sector, "issuer": b.issuer, "name": b.name,
            "coupon": b.coupon * 100,
            "freq": b.freq, "issue": b.issue.date(), "maturity": b.maturity.date(),
            "years_left": round(float(b.remaining_years(to_years(end))), 2),
        } for b in self.bonds])
