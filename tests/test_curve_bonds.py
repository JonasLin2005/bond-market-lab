import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from bondlab.bonds import Bond, to_years
from bondlab.curve import NelsonSiegel, ns_loadings, zero_matrix

CURVE = NelsonSiegel(0.0225, -0.009, -0.006, 1.8)


def test_short_and_long_limits():
    assert CURVE.zero(1e-8) == pytest.approx(CURVE.b0 + CURVE.b1, abs=1e-9)
    assert CURVE.zero(1e7) == pytest.approx(CURVE.b0, abs=1e-6)


def test_zero_is_average_forward():
    for t in (0.5, 3.0, 12.0, 30.0):
        avg = quad(CURVE.forward, 0, t)[0] / t
        assert CURVE.zero(t) == pytest.approx(avg, abs=1e-12)


def test_discount_and_loadings():
    t = np.array([1.0, 5.0, 10.0])
    assert np.allclose(CURVE.discount(t), np.exp(-CURVE.zero(t) * t))
    L = ns_loadings(t, 1.8)
    assert L.shape == (3, 3) and np.allclose(L[:, 0], 1)
    Z = zero_matrix(np.tile(CURVE.betas, (2, 1)), 1.8, np.tile(t, (2, 1)))
    assert np.allclose(Z, CURVE.zero(t))


def annual(coupon=0.025, issue="2020-03-15", years=10, freq=1):
    issue = pd.Timestamp(issue)
    return Bond("X", "treasury", coupon, issue, issue + pd.DateOffset(years=years), freq)


def test_schedule():
    b = annual(freq=2)
    assert b.schedule[0] == b.issue and b.schedule[-1] == b.maturity
    assert len(b.cashflows) == 20
    assert b.cashflows[-1] == pytest.approx(101.25)


def test_par_bond_on_a_coupon_date():
    b = annual(0.03)
    t = to_years(pd.Timestamp("2024-03-15"))
    assert b.accrued(t) == pytest.approx(0.0, abs=1e-12)
    assert b.dirty_from_yield(t, 0.03) == pytest.approx(100.0, abs=1e-10)


def test_accrued_is_linear_within_the_period():
    b = annual(0.04)
    s = to_years(pd.Timestamp("2024-03-15"))
    e = to_years(pd.Timestamp("2025-03-15"))
    assert b.accrued((s + e) / 2) == pytest.approx(2.0)


@pytest.mark.parametrize("freq", [1, 2])
def test_price_yield_round_trip(freq):
    b = annual(0.027, freq=freq)
    rng = np.random.default_rng(0)
    t = to_years(pd.Timestamp("2021-01-01")) + rng.uniform(0, 8, 200)
    y = rng.uniform(-0.005, 0.08, 200)
    back = b.yield_from_dirty(t, b.dirty_from_yield(t, y))
    assert np.max(np.abs(back - y)) < 1e-12
    clean = b.clean_from_yield(t, y)
    assert np.allclose(b.yield_from_clean(t, clean), y, atol=1e-12)
    assert isinstance(b.yield_from_dirty(t[0], b.dirty_from_yield(t[0], y[0])), float)


def test_duration_convexity_dv01_match_finite_differences():
    b = annual(0.025, years=30, freq=2)
    t, y, h = to_years(pd.Timestamp("2023-07-01")), 0.023, 1e-5
    p = b.dirty_from_yield(t, y)
    up, dn = b.dirty_from_yield(t, y + h), b.dirty_from_yield(t, y - h)
    assert b.modified_duration(t, y) == pytest.approx(-(up - dn) / (2 * h) / p, rel=1e-6)
    assert b.convexity(t, y) == pytest.approx((up - 2 * p + dn) / h**2 / p, rel=1e-4)
    assert b.dv01(t, y) == pytest.approx(b.dirty_from_yield(t, y - 1e-4) - p, rel=1e-2)


def test_curve_price_is_sum_of_discounted_cash_flows():
    b = annual(0.03, freq=2)
    ts = pd.Timestamp("2022-11-02")
    t = to_years(ts)
    live = b.payment_dates > ts
    ttm = to_years(b.payment_dates[live]) - t
    manual = np.sum(b.cashflows[live] * CURVE.discount(ttm))
    assert b.dirty_from_curve(t, CURVE.betas, CURVE.tau) == pytest.approx(manual, rel=1e-13)
    spread = 0.01
    with_spread = np.sum(b.cashflows[live] * np.exp(-(CURVE.zero(ttm) + spread) * ttm))
    got = b.dirty_from_curve(t, CURVE.betas, CURVE.tau, lambda k: np.full_like(k, spread))
    assert got == pytest.approx(with_spread, rel=1e-13)


def test_outside_life_raises():
    b = annual()
    with pytest.raises(ValueError):
        b.accrued(to_years(pd.Timestamp("2019-01-01")))
    with pytest.raises(ValueError):
        b.dirty_from_yield(to_years(b.maturity), 0.02)


def test_bad_bonds_rejected():
    with pytest.raises(ValueError):
        Bond("X", "treasury", 0.02, "2020-01-01", "2019-01-01")
    with pytest.raises(ValueError):
        Bond("X", "treasury", 0.02, "2020-01-01", "2030-01-01", freq=3)
