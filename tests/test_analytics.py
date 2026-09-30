import numpy as np
import pandas as pd
import pytest

from bondlab import store
from bondlab.bonds import to_years
from bondlab.cli import main
from bondlab.curve import NelsonSiegel
from bondlab.experiment import check_times, fit_at, run_experiment, summarise
from bondlab.fit import curve_error_bp, fit_curve
from bondlab.newissue import par_yield, price_new_issue, true_fair_yield
from bondlab.pnl import COMPONENTS, build_sheet, daily_totals, default_positions, write_sheet
from bondlab.spreads import price_on_curve, z_spread


def treasuries(market):
    return [b for b in market.bonds if b.sector == "treasury"]


def test_fit_recovers_the_curve_from_exact_yields(market):
    at = market.minutes[-100]
    bonds = treasuries(market)
    y = np.array([market.fair_yield(b.code, [at])[0] for b in bonds])
    res = fit_curve(bonds, to_years(at), y)
    assert np.abs(curve_error_bp(res.curve, market.true_curve(at))).max() < 1e-6
    fixed = fit_curve(bonds, to_years(at), y, tau=market.model.tau)
    assert fixed.curve.tau == market.model.tau and fixed.rmse_bp < 1e-6
    with pytest.raises(ValueError):
        fit_curve(bonds[:3], to_years(at), y[:3])


def test_refit_from_logged_quotes(lab, market):
    times = check_times(lab["con"], 60)
    assert times and all(t.minute == 0 for t in times)
    s = summarise(run_experiment(lab["con"], market, times)).set_index("variant")
    assert s.loc["truth", "worst_bp"] < 1e-5
    assert s.loc["quotes", "fit_residual_bp"] < 1.0
    assert s.loc["quotes", "worst_bp"] < 3.0
    assert s.loc["no_long", "bonds"] == s.loc["quotes", "bonds"] - 1


def test_z_spread_round_trip(market):
    b = next(x for x in market.bonds if x.sector == "local_gov")
    c = NelsonSiegel(0.022, -0.01, -0.005, 2.0)
    t = to_years(market.minutes[-1])
    p = price_on_curve(b, t, c, 0.0042)
    assert z_spread(b, t, p, c) == pytest.approx(0.0042, abs=1e-12)


def test_par_yield_on_a_flat_curve():
    flat = NelsonSiegel(0.03, 0.0, 0.0, 1.0)
    assert par_yield(flat, 5) == pytest.approx(np.expm1(0.03), rel=1e-12)


@pytest.mark.parametrize("sector,issuer,tenor,tol", [
    ("treasury", None, 10, 2.0),
    ("local_gov", "Province A", 7, 6.0),
    ("urban_invest", "City E Urban Development Group", 3, 10.0),
])
def test_new_issue_lands_near_the_truth(lab, market, sessions, sector, issuer, tenor, tol):
    at = sessions[-1] + pd.Timedelta(hours=15)
    ni = price_new_issue(lab["con"], market, at, sector, tenor, issuer)
    truth = true_fair_yield(market, at, sector, tenor, issuer)
    assert abs(ni.fair_yield - truth) * 100 < tol
    lo, hi = ni.guidance
    assert lo < ni.guidance_mid < hi and hi - lo == pytest.approx(0.1)


def test_pnl_sheet_reconciles(lab, market, sessions):
    con = lab["con"]
    sheet = build_sheet(con, market, sessions[-30], sessions[-1])
    pos = default_positions(market)
    assert set(sheet["code"]) == set(pos)
    assert sheet["unexplained_k"].abs().max() < 1e-4
    parts = sheet[COMPONENTS].sum(axis=1)
    assert np.allclose(parts, sheet["total_k"])
    # total P&L equals the change in dirty value, scaled to the position
    b = market.by_code[sheet["code"].iloc[0]]
    g = sheet[sheet["code"] == b.code].reset_index(drop=True)
    t = [to_years(pd.Timestamp(d) + pd.Timedelta(hours=17)) for d in g["date"]]
    dirty = np.array([b.dirty_from_yield(tt, y / 100) for tt, y in zip(t, g["yield_pct"])])
    coupons = np.zeros(len(g))   # no coupon falls in a 30-session window for most bonds
    want = np.diff(dirty) * pos[b.code] * 10
    if not any((b.payment_dates > pd.Timestamp(g["date"].iloc[0])) &
               (b.payment_dates <= pd.Timestamp(g["date"].iloc[-1]))):
        assert np.allclose(g["total_k"].iloc[1:], want + coupons[1:])
    tot = daily_totals(sheet)
    assert tot["cumulative_k"].iloc[-1] == pytest.approx(sheet["total_k"].sum())
    write_sheet(con, sheet)
    by = store.run_query(con, "pnl_by_sector")
    assert by["total"].sum() == pytest.approx(sheet["total_k"].sum(), abs=0.5)


def test_attribution_tracks_the_true_curve_moves(lab, market, sessions):
    fitted = daily_totals(build_sheet(lab["con"], market, sessions[-40], sessions[-1]))
    truth = daily_totals(build_sheet(lab["con"], market, sessions[-40], sessions[-1], true_curves=True))
    curve_fit = fitted[["level_k", "slope_k", "curvature_k"]].sum(axis=1)
    curve_true = truth[["level_k", "slope_k", "curvature_k"]].sum(axis=1)
    assert np.corrcoef(curve_fit, curve_true)[0, 1] > 0.9
    assert np.allclose(fitted["total_k"], truth["total_k"])


def test_cli_end_to_end(tmp_path, capsys):
    db = str(tmp_path / "lab.sqlite")
    main(["demo", "--db", db, "--days", "30", "--log-days", "1", "--every", "300"])
    main(["query", "spread_by_tenor", "--db", db])
    main(["refit", "--db", db, "--every", "60"])
    main(["newissue", "--db", db, "--sector", "local_gov", "--tenor", "10", "--issuer", "Province B"])
    xlsx = str(tmp_path / "pnl.xlsx")
    main(["pnl", "--db", db, "--sessions", "15", "--xlsx", xlsx])
    out = capsys.readouterr().out
    assert "bid-offer by sector" in out and "fair yield" in out and "P&L attribution" in out
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.load_workbook(xlsx)
    assert wb.sheetnames == ["Daily P&L", "Positions", "By sector"]
