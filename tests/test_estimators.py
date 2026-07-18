"""Unit tests for src/03_realized_volatility.py (pure functions, no data
download needed). Run with:  pytest tests/ -v
"""
import importlib

import numpy as np
import pandas as pd
import pytest

rv_mod = importlib.import_module("03_realized_volatility")

N = 4000
SEED = 7


def gbm_ohlc(sigma_annual=0.20, n=N, seed=SEED):
    """Daily GBM closes with OHLC built from intraday sub-steps, so that
    range-based estimators see a genuine high/low path."""
    rng = np.random.default_rng(seed)
    sig_d = sigma_annual / np.sqrt(252)
    steps = 26  # intraday sub-steps
    dates = pd.bdate_range("2000-01-03", periods=n)
    o = np.empty(n); h = np.empty(n); l = np.empty(n); c = np.empty(n)
    price = 100.0
    for i in range(n):
        o[i] = price
        path = price * np.exp(np.cumsum(
            rng.normal(0, sig_d / np.sqrt(steps), steps)))
        h[i] = max(price, path.max())
        l[i] = min(price, path.min())
        c[i] = path[-1]
        price = c[i]
    return (pd.Series(o, dates), pd.Series(h, dates),
            pd.Series(l, dates), pd.Series(c, dates))


@pytest.fixture(scope="module")
def ohlc():
    return gbm_ohlc()


def test_close_to_close_matches_numpy():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 300))
    est = rv_mod.close_to_close(r, window=100).iloc[-1]
    expected = r.iloc[-100:].std() * np.sqrt(252)
    assert abs(est - expected) < 1e-12


def test_constant_price_gives_zero_vol():
    idx = pd.bdate_range("2020-01-01", periods=100)
    p = pd.Series(100.0, index=idx)
    r = np.log(p / p.shift(1)).dropna()
    assert rv_mod.close_to_close(r, 21).dropna().max() == 0
    assert rv_mod.parkinson(p, p, 21).dropna().max() == 0
    assert rv_mod.garman_klass(p, p, p, p, 21).dropna().max() == 0
    assert rv_mod.rogers_satchell(p, p, p, p, 21).dropna().max() == 0


def test_parkinson_deterministic():
    """If log(H/L) = x every day, Parkinson vol = sqrt(252 * x^2 / (4 ln 2))."""
    idx = pd.bdate_range("2020-01-01", periods=50)
    x = 0.02
    h = pd.Series(100 * np.exp(x), index=idx)
    l = pd.Series(100.0, index=idx)
    got = rv_mod.parkinson(h, l, window=21).dropna().iloc[-1]
    want = np.sqrt(252 * x**2 / (4 * np.log(2)))
    assert abs(got - want) < 1e-12


@pytest.mark.parametrize("est_name", ["parkinson", "garman_klass",
                                      "rogers_satchell", "yang_zhang"])
def test_estimators_recover_gbm_sigma(ohlc, est_name):
    """All range estimators should land near the true 20% annualized vol
    of a simulated GBM (loose 25% relative tolerance; they are noisy and
    discretization-biased at 26 sub-steps/day)."""
    o, h, l, c = ohlc
    fn = getattr(rv_mod, est_name)
    if est_name == "parkinson":
        est = fn(h, l, window=252)
    else:
        est = fn(o, h, l, c, window=252)
    mean_est = est.dropna().mean()
    assert 0.20 * 0.75 < mean_est < 0.20 * 1.25, f"{est_name}: {mean_est:.3f}"


def test_daily_rv_proxy_positive_and_scaled(ohlc):
    o, h, l, c = ohlc
    rv = rv_mod.daily_rv_proxy(o, h, l, c)
    assert (rv > 0).all()
    ann_vol = np.sqrt(rv.mean() * 252)
    assert 0.10 < ann_vol < 0.30  # true sigma is 0.20


def test_high_low_inversion_does_not_crash():
    """Clipping should keep estimators finite even on inconsistent rows."""
    idx = pd.bdate_range("2020-01-01", periods=60)
    rng = np.random.default_rng(1)
    c = pd.Series(100 + rng.normal(0, 1, 60).cumsum(), index=idx)
    o = c.shift(1).fillna(100)
    h, l = c * 0.99, c * 1.01  # deliberately inverted
    gk = rv_mod.garman_klass(o, h, l, c, 21).dropna()
    assert np.isfinite(gk).all()
