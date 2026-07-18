"""Unit tests for src/06_statistical_physics.py. Run with: pytest tests/ -v"""
import importlib

import numpy as np
import pandas as pd

phys = importlib.import_module("06_statistical_physics")

SEED = 11


def _idx(n):
    return pd.bdate_range("2005-01-03", periods=n)


def test_hurst_white_noise_near_half():
    rng = np.random.default_rng(SEED)
    r = pd.Series(rng.normal(0, 0.01, 3000), index=_idx(3000))
    h = phys.hurst_rolling(r, window=504).dropna()
    assert 0.35 < h.mean() < 0.65, f"mean Hurst {h.mean():.3f}"


def test_hurst_persistent_series_above_half():
    """AR(1) returns with phi=0.6 are positively persistent -> H > 0.5
    on average (aggregated-variance estimator)."""
    rng = np.random.default_rng(SEED)
    n = 3000
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = 0.6 * r[t - 1] + rng.normal(0, 0.01)
    h = phys.hurst_rolling(pd.Series(r, index=_idx(n)), window=504).dropna()
    assert h.mean() > 0.55, f"mean Hurst {h.mean():.3f}"


def test_entropy_bounds_and_ordering():
    rng = np.random.default_rng(SEED)
    n = 800
    # heavy near-constant series -> most mass in one bin -> low entropy
    spiky = pd.Series(np.where(rng.random(n) < 0.95, 0.0001,
                               rng.normal(0, 0.05, n)), index=_idx(n))
    normal = pd.Series(rng.normal(0, 0.01, n), index=_idx(n))
    e_spiky = phys.shannon_entropy_returns(spiky).dropna()
    e_norm = phys.shannon_entropy_returns(normal).dropna()
    for e in (e_spiky, e_norm):
        assert ((e >= 0) & (e <= 1)).all()
    assert e_spiky.mean() < e_norm.mean()


def test_susceptibility_recovers_ar_slope():
    """Construct |r_t| = 0.7|r_{t-1}| + noise; the rolling regression slope
    should average near 0.7."""
    rng = np.random.default_rng(SEED)
    n = 3000
    a = np.zeros(n)
    for t in range(1, n):
        a[t] = 0.7 * a[t - 1] + abs(rng.normal(0, 0.01))
    r = pd.Series(a * np.sign(rng.normal(size=n)), index=_idx(n))
    chi = phys.susceptibility(r, window=252).dropna()
    assert 0.55 < chi.mean() < 0.85, f"mean chi {chi.mean():.3f}"


def test_rolling_corr_features_two_regimes():
    """Assets driven by a strong common factor should show higher
    corr_length and lower eigen-entropy than independent assets."""
    rng = np.random.default_rng(SEED)
    n, k = 400, 4
    common = rng.normal(0, 0.01, n)
    coupled = pd.DataFrame(
        common[:, None] + 0.2 * rng.normal(0, 0.01, (n, k)), index=_idx(n))
    indep = pd.DataFrame(rng.normal(0, 0.01, (n, k)), index=_idx(n))
    f_c = phys.rolling_corr_features(coupled).dropna()
    f_i = phys.rolling_corr_features(indep).dropna()
    assert f_c["corr_length"].mean() > f_i["corr_length"].mean()
    assert f_c["entropy_eigen"].mean() < f_i["entropy_eigen"].mean()
    assert f_c["eig_max_frac"].mean() > f_i["eig_max_frac"].mean()


def test_no_look_ahead():
    """Changing FUTURE data must not change feature values in the past.
    This guards the walk-forward validity of every rolling feature."""
    rng = np.random.default_rng(SEED)
    n = 900
    r1 = pd.Series(rng.normal(0, 0.01, n), index=_idx(n))
    r2 = r1.copy()
    r2.iloc[-50:] = 0.10  # rewrite the future

    cut = n - 60  # compare well before the modification
    for fn, kw in [(phys.shannon_entropy_returns, {}),
                   (phys.susceptibility, {}),
                   (phys.hurst_rolling, {"window": 252})]:
        a = fn(r1, **kw).iloc[:cut]
        b = fn(r2, **kw).iloc[:cut]
        pd.testing.assert_series_equal(a, b, check_names=False)
