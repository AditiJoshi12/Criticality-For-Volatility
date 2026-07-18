"""Unit tests for src/05_baseline_models.py and src/07_regime_detection.py.
Run with: pytest tests/ -v
"""
import importlib

import numpy as np
import pandas as pd

models = importlib.import_module("05_baseline_models")
regimes = importlib.import_module("07_regime_detection")

SEED = 3


def test_qlike_zero_at_truth_and_positive_elsewhere():
    rv = np.array([1e-4, 5e-4, 2e-3])
    assert np.allclose(models.qlike(rv, rv), 0)
    assert (models.qlike(rv, rv * 2) > 0).all()
    assert (models.qlike(rv, rv * 0.5) > 0).all()


def test_qlike_penalizes_underprediction_more():
    """QLIKE is known to be asymmetric: underpredicting variance costs more
    than overpredicting by the same factor."""
    rv = np.array([1e-3])
    assert models.qlike(rv, rv / 2)[0] > models.qlike(rv, rv * 2)[0]


def test_diebold_mariano_antisymmetric_and_null():
    rng = np.random.default_rng(SEED)
    a = rng.random(500)
    b = rng.random(500)
    dm_ab, p_ab = models.diebold_mariano(a, b)
    dm_ba, p_ba = models.diebold_mariano(b, a)
    assert np.isclose(dm_ab, -dm_ba)
    assert np.isclose(p_ab, p_ba)
    # identical losses + tiny noise -> insignificant
    noise = rng.normal(0, 1e-6, 500)
    _, p = models.diebold_mariano(a, a + noise)
    assert p > 0.05


def test_ewma_forecast_is_causal():
    """EWMA at time t must not move when future observations change."""
    rng = np.random.default_rng(SEED)
    idx = pd.bdate_range("2010-01-01", periods=300)
    rv1 = pd.Series(np.exp(rng.normal(-9, 0.5, 300)), index=idx)
    rv2 = rv1.copy()
    rv2.iloc[-20:] *= 100
    a = models.ewma_forecast(rv1).iloc[:-25]
    b = models.ewma_forecast(rv2).iloc[:-25]
    pd.testing.assert_series_equal(a, b)


def test_hmm_recovers_three_separated_states():
    """Three well-separated Gaussian levels with persistent switching:
    fit_hmm should recover ~3 occupied, correctly ordered states and a
    diagonal-dominant transition matrix."""
    rng = np.random.default_rng(SEED)
    means = [-11.0, -9.5, -8.0]
    n = 3000
    state, xs, states = 0, [], []
    for _ in range(n):
        states.append(state)
        xs.append(rng.normal(means[state], 0.3))
        if rng.random() < 0.03:
            state = rng.integers(0, 3)
    X = np.array(xs).reshape(-1, 1)
    labels, post, A = regimes.fit_hmm(X)
    # all three states occupied
    assert len(np.unique(labels)) == 3
    # ordering: mean observation increases with label
    m = [X[labels == i].mean() for i in range(3)]
    assert m[0] < m[1] < m[2]
    # persistence
    assert np.all(np.diag(A) > 0.5)
    # rough label accuracy vs ground truth
    acc = (labels == np.array(states)).mean()
    assert acc > 0.85, f"accuracy {acc:.2f}"
