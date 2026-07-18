"""
06_statistical_physics.py  (Phase 4 - the core contribution)

Physics-inspired features, computed daily with strict point-in-time
discipline: every value at day t uses only data available at the close of
day t (trailing windows, past-only thresholds, past-only z-scores).

Cross-sectional features are computed on the SECTOR ETF universe (fixed,
investable, survivorship-bias-free) with a dynamic live universe: at each
date only ETFs with a fully populated trailing window participate, so
XLRE/XLC enter exactly when they listed and nothing is backfilled.

Features
  corr_length        mean pairwise correlation across live sectors
  eig_max_frac       top eigenvalue / k of the correlation matrix
                     (order parameter for the collective market mode)
  entropy_eigen      normalized Shannon entropy of the eigenvalue spectrum
  frac_eig_above_mp  fraction of eigenvalues above the Marchenko-Pastur
                     noise edge (random matrix theory: how much genuine
                     collective structure exists beyond noise)
  n_assets           size of the live universe (diagnostic, not a feature)
  entropy_returns    Shannon entropy of binned standardized target returns
  temperature        dispersion x realized vol x (1 - avg corr), z-scored
  susceptibility     rolling slope of |r_t| on |r_{t-1}| (chi = dM/dH)
  acf1               lag-1 autocorrelation of |r| (critical slowing down)
  var_of_vol         rolling variance of LOG daily RV (critical slowing down)
  hurst_vol          DFA roughness of log daily RV (rough-volatility view:
                     low H = rough; estimator validated on simulated fBm
                     in tests/)
  hawkes_branching   branching ratio n of an exponential-kernel Hawkes
                     process fit to large-move event times (seismology /
                     self-organized-criticality analogy: n -> 1 means each
                     shock triggers on average one aftershock and the
                     system is near criticality)
  criticality        composite early-warning score: mean trailing z-score
                     of the signals predicted to rise before a transition
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from common import ANN, MIN_ASSETS, PROC_DIR, SECTOR_TICKERS, TARGET

warnings.filterwarnings("ignore")

W = 63             # base rolling window (~3 months)
HURST_W = 252      # DFA window on log RV
HAWKES_W = 504     # Hawkes estimation window (~2 years of events)
HAWKES_REFIT = 5   # refit cadence in days (ffilled between refits)


def _load(name):
    return pd.read_parquet(os.path.join(PROC_DIR, name))


# ---------------------------------------------------------------------------
# Cross-sectional features on a dynamic (point-in-time) universe
# ---------------------------------------------------------------------------
def rolling_corr_features(rets, window=W, min_assets=MIN_ASSETS):
    """At each date t, use only assets whose trailing `window` is fully
    populated (listed and trading). Eigen-features are normalized by the
    live universe size k so values are comparable as k changes."""
    n = len(rets)
    cols = ["corr_length", "eig_max_frac", "entropy_eigen",
            "frac_eig_above_mp", "n_assets"]
    out = np.full((n, len(cols)), np.nan)
    vals = rets.values
    lam_plus_cache = {}
    for i in range(window, n):
        chunk = vals[i - window:i]
        live = ~np.isnan(chunk).any(axis=0)
        k = int(live.sum())
        if k < min_assets:
            continue
        X = chunk[:, live]
        sd = X.std(axis=0)
        if (sd == 0).any():
            continue
        C = np.corrcoef(X, rowvar=False)
        off = C[np.triu_indices(k, 1)]
        eig = np.clip(np.linalg.eigvalsh(C), 1e-12, None)
        p = eig / eig.sum()
        if k not in lam_plus_cache:  # Marchenko-Pastur upper edge, q = k/T
            lam_plus_cache[k] = (1 + np.sqrt(k / window)) ** 2
        out[i, 0] = off.mean()
        out[i, 1] = eig.max() / k
        out[i, 2] = -(p * np.log(p)).sum() / np.log(k)
        out[i, 3] = (eig > lam_plus_cache[k]).sum() / k
        out[i, 4] = k
    return pd.DataFrame(out, index=rets.index, columns=cols)


# ---------------------------------------------------------------------------
# Univariate features on the target asset
# ---------------------------------------------------------------------------
def shannon_entropy_returns(r, window=W, bins=10):
    def ent(x):
        z = (x - x.mean()) / (x.std() + 1e-12)
        hist, _ = np.histogram(z, bins=bins, range=(-4, 4))
        p = hist / hist.sum()
        p = p[p > 0]
        return -(p * np.log(p)).sum() / np.log(bins)
    return r.rolling(window).apply(ent, raw=True).rename("entropy_returns")


def market_temperature(rets, rv_ann, corr_length, window=21):
    disp = rets.std(axis=1)
    raw = (disp.rolling(window).mean()
           * rv_ann.rolling(window).mean()
           * (1 - corr_length).clip(lower=0))
    mu = raw.rolling(252, min_periods=126).mean()
    sd = raw.rolling(252, min_periods=126).std()
    return ((raw - mu) / (sd + 1e-12)).rename("temperature")


def susceptibility(r, window=W):
    a = r.abs()
    x, y = a.shift(1), a
    cov = x.rolling(window).cov(y)
    var = x.rolling(window).var()
    return (cov / (var + 1e-12)).rename("susceptibility")


def critical_slowing_down(r, log_rv, window=W):
    a = r.abs()
    acf1 = a.rolling(window).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if x.std() > 0 else np.nan,
        raw=True).rename("acf1")
    # variance of LOG RV: scale-free, unlike raw RV whose variance is ~1e-8
    vov = log_rv.rolling(window).var().rename("var_of_vol")
    return acf1, vov


# ---------------------------------------------------------------------------
# DFA roughness of log-volatility (rough volatility perspective)
# ---------------------------------------------------------------------------
def dfa_exponent(x, scales=(8, 13, 21, 34, 55)):
    """Detrended fluctuation analysis (order-1). For fractional Gaussian
    noise input the DFA exponent estimates the Hurst exponent H.
    Applied to log RV, low values = rough volatility paths."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    y = np.cumsum(x - x.mean())
    logs_s, logs_f = [], []
    for s in scales:
        m = n // s
        if m < 4:
            continue
        segs = y[:m * s].reshape(m, s)
        t = np.arange(s)
        # order-1 polynomial detrend per segment (vectorized)
        t_mean = t.mean()
        beta = ((segs * (t - t_mean)).sum(axis=1)
                / ((t - t_mean) ** 2).sum())
        alpha = segs.mean(axis=1) - beta * t_mean
        resid = segs - (alpha[:, None] + beta[:, None] * t)
        f2 = (resid ** 2).mean()
        if f2 <= 0:
            continue
        logs_s.append(np.log(s))
        logs_f.append(0.5 * np.log(f2))
    if len(logs_s) < 3:
        return np.nan
    return np.polyfit(logs_s, logs_f, 1)[0]


def rolling_dfa_hurst(series, window=HURST_W):
    return series.rolling(window).apply(dfa_exponent, raw=True)


# ---------------------------------------------------------------------------
# Hawkes branching ratio (seismology / SOC analogy)
# ---------------------------------------------------------------------------
def _hawkes_neg_loglik(x, t, T):
    """Exponential-kernel Hawkes: lambda(t) = mu + alpha * sum exp(-beta dt).
    Parameterized as (log mu, logit n, log beta) with alpha = n * beta, so
    the branching ratio n is constrained to (0, 1)."""
    log_mu, logit_n, log_beta = x
    mu = np.exp(log_mu)
    nbr = 1.0 / (1.0 + np.exp(-logit_n))
    beta = np.exp(log_beta)
    alpha = nbr * beta
    A = 0.0
    ll = 0.0
    for i in range(len(t)):
        if i > 0:
            A = np.exp(-beta * (t[i] - t[i - 1])) * (1.0 + A)
        ll += np.log(mu + alpha * A)
    ll -= mu * T
    ll -= nbr * np.sum(1.0 - np.exp(-beta * (T - t)))
    return -ll


def fit_hawkes_branching(event_times, T, n_starts=3):
    """MLE of the branching ratio n in (0, 1). Windows where the process is
    effectively supercritical push n toward the upper bound (~0.99+), which
    is exactly the near-criticality signal we want to capture."""
    t = np.asarray(event_times, dtype=float)
    if len(t) < 10:
        return np.nan
    best, best_val = np.nan, np.inf
    rate = len(t) / T
    starts = [(np.log(rate * (1 - n0)), np.log(n0 / (1 - n0)), np.log(b0))
              for n0, b0 in [(0.3, 0.5), (0.7, 0.2), (0.5, 1.0)][:n_starts]]
    for x0 in starts:
        try:
            res = minimize(_hawkes_neg_loglik, x0, args=(t, T),
                           method="Nelder-Mead",
                           options={"maxiter": 400, "xatol": 1e-3,
                                    "fatol": 1e-3})
            if res.fun < best_val:
                best_val = res.fun
                best = 1.0 / (1.0 + np.exp(-res.x[1]))
        except Exception:
            continue
    return best


def hawkes_branching_series(r, window=HAWKES_W, refit=HAWKES_REFIT,
                            thresh_window=252, q=0.90, min_events=20):
    """Events = days where |return| exceeds a PAST-ONLY trailing quantile.
    The branching ratio is re-estimated every `refit` days on the trailing
    `window` of events and held constant in between (causal ffill)."""
    a = r.abs()
    thr = a.rolling(thresh_window).quantile(q).shift(1)  # known at close t-1
    is_event = (a > thr) & thr.notna()
    ev_idx = np.where(is_event.values)[0]
    out = pd.Series(np.nan, index=r.index, name="hawkes_branching")
    start = thresh_window + window
    for i in range(start, len(r), refit):
        ev = ev_idx[(ev_idx > i - window) & (ev_idx <= i)]
        if len(ev) < min_events:
            continue
        t = ev - (i - window)  # event times in days within the window
        out.iloc[i] = fit_hawkes_branching(t.astype(float), float(window))
    return out.ffill(limit=refit + 2)


# ---------------------------------------------------------------------------
def main():
    rets_all = _load("log_returns.parquet")
    rv = _load("daily_rv.parquet")
    r = rets_all[TARGET].dropna()
    rv = rv.reindex(r.index).ffill()
    log_rv = np.log(rv["rv"])

    sectors = [c for c in SECTOR_TICKERS if c in rets_all.columns]
    if len(sectors) >= MIN_ASSETS:
        cross_rets = rets_all.reindex(r.index)[sectors]
        min_assets = MIN_ASSETS
        print(f"Cross-sectional universe: {len(sectors)} sector ETFs "
              f"(dynamic, min {min_assets} live)")
    else:
        cross_rets = rets_all.reindex(r.index).drop(columns=["VIX"],
                                                    errors="ignore")
        # adapt the threshold so the pipeline still runs instead of
        # silently emitting all-NaN features that empty the model matrix
        min_assets = max(2, min(MIN_ASSETS, cross_rets.shape[1]))
        print("=" * 70)
        print(f"WARNING: only {len(sectors)} sector ETF files found; falling "
              f"back to {cross_rets.shape[1]} broad index columns with "
              f"min_assets={min_assets}.")
        print("Correlation features on near-identical indices are close to "
              "degenerate;\nresults are for pipeline continuity only. For "
              "meaningful cross-sectional\nfeatures re-run: python "
              "src/01_download_data.py  (downloads the 11 sector ETFs)")
        print("=" * 70)

    corr_feats = rolling_corr_features(cross_rets, min_assets=min_assets)
    ent_r = shannon_entropy_returns(r)
    temp = market_temperature(cross_rets, rv["rv_ann"],
                              corr_feats["corr_length"])
    chi = susceptibility(r)
    acf1, vov = critical_slowing_down(r, log_rv)
    print("Estimating DFA vol-roughness (rolling)...")
    hurst_vol = rolling_dfa_hurst(log_rv).rename("hurst_vol")
    print("Estimating Hawkes branching ratio (rolling MLE)...")
    hawkes = hawkes_branching_series(r)

    feats = pd.concat([corr_feats, ent_r, temp, chi, acf1, vov,
                       hurst_vol, hawkes], axis=1)

    # Composite criticality score: mean trailing z-score of the signals
    # theory predicts should RISE before a transition. Causal by
    # construction (trailing means/stds only).
    rise = feats[["corr_length", "eig_max_frac", "frac_eig_above_mp",
                  "susceptibility", "acf1", "var_of_vol",
                  "hawkes_branching"]].copy()
    rise["entropy_eigen_inv"] = -feats["entropy_eigen"]
    z = (rise - rise.rolling(252, min_periods=126).mean()) / \
        (rise.rolling(252, min_periods=126).std() + 1e-12)
    feats["criticality"] = z.mean(axis=1)

    feats.to_parquet(os.path.join(PROC_DIR, "physics_features.parquet"))
    print(f"\nPhysics features: {feats.shape}, "
          f"non-null from {feats.dropna().index.min().date()}")
    print(feats.dropna().describe().round(3).T[["mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
