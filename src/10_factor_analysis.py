"""
10_factor_analysis.py  (Phase 6b - alpha-research-style feature evaluation)

Treats every physics feature as a candidate "factor" and evaluates it the
way a quant alpha team would, adapted from the cross-sectional setting to
this time-series problem:

  IC decay        Spearman rank IC of feature_t vs the CHANGE in future
                  log RV over horizons h = 1..22 days, with Newey-West
                  t-statistics. Using the change (future minus current
                  level) rather than the level prevents a feature from
                  scoring well merely by co-moving with today's volatility.
  Orthogonalized  Each feature is residualized, walk-forward, against the
  IC              baseline information set (HAR terms + log VIX): an
                  expanding-window OLS is refit monthly and only trailing
                  coefficients produce the residual. The IC of the residual
                  is the feature's MARGINAL information beyond what the
                  baseline already knows -- the analogue of neutralizing a
                  new alpha against the existing factor pool.
  Quantile        Days are bucketed by the feature's PAST-ONLY trailing
  analysis        quintile (252d window, shifted); mean forward RV change
                  per bucket plus a monotonicity score. A real signal
                  should order the buckets.
  Redundancy      Correlation matrix across features (is the pool actually
                  diverse, or ten copies of one signal?).

Outputs: results/factor_ic.csv, factor_ic_orth.csv, factor_quantiles.csv,
feature_corr.csv, fig7_ic_decay.png, fig8_feature_corr.png.

All conditioning information is point-in-time: trailing quantile edges,
walk-forward orthogonalization betas. The future enters only through the
evaluation target, as it must.
"""
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from common import PROC_DIR, RES_DIR

warnings.filterwarnings("ignore")

FEATURES = ["corr_length", "eig_max_frac", "entropy_eigen",
            "frac_eig_above_mp", "entropy_returns", "temperature",
            "susceptibility", "acf1", "var_of_vol", "hurst_vol",
            "hawkes_branching", "criticality"]
BASELINE = ["rv_d", "rv_w", "rv_m", "log_vix"]
IC_HORIZONS = list(range(1, 23))
MIN_TRAIN = 750
REFIT = 21
Q_WINDOW = 252
N_Q = 5
plt.rcParams.update({"figure.dpi": 120, "font.size": 8})


# ---------------------------------------------------------------------------
def forward_logrv_change(log_rv, h):
    """log mean daily RV over t+1..t+h minus log RV_t (change, not level)."""
    rv = np.exp(log_rv)
    fwd = np.log(rv.rolling(h).mean().shift(-h))
    return fwd - log_rv


def nw_tstat(d, lag=None):
    """Newey-West t-stat that the mean of series d is zero."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 50:
        return np.nan
    if lag is None:
        lag = int(np.floor(1.5 * n ** (1 / 3)))
    s = np.var(d, ddof=0)
    for k in range(1, lag + 1):
        cov = np.cov(d[k:], d[:-k], ddof=0)[0, 1]
        s += 2 * (1 - k / (lag + 1)) * cov
    return d.mean() / np.sqrt(s / n)


def rank_ic(x, y):
    """Spearman IC plus a NW t-stat computed on the demeaned rank products
    (accounts for serial dependence, unlike the naive Spearman p-value)."""
    m = x.notna() & y.notna()
    if m.sum() < 100:
        return np.nan, np.nan
    rx = x[m].rank()
    ry = y[m].rank()
    ic = sps.spearmanr(rx, ry).statistic
    zx = (rx - rx.mean()) / rx.std()
    zy = (ry - ry.mean()) / ry.std()
    return ic, nw_tstat((zx * zy).values)


def walk_forward_residual(feature, X):
    """Expanding-window OLS residual of `feature` on baseline columns `X`,
    refit every REFIT days; residual at t uses only betas fit on data < t."""
    data = pd.concat([feature.rename("f"), X], axis=1).dropna()
    resid = pd.Series(np.nan, index=data.index)
    beta, mu = None, None
    Xv = np.column_stack([np.ones(len(data)), data[X.columns].values])
    yv = data["f"].values
    for i in range(MIN_TRAIN, len(data)):
        if beta is None or (i - MIN_TRAIN) % REFIT == 0:
            beta, *_ = np.linalg.lstsq(Xv[:i], yv[:i], rcond=None)
        resid.iloc[i] = yv[i] - Xv[i] @ beta
    return resid.reindex(feature.index)


def trailing_quantile_bucket(x, window=Q_WINDOW, n_q=N_Q):
    """Bucket x_t by PAST-ONLY trailing quantile edges (edges from
    t-window..t-1). Returns integer buckets 0..n_q-1 (NaN where edges
    unavailable)."""
    qs = np.linspace(0, 1, n_q + 1)[1:-1]
    edges = pd.concat(
        [x.rolling(window).quantile(q).shift(1) for q in qs], axis=1)
    b = pd.Series(np.nan, index=x.index)
    valid = x.notna() & edges.notna().all(axis=1)
    e = edges[valid].values
    xv = x[valid].values
    b[valid] = (xv[:, None] > e).sum(axis=1)
    return b


# ---------------------------------------------------------------------------
def main():
    os.makedirs(RES_DIR, exist_ok=True)
    df = pd.read_parquet(os.path.join(PROC_DIR, "model_matrix.parquet"))
    if len(df) < MIN_TRAIN + 250:
        raise SystemExit(
            f"Model matrix has only {len(df)} rows; need at least "
            f"MIN_TRAIN + 250 for walk-forward evaluation. Re-run the "
            f"data download and steps 02-06, then 04.")
    rv = pd.read_parquet(os.path.join(PROC_DIR, "daily_rv.parquet"))["rv"]
    log_rv = np.log(rv).reindex(df.index)
    feats = [f for f in FEATURES if f in df.columns]
    baseline = [b for b in BASELINE if b in df.columns]

    targets = {h: forward_logrv_change(log_rv, h) for h in IC_HORIZONS}

    # ---------- raw and orthogonalized IC decay ----------
    ic_rows, orth_rows = [], []
    resids = {}
    print("Computing IC decay (raw and orthogonalized vs "
          f"[{', '.join(baseline)}])...")
    for f in feats:
        resids[f] = walk_forward_residual(df[f], df[baseline])
        for h in IC_HORIZONS:
            ic, t = rank_ic(df[f], targets[h])
            ic_rows.append({"feature": f, "horizon": h, "IC": ic, "t_NW": t})
            ic_o, t_o = rank_ic(resids[f], targets[h])
            orth_rows.append({"feature": f, "horizon": h, "IC": ic_o,
                              "t_NW": t_o})
    ic_df = pd.DataFrame(ic_rows)
    orth_df = pd.DataFrame(orth_rows)
    ic_df.to_csv(os.path.join(RES_DIR, "factor_ic.csv"), index=False)
    orth_df.to_csv(os.path.join(RES_DIR, "factor_ic_orth.csv"), index=False)

    summary = pd.DataFrame({
        "IC_h5_raw": ic_df[ic_df.horizon == 5].set_index("feature")["IC"],
        "t_h5_raw": ic_df[ic_df.horizon == 5].set_index("feature")["t_NW"],
        "IC_h5_orth": orth_df[orth_df.horizon == 5].set_index("feature")["IC"],
        "t_h5_orth": orth_df[orth_df.horizon == 5].set_index("feature")["t_NW"],
    }).sort_values("IC_h5_orth", ascending=False)
    print("\nIC at h=5 (raw vs orthogonalized; |t_NW| > 2 is the bar):")
    print(summary.round(3))

    # ---------- quantile / monotonicity analysis (h=5) ----------
    q_rows = []
    y5 = targets[5]
    for f in feats:
        b = trailing_quantile_bucket(df[f])
        for q in range(N_Q):
            sel = (b == q) & y5.notna()
            q_rows.append({"feature": f, "quintile": q + 1,
                           "mean_fwd_dlogrv": y5[sel].mean(),
                           "n": int(sel.sum())})
    qdf = pd.DataFrame(q_rows)
    mono = (qdf.groupby("feature")
            .apply(lambda g: sps.spearmanr(g["quintile"],
                                           g["mean_fwd_dlogrv"]).statistic)
            .rename("monotonicity"))
    qdf = qdf.merge(mono, on="feature")
    qdf.to_csv(os.path.join(RES_DIR, "factor_quantiles.csv"), index=False)
    print("\nQuintile monotonicity (Q1->Q5 ordering of forward RV change, "
          "+1 = perfectly increasing):")
    print(mono.sort_values(ascending=False).round(2).to_string())

    # ---------- redundancy ----------
    corr = df[feats].corr(method="spearman")
    corr.to_csv(os.path.join(RES_DIR, "feature_corr.csv"))

    # ---------- figures ----------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (data, title) in zip(
            axes, [(ic_df, "Raw IC"),
                   (orth_df, "Orthogonalized IC (marginal vs HAR+VIX)")]):
        for f in feats:
            g = data[data.feature == f]
            ax.plot(g["horizon"], g["IC"], lw=0.9, label=f)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("horizon (days)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("rank IC vs forward Δ log RV")
    axes[1].legend(fontsize=5.5, ncol=2)
    fig.suptitle("Factor-style IC decay of physics features")
    fig.tight_layout()
    fig.savefig(os.path.join(RES_DIR, "fig7_ic_decay.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats)), feats, rotation=90)
    ax.set_yticks(range(len(feats)), feats)
    fig.colorbar(im, label="Spearman corr")
    ax.set_title("Feature redundancy")
    fig.tight_layout()
    fig.savefig(os.path.join(RES_DIR, "fig8_feature_corr.png"))
    plt.close(fig)
    print("\nSaved factor_ic*.csv, factor_quantiles.csv, feature_corr.csv, "
          "fig7, fig8")


if __name__ == "__main__":
    main()
