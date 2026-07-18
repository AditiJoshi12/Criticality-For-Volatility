"""
05_baseline_models.py  (Phases 2 & 5)

Walk-forward, out-of-sample forecasts of realized variance at horizons
h = 1, 5, 22 days (log of mean daily RV over the next h days).

Models
  h=1 only:  RandomWalk, EWMA, GARCH(1,1), EGARCH(1,1)
  all h:     HAR                      (OLS on daily/weekly/monthly RV)
             HAR+Phys                 (elastic net, standardized in-window)
             HAR+VIX                  (OLS)
             HAR+VIX+Phys             (elastic net)
  The elastic net handles the correlated, noisy physics block; comparing
  with/without VIX shows whether VIX crowds the physics features out.

Evaluation
  - identical OOS days for all models at a given horizon
  - RMSE / MAE on log RV, QLIKE on variance
  - Diebold-Mariano tests (HAR+Phys vs HAR, and +VIX variants)
  - CONDITIONAL evaluation: the same losses and DM tests restricted to
    windows around high-volatility episode onsets (model-free events from
    common.high_vol_event_labels, so no HMM leakage) vs. calm days.
    Physics features are hypothesized to matter near transitions; an
    unconditional average would hide that.

Look-ahead discipline: expanding training windows, scaler and elastic-net
hyperparameters fit inside each training window only; event labels are used
only to SLICE the evaluation sample ex post, never as model inputs.
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from common import PROC_DIR, RES_DIR, TARGET, high_vol_event_labels

warnings.filterwarnings("ignore")

HAR_COLS = ["rv_d", "rv_w", "rv_m"]
PHYS_COLS = ["corr_length", "eig_max_frac", "entropy_eigen",
             "frac_eig_above_mp", "entropy_returns", "temperature",
             "susceptibility", "acf1", "var_of_vol", "hurst_vol",
             "hawkes_branching", "criticality"]
HORIZONS = [1, 5, 22]

MIN_TRAIN = 750
REFIT_EVERY = 21
GARCH_REFIT = 63
EVENT_WINDOW = 10   # +/- days around an episode onset counted as "transition"


# ---------------------------------------------------------------------------
# losses & tests
# ---------------------------------------------------------------------------
def qlike(rv_true, rv_pred):
    rv_pred = np.clip(rv_pred, 1e-12, None)
    return rv_true / rv_pred - np.log(rv_true / rv_pred) - 1


def diebold_mariano(loss_a, loss_b, h=1):
    """DM test with Newey-West variance. Negative stat: A beats B.
    For h-step overlapping forecasts, the lag length grows with h."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 30:
        return np.nan, np.nan
    dbar = d.mean()
    lag = max(h - 1, int(np.floor(1.5 * n ** (1 / 3))))
    s = np.var(d, ddof=0)
    for k in range(1, lag + 1):
        cov = np.cov(d[k:], d[:-k], ddof=0)[0, 1]
        s += 2 * (1 - k / (lag + 1)) * cov
    dm = dbar / np.sqrt(s / n)
    return dm, 2 * (1 - stats.norm.cdf(abs(dm)))


# ---------------------------------------------------------------------------
# walk-forward learners
# ---------------------------------------------------------------------------
def walk_forward_ols(df, cols, target):
    preds = pd.Series(index=df.index, dtype=float)
    model = None
    for i in range(MIN_TRAIN, len(df)):
        if model is None or (i - MIN_TRAIN) % REFIT_EVERY == 0:
            tr = df.iloc[:i]
            model = LinearRegression().fit(tr[cols], tr[target])
        preds.iloc[i] = model.predict(df[cols].iloc[[i]])[0]
    return preds.dropna()


def walk_forward_enet(df, cols, target):
    """Elastic net with in-window standardization and time-series CV.
    Everything (scaler, alpha, coefficients) is re-estimated on the
    expanding training window only."""
    preds = pd.Series(index=df.index, dtype=float)
    model, scaler = None, None
    alphas = np.logspace(-4, -0.5, 8)
    for i in range(MIN_TRAIN, len(df)):
        if model is None or (i - MIN_TRAIN) % REFIT_EVERY == 0:
            tr = df.iloc[:i]
            scaler = StandardScaler().fit(tr[cols])
            Xtr = scaler.transform(tr[cols])
            model = ElasticNetCV(l1_ratio=[0.5, 1.0], alphas=alphas,
                                 cv=TimeSeriesSplit(3), max_iter=5000
                                 ).fit(Xtr, tr[target])
        preds.iloc[i] = model.predict(
            scaler.transform(df[cols].iloc[[i]]))[0]
    return preds.dropna()


def walk_forward_garch(rets, index, egarch=False):
    from arch import arch_model
    r = rets.reindex(index).dropna() * 100
    preds = pd.Series(index=index, dtype=float)
    res, params = None, None
    for i in range(MIN_TRAIN, len(index)):
        t = index[i]
        if t not in r.index:
            continue
        pos = r.index.get_loc(t)
        if res is None or (i - MIN_TRAIN) % GARCH_REFIT == 0:
            am = arch_model(r.iloc[:pos + 1], vol="EGARCH" if egarch else "GARCH",
                            p=1, o=1 if egarch else 0, q=1, dist="normal")
            res = am.fit(disp="off", show_warning=False)
            params = res.params
        am_t = arch_model(r.iloc[:pos + 1], vol="EGARCH" if egarch else "GARCH",
                          p=1, o=1 if egarch else 0, q=1, dist="normal")
        fc = am_t.fix(params).forecast(horizon=1, reindex=False)
        v = fc.variance.values[-1, 0] / 100 ** 2
        # EGARCH's exponential recursion can blow up when stale fixed
        # parameters meet an extreme return; cap forecasts at a very
        # generous but finite level (200% annualized vol) and floor them.
        v_cap = (2.0 ** 2) / 252
        preds.loc[t] = min(max(v, 1e-10), v_cap)
    return np.log(preds.dropna())


def ewma_forecast(rv, lam=0.94):
    return np.log(rv.ewm(alpha=1 - lam).mean().clip(lower=1e-12))


# ---------------------------------------------------------------------------
def evaluate(forecasts, df, target, h, transition_mask):
    eval_idx = df.index[MIN_TRAIN:]
    fc = {k: v.reindex(eval_idx) for k, v in forecasts.items()}
    common_idx = pd.DataFrame(fc).dropna().index
    y = df.loc[common_idx, target]
    rv_true = np.exp(y)
    tmask = transition_mask.reindex(common_idx).fillna(False)

    rows, losses = [], {}
    for name, p in fc.items():
        p = p.loc[common_idx]
        ql = qlike(rv_true, np.exp(p))
        losses[name] = ql
        rows.append({
            "horizon": h, "model": name, "n_days": len(common_idx),
            "RMSE": np.sqrt(((y - p) ** 2).mean()),
            "MAE": (y - p).abs().mean(),
            "QLIKE": ql.mean(),
            "QLIKE_transition": ql[tmask].mean(),
            "QLIKE_calm": ql[~tmask].mean(),
            "n_transition": int(tmask.sum()),
        })
    return pd.DataFrame(rows), losses, tmask


def dm_report(losses, tmask, h):
    pairs = [("HAR+Phys", "HAR"), ("HAR+VIX+Phys", "HAR+VIX")]
    rows = []
    for a, b in pairs:
        if a not in losses or b not in losses:
            continue
        for scope, sel in [("all", slice(None)),
                           ("transition", tmask.values),
                           ("calm", ~tmask.values)]:
            la = losses[a].values[sel] if scope != "all" else losses[a].values
            lb = losses[b].values[sel] if scope != "all" else losses[b].values
            dm, p = diebold_mariano(la, lb, h=h)
            rows.append({"horizon": h, "comparison": f"{a} vs {b}",
                         "scope": scope, "dm_stat": dm, "p_value": p,
                         "verdict": ("physics helps" if dm < 0 and p < 0.05
                                     else "physics hurts" if dm > 0 and p < 0.05
                                     else "no significant difference")})
    return pd.DataFrame(rows)


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    df = pd.read_parquet(os.path.join(PROC_DIR, "model_matrix.parquet"))
    if len(df) < MIN_TRAIN + 250:
        raise SystemExit(
            f"Model matrix has only {len(df)} rows; need at least "
            f"MIN_TRAIN + 250 for walk-forward evaluation. Re-run the "
            f"data download and steps 02-06, then 04.")
    rv = pd.read_parquet(os.path.join(PROC_DIR, "daily_rv.parquet"))["rv"]
    rets = pd.read_parquet(os.path.join(PROC_DIR, "log_returns.parquet"))[TARGET]

    phys_cols = [c for c in PHYS_COLS if c in df.columns]
    has_vix = "log_vix" in df.columns

    # model-free transition windows for conditional evaluation (ex post
    # slicing of the test sample; never a model input)
    _, starts = high_vol_event_labels(rv)
    starts = starts.reindex(df.index).fillna(False)
    transition = starts.rolling(2 * EVENT_WINDOW + 1, center=True,
                                min_periods=1).max().astype(bool)

    all_metrics, all_dm = [], []
    for h in HORIZONS:
        target = f"target_h{h}"
        print(f"\n=== Horizon {h}d ===")
        forecasts = {}
        if h == 1:
            forecasts["RandomWalk"] = df["rv_d"]
            forecasts["EWMA"] = ewma_forecast(rv).reindex(df.index)
            print("  GARCH / EGARCH (walk-forward)...")
            forecasts["GARCH(1,1)"] = walk_forward_garch(rets, df.index)
            forecasts["EGARCH(1,1)"] = walk_forward_garch(rets, df.index,
                                                          egarch=True)
        print("  HAR (OLS)...")
        forecasts["HAR"] = walk_forward_ols(df, HAR_COLS, target)
        print("  HAR+Phys (elastic net)...")
        forecasts["HAR+Phys"] = walk_forward_enet(df, HAR_COLS + phys_cols,
                                                  target)
        if has_vix:
            print("  HAR+VIX (OLS)...")
            forecasts["HAR+VIX"] = walk_forward_ols(
                df, HAR_COLS + ["log_vix"], target)
            print("  HAR+VIX+Phys (elastic net)...")
            forecasts["HAR+VIX+Phys"] = walk_forward_enet(
                df, HAR_COLS + ["log_vix"] + phys_cols, target)

        metrics, losses, tmask = evaluate(forecasts, df, target, h, transition)
        all_metrics.append(metrics)
        all_dm.append(dm_report(losses, tmask, h))
        print(metrics.set_index("model")[
            ["QLIKE", "QLIKE_transition", "QLIKE_calm"]].round(4))

        if h == 1:
            pd.DataFrame({k: v.reindex(df.index) for k, v in
                          forecasts.items()}).dropna().to_parquet(
                os.path.join(RES_DIR, "forecasts.parquet"))

    metrics = pd.concat(all_metrics, ignore_index=True)
    dm_table = pd.concat(all_dm, ignore_index=True)
    metrics.to_csv(os.path.join(RES_DIR, "forecast_metrics.csv"), index=False)
    dm_table.to_csv(os.path.join(RES_DIR, "dm_tests.csv"), index=False)
    print("\n=== Diebold-Mariano summary (QLIKE) ===")
    print(dm_table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
