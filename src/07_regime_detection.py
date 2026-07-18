"""
07_regime_detection.py  (Phase 3 + early warning)

Two layers, both free of the look-ahead problems of a full-sample HMM:

1. Regime labelling
   - Full-sample 3-state Gaussian HMM (10 EM restarts) -> descriptive
     labels/plots ONLY. These are in-sample by construction and are never
     used to score predictive performance.
   - Walk-forward HMM: refit on an expanding window every REFIT days,
     P(High tomorrow) taken from the filtered state probabilities of the
     LAST observation and the transition matrix. Fully causal.

2. Early-warning evaluation (the scientific test)
   - Events are MODEL-FREE: mean daily RV over the next 5 days exceeds a
     past-only trailing quantile (common.high_vol_event_labels). No HMM in
     the labels -> no leakage from a full-sample fit.
   - Classifiers (logistic regression) trained walk-forward on expanding
     windows, standardized in-window, evaluated only on days where the
     market is currently calm (label 0 yesterday-known information).
   - Feature sets: vol-only benchmark, criticality, criticality+vol,
     criticality+hawkes+vol.
   - Metrics: ROC AUC and average precision with moving-block bootstrap
     confidence intervals (block length 21 days, respecting serial
     dependence).
   - Lead-time analysis: for each episode onset, how many days earlier did
     the criticality score cross its past-only 90th percentile? Plus the
     false-alarm rate of that crossing rule.
"""
import os
import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from common import PROC_DIR, RES_DIR, high_vol_event_labels

warnings.filterwarnings("ignore")

K = 3
HORIZON = 5
MIN_TRAIN = 750
HMM_REFIT = 63
CLF_REFIT = 63
BOOT_B = 500
BLOCK = 21
ALARM_Q = 0.90
LEAD_LOOKBACK = 40   # days before onset within which an alarm counts


# ---------------------------------------------------------------------------
def fit_hmm(x, n_restarts=10):
    best, best_ll = None, -np.inf
    for seed in range(n_restarts):
        hmm = GaussianHMM(n_components=K, covariance_type="diag",
                          n_iter=500, random_state=seed, min_covar=1e-4)
        try:
            hmm.fit(x)
            ll = hmm.score(x)
        except Exception:
            continue
        if ll > best_ll:
            best, best_ll = hmm, ll
    hmm = best
    order = np.argsort(hmm.means_.ravel())
    post = hmm.predict_proba(x)[:, order]
    labels = post.argmax(axis=1)
    A = hmm.transmat_[np.ix_(order, order)]
    return labels, post, A


def walk_forward_hmm_phigh(x, refit=HMM_REFIT, restarts=3):
    """Causal P(High tomorrow): at day t, fit on data up to t (refit every
    `refit` days), take the filtered probability of the last observation,
    and propagate one step with the transition matrix."""
    X = x.values.reshape(-1, 1)
    out = pd.Series(np.nan, index=x.index, name="P_High_wf")
    hmm, order = None, None
    for i in range(MIN_TRAIN, len(x)):
        if hmm is None or (i - MIN_TRAIN) % refit == 0:
            best, best_ll = None, -np.inf
            for seed in range(restarts):
                m = GaussianHMM(n_components=K, covariance_type="diag",
                                n_iter=300, random_state=seed,
                                min_covar=1e-4)
                try:
                    m.fit(X[:i])
                    ll = m.score(X[:i])
                except Exception:
                    continue
                if ll > best_ll:
                    best, best_ll = m, ll
            hmm = best
            order = np.argsort(hmm.means_.ravel())
        post_last = hmm.predict_proba(X[:i + 1])[-1, order]
        A = hmm.transmat_[np.ix_(order, order)]
        out.iloc[i] = (post_last @ A)[K - 1]
    return out


# ---------------------------------------------------------------------------
def walk_forward_logit(F, y, valid):
    """Expanding-window logistic regression. Returns OOS scores on `valid`
    days after MIN_TRAIN. Scaler and coefficients fit on training rows only."""
    scores = pd.Series(np.nan, index=F.index)
    model, scaler = None, None
    rows = np.arange(len(F))
    for i in rows[rows >= MIN_TRAIN]:
        if not valid.iloc[i]:
            continue
        if model is None or (i - MIN_TRAIN) % CLF_REFIT == 0:
            tr = valid.iloc[:i].values
            Xtr, ytr = F.iloc[:i][tr], y.iloc[:i][tr]
            if ytr.nunique() < 2:
                continue
            scaler = StandardScaler().fit(Xtr)
            model = LogisticRegression(max_iter=1000).fit(
                scaler.transform(Xtr), ytr)
        scores.iloc[i] = model.predict_proba(
            scaler.transform(F.iloc[[i]]))[0, 1]
    return scores


def block_bootstrap_ci(y, s, metric, B=BOOT_B, block=BLOCK, seed=0):
    """Moving-block bootstrap CI for a rank metric on serially dependent
    data. Resamples contiguous blocks of (y, score) pairs."""
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    n = len(y)
    n_blocks = int(np.ceil(n / block))
    vals = []
    for _ in range(B):
        starts = rng.integers(0, n - block, n_blocks)
        idx = np.concatenate([np.arange(st, st + block) for st in starts])[:n]
        yy, ss = y[idx], s[idx]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(metric(yy, ss))
    if not vals:
        return np.nan, np.nan
    return np.percentile(vals, [2.5, 97.5])


# ---------------------------------------------------------------------------
def lead_time_analysis(crit, onsets):
    """Alarm rule: criticality crosses its past-only trailing 90th
    percentile. For each onset, lead time = onset - most recent alarm within
    LEAD_LOOKBACK days (NaN if none). False-alarm rate = fraction of alarm
    days not followed by an onset within LEAD_LOOKBACK days."""
    thr = crit.rolling(504, min_periods=252).quantile(ALARM_Q).shift(1)
    alarm = (crit > thr) & thr.notna()
    onset_days = list(onsets[onsets].index)
    alarm_pos = crit.index.get_indexer(alarm[alarm].index)

    rows = []
    for d in onset_days:
        i = crit.index.get_loc(d)
        prior = alarm_pos[(alarm_pos < i) & (alarm_pos >= i - LEAD_LOOKBACK)]
        rows.append({"onset": d.date(),
                     "lead_days": int(i - prior.max()) if len(prior) else np.nan,
                     "detected": bool(len(prior))})
    lead = pd.DataFrame(rows)

    onset_pos = set(crit.index.get_indexer(onset_days))
    fa = 0
    for a in alarm_pos:
        if not any(a < o <= a + LEAD_LOOKBACK for o in onset_pos):
            fa += 1
    fa_rate = fa / max(len(alarm_pos), 1)
    return lead, fa_rate, alarm


# ---------------------------------------------------------------------------
def main():
    os.makedirs(RES_DIR, exist_ok=True)
    rv = pd.read_parquet(os.path.join(PROC_DIR, "daily_rv.parquet"))["rv"]
    log_rv = np.log(rv).dropna()
    x = log_rv.rolling(5).mean().dropna()

    # descriptive full-sample HMM (plots only)
    labels, post, A = fit_hmm(x.values.reshape(-1, 1))
    names = ["Low", "Medium", "High"]
    out = pd.DataFrame({"log_rv_smooth": x.values,
                        "regime_hmm": [names[i] for i in labels]},
                       index=x.index)
    for i, n in enumerate(names):
        out[f"P_{n}"] = post[:, i]
    state_means = np.array([x.values[labels == i].mean() for i in range(K)])
    print("Full-sample HMM (DESCRIPTIVE ONLY - in-sample labels)")
    print("  state means (annualized vol):",
          np.round(np.sqrt(np.exp(state_means) * 252), 3))
    pd.DataFrame(A, index=names, columns=names).to_csv(
        os.path.join(RES_DIR, "transition_matrix.csv"))

    print("Walk-forward HMM P(High) (causal, refit every "
          f"{HMM_REFIT}d) - this can take a few minutes...")
    out["P_High_wf"] = walk_forward_hmm_phigh(x)
    out.to_parquet(os.path.join(RES_DIR, "regimes.parquet"))

    # ---------------- early-warning evaluation ----------------
    phys = pd.read_parquet(
        os.path.join(PROC_DIR, "physics_features.parquet")).reindex(x.index)
    labels_ev, starts = high_vol_event_labels(rv, horizon=HORIZON)
    labels_ev = labels_ev.reindex(x.index)
    starts = starts.reindex(x.index).fillna(False)

    # evaluate only on days that are currently calm by past-known info:
    # yesterday's trailing threshold vs yesterday's RV
    thr_past = rv.rolling(252).quantile(0.85).shift(1).reindex(x.index)
    currently_calm = (rv.reindex(x.index).shift(1) <= thr_past)
    y = labels_ev
    valid = currently_calm & y.notna() & phys["criticality"].notna()

    vol_feat = x.rolling(21).mean().rename("vol_trail")
    feature_sets = {
        "vol_only": pd.DataFrame({"v": vol_feat}),
        "criticality": pd.DataFrame({"c": phys["criticality"]}),
        "crit+vol": pd.DataFrame({"c": phys["criticality"], "v": vol_feat}),
        "crit+hawkes+vol": pd.DataFrame({"c": phys["criticality"],
                                         "h": phys["hawkes_branching"],
                                         "v": vol_feat}),
    }

    rows = []
    for name, F in feature_sets.items():
        v = valid & F.notna().all(axis=1)
        s = walk_forward_logit(F, y.fillna(0).astype(int), v)
        mask = s.notna() & v
        yy, ss = y[mask].astype(int), s[mask]
        if yy.nunique() < 2:
            continue
        auc = roc_auc_score(yy, ss)
        ap = average_precision_score(yy, ss)
        lo, hi = block_bootstrap_ci(yy.values, ss.values, roc_auc_score)
        rows.append({"features": name, "n_oos": len(yy),
                     "event_rate": yy.mean(), "AUC": auc,
                     "AUC_ci_lo": lo, "AUC_ci_hi": hi,
                     "avg_precision": ap})
        print(f"  {name:16s} AUC={auc:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"AP={ap:.3f}  n={len(yy)}")
    ew = pd.DataFrame(rows)
    ew.to_csv(os.path.join(RES_DIR, "early_warning_auc.csv"), index=False)

    # ---------------- lead-time analysis ----------------
    lead, fa_rate, alarm = lead_time_analysis(phys["criticality"], starts)
    lead.to_csv(os.path.join(RES_DIR, "lead_times.csv"), index=False)
    starts.to_frame("onset").to_parquet(os.path.join(RES_DIR,
                                                     "event_onsets.parquet"))
    det = lead["detected"].mean() if len(lead) else np.nan
    med = lead["lead_days"].median()
    print(f"\nLead-time analysis (alarm = criticality > past-only "
          f"q{int(ALARM_Q * 100)}):")
    print(f"  onsets: {len(lead)}   detected within {LEAD_LOOKBACK}d "
          f"before onset: {det:.0%}   median lead: {med} days")
    print(f"  false-alarm rate of alarm days: {fa_rate:.0%}")
    print("  (a useful early-warning signal needs BOTH high detection and "
          "a false-alarm rate well below the vol-only alternative)")


if __name__ == "__main__":
    main()
