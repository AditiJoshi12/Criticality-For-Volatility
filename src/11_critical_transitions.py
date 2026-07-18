"""
11_critical_transitions.py  -- THE HEADLINE EXPERIMENTS

Hypothesis: early-warning signals inspired by critical phenomena detect
volatility regime transitions EARLIER than volatility-based measures.
The right test is therefore not average forecast loss but timing:
who moves first, how reliably, and at what false-alarm cost.

Experiment 1 - Multi-signal event study ("who moves first?")
  All signals are put on a common causal z-scale (trailing 252d mean/std),
  aligned on model-free high-volatility episode onsets, and averaged. A
  placebo band is built by drawing many random sets of pseudo-onsets. For
  each signal we report the FIRST day, relative to onset, where the onset
  average exceeds the placebo 95% band and stays above it -- an estimate
  of how early the signal separates from noise.

Experiment 2 - Lead-time comparison table
  One transparent alarm rule for every signal (signal exceeds its own
  past-only trailing 90th percentile). For each signal: detection rate
  (alarm within the 40 pre-onset days), mean/median lead time, false-alarm
  rate, and alarms per year. Signals compared: trailing realized vol, VIX,
  criticality, Hawkes branching, and a criticality+VIX combination
  (mean of causal z-scores). The physics case requires beating the
  vol-based rows on lead time WITHOUT a worse false-alarm rate.

Experiment 3 - Transition-conditional feature importance
  Permutation importance of all features for the 5-day-ahead target,
  computed separately on transition-window test days vs calm test days.
  The hypothesis predicts physics features matter near transitions and
  are irrelevant in calm markets; classical vol terms should dominate
  the calm sample.

Point-in-time discipline: signal z-scores and alarm thresholds are
trailing-only; onsets (which use future data, as labels must) enter only
as evaluation events, never as inputs.
"""
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance

from common import (CORE_FEATURES, MECHANISM_MAP, PROC_DIR, RES_DIR,
                    high_vol_event_labels)

warnings.filterwarnings("ignore")

BACK, FWD = 40, 10          # event window (days before / after onset)
N_PLACEBO = 1000            # placebo draws for the event-study band
ALARM_WINDOW = 504          # trailing window for alarm thresholds
ALARM_Q = 0.90
LEAD_LOOKBACK = 40
Z_WINDOW = 252
plt.rcParams.update({"figure.dpi": 130, "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})


def _load():
    phys = pd.read_parquet(os.path.join(PROC_DIR, "physics_features.parquet"))
    rv = pd.read_parquet(os.path.join(PROC_DIR, "daily_rv.parquet"))["rv"]
    idx = phys.index.intersection(rv.index)
    phys, rv = phys.reindex(idx), rv.reindex(idx)
    vix = None
    vp = os.path.join(PROC_DIR, "vix.parquet")
    if os.path.exists(vp):
        vix = np.log(pd.read_parquet(vp)["VIX"]).reindex(idx)
    return phys, rv, vix


def causal_z(s, window=Z_WINDOW):
    mu = s.rolling(window, min_periods=window // 2).mean()
    sd = s.rolling(window, min_periods=window // 2).std()
    return (s - mu) / (sd + 1e-12)


# ---------------------------------------------------------------------------
# Experiment 1
# ---------------------------------------------------------------------------
def first_separation_day(onset_mean, placebo_band_hi, min_run=3):
    """First day t in [-BACK, 0] where the onset mean exceeds the placebo
    95% band and stays above for `min_run` consecutive days. Returns the
    day relative to onset (negative = leads) or NaN."""
    t_axis = np.arange(-BACK, FWD + 1)
    above = onset_mean > placebo_band_hi
    for i in range(len(t_axis) - min_run + 1):
        if t_axis[i] > 0:
            break
        if above[i:i + min_run].all():
            return int(t_axis[i])
    return np.nan


def event_study(signals, onsets, rng_seed=0):
    """Returns per-signal aligned onset means, placebo bands, and the
    first-separation day. Placebo band: for each draw, sample n_onsets
    random days and average -- the distribution of the MEAN trajectory
    under the null of no event alignment."""
    rng = np.random.RandomState(rng_seed)
    idx = signals.index
    onset_pos = idx.get_indexer(onsets[onsets].index)
    onset_pos = onset_pos[(onset_pos > BACK + Z_WINDOW)
                          & (onset_pos < len(idx) - FWD)]
    out = {}
    span = np.arange(-BACK, FWD + 1)
    for col in signals.columns:
        v = signals[col].values
        stack = np.array([v[p - BACK:p + FWD + 1] for p in onset_pos])
        onset_mean = np.nanmean(stack, axis=0)
        pl_means = np.empty((N_PLACEBO, len(span)))
        lo, hi = BACK + Z_WINDOW, len(v) - FWD - 1
        for b in range(N_PLACEBO):
            pos = rng.randint(lo, hi, len(onset_pos))
            pl = np.array([v[p - BACK:p + FWD + 1] for p in pos])
            pl_means[b] = np.nanmean(pl, axis=0)
        band_lo, band_hi = np.nanpercentile(pl_means, [2.5, 97.5], axis=0)
        out[col] = {"mean": onset_mean, "band_lo": band_lo,
                    "band_hi": band_hi,
                    "first_day": first_separation_day(onset_mean, band_hi),
                    "n_onsets": len(onset_pos)}
    return out, span


def fig9(results, span):
    cols = list(results.keys())
    n = len(cols)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.1 * ((n + 1) // 2), 5.6),
                             sharex=True)
    axes = axes.ravel()
    for ax, col in zip(axes, cols):
        r = results[col]
        ax.fill_between(span, r["band_lo"], r["band_hi"], color="gray",
                        alpha=0.25, label="placebo 95%")
        ax.plot(span, r["mean"], color="tab:red", lw=1.2, label="onsets")
        ax.axvline(0, color="k", lw=0.6)
        if np.isfinite(r["first_day"]):
            ax.axvline(r["first_day"], color="tab:blue", ls="--", lw=1.0)
            ax.set_title(f"{col}  (separates at t = {r['first_day']:+d}d)",
                         fontsize=8)
        else:
            ax.set_title(f"{col}  (no separation)", fontsize=8)
        ax.set_xlabel("days to onset")
    for ax in axes[len(cols):]:
        ax.axis("off")
    axes[0].set_ylabel("causal z-score")
    axes[0].legend(fontsize=6)
    fig.suptitle("Which signals move first? Event study around "
                 f"{results[cols[0]]['n_onsets']} high-vol episode onsets",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(RES_DIR, "fig9_who_moves_first.png"),
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 2
# ---------------------------------------------------------------------------
def _alarm_episodes(alarm, merge_gap=5):
    """Group alarm days into episodes: consecutive (or near-consecutive,
    gap <= merge_gap) alarm days count as ONE warning. Returns episode
    start positions. Episode-level evaluation avoids the saturation problem
    of day-level alarms (a q90 day-alarm fires somewhere in almost any
    40-day window by chance alone)."""
    pos = np.where(alarm.values)[0]
    if len(pos) == 0:
        return np.array([], dtype=int)
    starts = [pos[0]]
    for a, b in zip(pos[:-1], pos[1:]):
        if b - a > merge_gap:
            starts.append(b)
    return np.array(starts)


def alarm_metrics(signal, onsets, idx):
    """Episode-level early-warning metrics.
    - lead time = onset minus the EARLIEST episode start in the lookback
      window (how early did the warning begin, not how recently it fired)
    - false alarm = episode with no onset within LEAD_LOOKBACK days after
      its start
    - detection is reported but NOTE: with common alarm frequencies it is
      close to saturated over a 40d lookback; compare against the `chance`
      row (circularly shifted signal) before crediting any signal."""
    thr = signal.rolling(ALARM_WINDOW,
                         min_periods=ALARM_WINDOW // 2).quantile(ALARM_Q).shift(1)
    alarm = ((signal > thr) & thr.notna()).reindex(idx).fillna(False)
    ep_starts = _alarm_episodes(alarm)
    onset_pos = idx.get_indexer(onsets[onsets].index)
    onset_pos = onset_pos[onset_pos >= 0]

    leads = []
    for o in onset_pos:
        prior = ep_starts[(ep_starts < o) & (ep_starts >= o - LEAD_LOOKBACK)]
        leads.append(o - prior.min() if len(prior) else np.nan)
    leads = np.array(leads, dtype=float)

    fa = sum(1 for s in ep_starts
             if not ((onset_pos > s) & (onset_pos <= s + LEAD_LOOKBACK)).any())
    n_years = len(idx) / 252
    return {"detection_rate": np.isfinite(leads).mean() if len(leads) else np.nan,
            "mean_lead_d": np.nanmean(leads),
            "median_lead_d": np.nanmedian(leads),
            "false_alarm_rate": fa / max(len(ep_starts), 1),
            "episodes_per_year": len(ep_starts) / n_years,
            "n_onsets": int(len(onset_pos))}


def chance_baseline(signal, onsets, idx, shifts=(500, 1000, 1500, 2000)):
    """Metrics for circularly shifted copies of the signal: what detection
    rate / lead / false-alarm rate a signal with the SAME marginal
    distribution and autocorrelation achieves with no event alignment."""
    rows = []
    v = signal.reindex(idx)
    for s in shifts:
        shifted = pd.Series(np.roll(v.values, s), index=idx)
        rows.append(alarm_metrics(shifted, onsets, idx))
    return pd.DataFrame(rows).median().to_dict()


# ---------------------------------------------------------------------------
# Experiment 3
# ---------------------------------------------------------------------------
def transition_conditional_importance(onsets):
    path = os.path.join(PROC_DIR, "model_matrix.parquet")
    if not os.path.exists(path):
        print("Experiment 3 skipped: model_matrix.parquet missing "
              "(run steps 04-06 first).")
        return None
    df = pd.read_parquet(path)
    target = "target_h5"
    cols = [c for c in df.columns if not c.startswith("target_")]
    split = int(len(df) * 0.7)
    m = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                  learning_rate=0.05, random_state=0)
    m.fit(df[cols].iloc[:split], df[target].iloc[:split])

    on = onsets.reindex(df.index).fillna(False)
    trans = on.rolling(21, center=True, min_periods=1).max().astype(bool)
    test = df.iloc[split:]
    t_mask = trans.iloc[split:].values
    out = {}
    for name, sel in [("transition", t_mask), ("calm", ~t_mask)]:
        sub = test[sel]
        if len(sub) < 100:
            continue
        imp = permutation_importance(m, sub[cols], sub[target],
                                     n_repeats=10, random_state=0)
        out[name] = pd.Series(imp.importances_mean, index=cols)
    imp_df = pd.DataFrame(out)
    imp_df.to_csv(os.path.join(RES_DIR, "importance_by_state.csv"))

    plot = imp_df.sort_values("transition")
    fig, ax = plt.subplots(figsize=(6, 0.32 * len(plot) + 1.5))
    y = np.arange(len(plot))
    ax.barh(y + 0.2, plot["transition"], height=0.38, color="tab:red",
            label="transition windows")
    ax.barh(y - 0.2, plot["calm"], height=0.38, color="tab:gray",
            label="calm periods")
    ax.set_yticks(y, plot.index)
    ax.set_xlabel("permutation importance (5d target, test set)")
    ax.legend(fontsize=7)
    ax.set_title("Where do physics features matter?")
    fig.tight_layout()
    fig.savefig(os.path.join(RES_DIR, "fig11_importance_by_state.png"))
    plt.close(fig)
    return imp_df


# ---------------------------------------------------------------------------
def main():
    os.makedirs(RES_DIR, exist_ok=True)
    phys, rv, vix = _load()
    _, onsets = high_vol_event_labels(rv)
    onsets = onsets.reindex(phys.index).fillna(False)

    # signals on a common causal z-scale
    sig = pd.DataFrame(index=phys.index)
    sig["realized_vol"] = causal_z(np.log(rv).rolling(5).mean())
    if vix is not None:
        sig["VIX"] = causal_z(vix)
    sig["criticality"] = phys["criticality"]          # already a z-score
    sig["hawkes_branching"] = causal_z(phys["hawkes_branching"])
    sig["corr_length"] = causal_z(phys["corr_length"])
    sig["entropy_eigen"] = causal_z(-phys["entropy_eigen"])  # order rises
    if vix is not None:
        sig["crit+VIX"] = (sig["criticality"] + sig["VIX"]) / 2

    # ---- Experiment 1 ----
    print(f"Experiment 1: event study around "
          f"{int(onsets.sum())} onsets ({N_PLACEBO} placebo draws)...")
    ev_cols = [c for c in ["criticality", "hawkes_branching", "corr_length",
                           "entropy_eigen", "realized_vol", "VIX"]
               if c in sig.columns]
    results, span = event_study(sig[ev_cols], onsets)
    fig9(results, span)
    first = pd.Series({c: results[c]["first_day"] for c in ev_cols},
                      name="first_separation_day").sort_values()
    first.to_csv(os.path.join(RES_DIR, "first_mover.csv"))
    print("First day of separation from placebo band (negative = leads):")
    print(first.to_string())

    # ---- Experiment 2 ----
    print("\nExperiment 2: lead-time comparison (alarm = past-only "
          f"q{int(ALARM_Q * 100)} crossing)...")
    rows = {}
    for c in ["realized_vol", "VIX", "criticality", "hawkes_branching",
              "crit+VIX"]:
        if c in sig.columns:
            rows[c] = alarm_metrics(sig[c].dropna(), onsets, phys.index)
    rows["(chance)"] = chance_baseline(sig["criticality"].dropna(), onsets,
                                       phys.index)
    lt = pd.DataFrame(rows).T
    lt.to_csv(os.path.join(RES_DIR, "leadtime_comparison.csv"))
    print(lt.round(3).to_string())
    print("How to read this table:")
    print(f"  - detection is partly saturated over a {LEAD_LOOKBACK}d "
          "lookback; credit only detection\n    clearly above the (chance) "
          "row")
    print(f"  - a median lead near {int(LEAD_LOOKBACK * 0.75)}d+ is a "
          "window-edge artifact (unaligned episodes),\n    not early "
          "warning; genuine leads sit well inside the window")
    print("  - the physics case = longer genuine lead than the vol rows "
          "at comparable\n    false-alarm rate and episodes/year")

    # ---- Experiment 3 ----
    print("\nExperiment 3: transition-conditional importance...")
    imp = transition_conditional_importance(onsets)
    if imp is not None:
        core = [c for c in CORE_FEATURES if c in imp.index]
        ratio = (imp.loc[core, "transition"].sum()
                 / max(imp["transition"].clip(lower=0).sum(), 1e-12))
        print(f"Physics share of total transition-window importance: "
              f"{ratio:.0%}")
        print(imp.round(4).sort_values("transition",
                                       ascending=False).head(8).to_string())

    print("\nMechanism map used for the narrative:")
    for k, v in MECHANISM_MAP.items():
        print(f"  {k:22s} {v}")


if __name__ == "__main__":
    main()
