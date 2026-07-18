"""
09_visualizations.py  (Phase 6 + figures)

Produces:
  fig1_signals.png        rolling Hurst, temperature, criticality, RV (+VIX)
  fig2_regimes.png        regime timeline over price with P(High)
  fig3_transition.png     transition-matrix heatmap
  fig4_importance.png     permutation importance of physics features
  fig5_equity.png         vol-targeting equity curves
"""
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ANN, PROC_DIR, RES_DIR, TARGET
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

plt.rcParams.update({"figure.dpi": 120, "font.size": 8})


def fig1_signals():
    phys = pd.read_parquet(os.path.join(PROC_DIR, "physics_features.parquet"))
    rv = pd.read_parquet(os.path.join(PROC_DIR, "daily_rv.parquet"))["rv_ann"]
    panels = [("Realized vol (ann.)", rv.rolling(5).mean(), "tab:red"),
              ("Vol roughness H (DFA)", phys["hurst_vol"], "tab:blue"),
              ("Hawkes branching n", phys["hawkes_branching"], "tab:brown"),
              ("Market temperature (z)", phys["temperature"], "tab:orange"),
              ("Criticality score (z)", phys["criticality"], "tab:purple")]
    vix_path = os.path.join(PROC_DIR, "vix.parquet")
    if os.path.exists(vix_path):
        vix = pd.read_parquet(vix_path)["VIX"]
        panels.insert(1, ("VIX", vix, "tab:green"))
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 2 * len(panels)),
                             sharex=True)
    for ax, (name, s, c) in zip(axes, panels):
        ax.plot(s.index, s.values, color=c, lw=0.7)
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    fig.suptitle("Statistical-physics signals vs volatility")
    fig.tight_layout()
    fig.savefig(os.path.join(RES_DIR, "fig1_signals.png"))
    plt.close(fig)


def fig2_regimes():
    reg = pd.read_parquet(os.path.join(RES_DIR, "regimes.parquet"))
    close = pd.read_parquet(os.path.join(PROC_DIR, "adj_close.parquet"))["SPY"] \
        .reindex(reg.index)
    colors = {"Low": "#2ca02c", "Medium": "#ff7f0e", "High": "#d62728"}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(close.index, close.values, "k-", lw=0.6)
    for name, c in colors.items():
        m = reg["regime_hmm"] == name
        a1.fill_between(reg.index, close.min(), close.max(), where=m,
                        color=c, alpha=0.15, label=name)
    a1.set_yscale("log")
    a1.set_ylabel("Price (log)")
    a1.legend(loc="upper left", ncol=3)
    a2.plot(reg.index, reg["P_High"], color="#d62728", lw=0.5, alpha=0.5,
            label="full-sample (descriptive)")
    if "P_High_wf" in reg.columns:
        a2.plot(reg.index, reg["P_High_wf"], color="k", lw=0.6,
                label="walk-forward (causal)")
    a2.legend(fontsize=6, loc="upper left")
    a2.set_ylabel("P(High)")
    a2.grid(alpha=0.3)
    fig.suptitle("HMM volatility regimes")
    fig.tight_layout()
    fig.savefig(os.path.join(RES_DIR, "fig2_regimes.png"))
    plt.close(fig)


def fig3_transition():
    A = pd.read_csv(os.path.join(RES_DIR, "transition_matrix.csv"), index_col=0)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(A.values, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(3), A.columns)
    ax.set_yticks(range(3), A.index)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{A.values[i, j]:.2f}", ha="center",
                    color="w" if A.values[i, j] < 0.6 else "k")
    ax.set_title("Regime transition matrix")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(RES_DIR, "fig3_transition.png"))
    plt.close(fig)


def fig4_importance(target="target_h1"):
    df = pd.read_parquet(os.path.join(PROC_DIR, "model_matrix.parquet"))
    cols = [c for c in df.columns if not c.startswith("target_")
            and not df[c].isna().all()]
    if len(df) < 500 or target not in df.columns:
        print("fig4 skipped: model matrix too small or target missing "
              f"({len(df)} rows). Re-run data download and steps 02-06, 04.")
        return
    split = int(len(df) * 0.7)
    m = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                  learning_rate=0.05, random_state=0)
    m.fit(df[cols].iloc[:split], df[target].iloc[:split])
    imp = permutation_importance(m, df[cols].iloc[split:],
                                 df[target].iloc[split:],
                                 n_repeats=10, random_state=0)
    s = pd.Series(imp.importances_mean, index=cols).sort_values()
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(s) + 1.5))
    s.plot.barh(ax=ax, color="tab:blue")
    ax.set_title("Permutation importance (test set, log-RV forecast)")
    fig.tight_layout()
    fig.savefig(os.path.join(RES_DIR, "fig4_importance.png"))
    s.sort_values(ascending=False).to_csv(
        os.path.join(RES_DIR, "feature_importance.csv"))
    plt.close(fig)


def fig5_equity():
    path = os.path.join(RES_DIR, "equity_curves.parquet")
    if not os.path.exists(path):
        return
    curves = pd.read_parquet(path)
    fig, ax = plt.subplots(figsize=(9, 4))
    for c in curves.columns:
        ax.plot(curves.index, curves[c], lw=0.8, label=c)
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.set_title("Vol-targeting equity curves (net of costs)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RES_DIR, "fig5_equity.png"))
    plt.close(fig)


def fig6_event_study(back=40, fwd=20):
    """Average trajectory of criticality and Hawkes branching around
    model-free high-vol episode onsets, vs. shuffled placebo onsets."""
    phys = pd.read_parquet(os.path.join(PROC_DIR, "physics_features.parquet"))
    onsets = pd.read_parquet(os.path.join(RES_DIR, "event_onsets.parquet"))
    onset_days = onsets.index[onsets["onset"]]
    rng = np.random.RandomState(0)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharex=True)
    for ax, col, label in [(axes[0], "criticality", "Criticality (z)"),
                           (axes[1], "hawkes_branching", "Hawkes branching n")]:
        sig = phys[col]
        pos = phys.index.get_indexer(onset_days)
        pos = pos[(pos > back) & (pos < len(sig) - fwd)]
        if len(pos) == 0:
            continue
        stack = np.array([sig.values[p - back:p + fwd + 1] for p in pos])
        placebo_pos = rng.randint(back, len(sig) - fwd, 500)
        pstack = np.array([sig.values[p - back:p + fwd + 1]
                           for p in placebo_pos])
        t = np.arange(-back, fwd + 1)
        m = np.nanmean(stack, axis=0)
        se = np.nanstd(stack, axis=0) / np.sqrt(len(pos))
        ax.plot(t, m, color="tab:red", label=f"onsets (n={len(pos)})")
        ax.fill_between(t, m - 2 * se, m + 2 * se, color="tab:red", alpha=0.2)
        ax.plot(t, np.nanmean(pstack, axis=0), color="gray", ls="--",
                label="placebo days")
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("days relative to episode onset")
        ax.set_title(label)
        ax.legend(fontsize=6)
    fig.suptitle("Event study around high-volatility episode onsets")
    fig.tight_layout()
    fig.savefig(os.path.join(RES_DIR, "fig6_event_study.png"))
    plt.close(fig)


if __name__ == "__main__":
    for fn in (fig1_signals, fig2_regimes, fig3_transition,
               fig4_importance, fig5_equity, fig6_event_study):
        try:
            fn()
        except FileNotFoundError as e:
            print(f"{fn.__name__} skipped (missing input: {e.filename}) — "
                  f"an upstream step likely did not produce its output.")
        except Exception as e:  # noqa: BLE001 — figures must not kill the run
            print(f"{fn.__name__} FAILED: {type(e).__name__}: {e}")
    print("Figures written to results/")
