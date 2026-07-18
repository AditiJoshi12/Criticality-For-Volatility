# Statistical Physics and Rough Volatility for Volatility Regime Detection

**Can measures inspired by statistical physics detect transitions between low- and
high-volatility regimes earlier or more accurately than traditional volatility models?**

This is a forecasting study, not a physics simulation. Everything runs on free
daily OHLCV data and is reproducible end-to-end from one command.

---

## The idea

Markets switch between calm and turbulent phases in ways that look, at least
superficially, like phase transitions in physical systems: co-movement rises,
the system's response to shocks grows, and fluctuations become more persistent
before the switch. Statistical physics has a vocabulary for exactly this —
correlation length, order parameters, susceptibility, entropy, critical slowing
down. This project turns those concepts into **daily, causal features** and asks
one testable question:

> Do these features add measurable value for volatility regime detection and
> forecasting over established econometric baselines (EWMA, GARCH/EGARCH, HAR-RV)?

The headline deliverables, computed for every trading day:

1. a next-day realized-variance forecast,
2. the probability of being in (and entering) a High-volatility regime,
3. a Low / Medium / High regime label, and
4. a composite **criticality score** — the early-warning indicator and the
   project's novel contribution.

## Physics-inspired features

| Feature | Physics analogy | Construction (all rolling, past-only) |
|---|---|---|
| `corr_length` | correlation length | mean pairwise correlation across live sector ETFs, 63d |
| `eig_max_frac` | order parameter | top eigenvalue / k of the sector correlation matrix |
| `entropy_eigen` | spectral entropy | normalized entropy of the eigenvalue distribution |
| `frac_eig_above_mp` | random matrix theory | fraction of eigenvalues above the Marchenko–Pastur noise edge (genuine collective structure vs noise) |
| `entropy_returns` | disorder | entropy of binned standardized returns |
| `temperature` | temperature | dispersion × realized vol × (1 − avg corr), trailing z-score |
| `susceptibility` | χ = dM/dH | rolling slope of \|r_t\| on \|r_{t−1}\| |
| `acf1`, `var_of_vol` | critical slowing down | lag-1 autocorr of \|r\|; rolling Var(log RV) |
| `hurst_vol` | rough volatility | DFA exponent of log daily RV, 252d (validated on simulated fBm in tests) |
| `hawkes_branching` | seismology / self-organized criticality | branching ratio n of an exponential-kernel Hawkes process fit by rolling MLE to large-move event times; n → 1 means each shock triggers on average one aftershock — the system sits near criticality (estimator validated on simulated Hawkes in tests) |
| `criticality` | proximity to transition | mean trailing z-score of the pre-transition signals |

All cross-sectional features are computed on the **11 SPDR sector ETFs** with a
**dynamic, point-in-time universe**: at each date only ETFs whose trailing
window is fully populated participate, so XLRE (2015) and XLC (2018) enter
exactly when they listed and nothing is backfilled. Sector ETFs are used
instead of individual stocks deliberately — they are a fixed, investable
universe that avoids **survivorship bias**. Selecting today's index
constituents and extending them into the past would leak the survivors'
identities into history.

## How the question is answered

**Forecasting (Phases 1–2, 5).** Realized variance (daily Garman–Klass proxy,
no intraday data required) is forecast walk-forward on expanding windows at
**three horizons (1, 5, 22 days)**. Baselines: random walk, EWMA, GARCH(1,1),
EGARCH(1,1) (h=1), and HAR-RV at all horizons. The physics block enters via an
**elastic net** with in-window standardization and time-series CV — throwing
ten correlated, noisy regressors into plain OLS mostly measures overfitting,
not information. A **VIX ablation** (HAR vs HAR+VIX, each ± physics) shows
whether VIX crowds the physics features out. Diebold–Mariano tests are
reported **unconditionally and conditionally**: losses are also compared on
±10-day windows around model-free high-volatility episode onsets vs calm
days, because features built to detect transitions can be valuable near
transitions yet invisible in a full-sample average.

**Regimes (Phase 3).** Two layers. A full-sample 3-state Gaussian HMM (10 EM
restarts, states sorted by mean vol) provides *descriptive* labels for plots
only — full-sample labels are in-sample by construction and are never scored.
A **walk-forward HMM**, refit on expanding windows, produces a fully causal
P(High tomorrow) from the filtered state probabilities and transition matrix.

**Early warning (Phase 4).** Events are **model-free**: mean daily RV over the
next 5 days exceeding a past-only trailing quantile — no HMM in the labels, so
no leakage from a full-sample fit. Logistic classifiers are trained
**walk-forward** on expanding windows and scored only on days that are
currently calm by past-known information. Reported: ROC AUC and average
precision with **moving-block bootstrap confidence intervals** (block length
21 days, respecting serial dependence) for volatility-only, criticality-only,
and combined feature sets; a **lead-time table** (how many days before each
episode onset the criticality score first crossed its past-only 90th
percentile) with the accompanying **false-alarm rate**; and an **event-study
figure** of the average criticality and Hawkes-branching trajectories around
onsets vs placebo days.

**Interpretability (Phase 6).** Permutation importance on a held-out test set.

**Factor-style evaluation (Phase 6b).** Each physics feature is additionally
evaluated the way an alpha research desk would vet a candidate signal,
adapted to the time-series setting: rank **IC decay curves** over horizons
1–22 days against the *change* in forward log RV (so a feature can't score
by merely tracking today's vol level), with Newey–West t-statistics;
**orthogonalized IC** after walk-forward residualization against the
baseline information set (HAR terms + VIX) — the marginal information
beyond the existing "factor pool"; **quantile analysis** with past-only
trailing quintile buckets and a monotonicity score; and a feature
**redundancy matrix**. The vol-targeting backtest reports win rate, Calmar,
information ratio vs buy-and-hold, and an ex-post **PnL attribution by
volatility regime**.

**Economic value (Phase 7).** A 10%-target volatility-targeting strategy
(leverage cap 2×, 1 bp per unit turnover) is run off each model's forecasts and
compared on Sharpe ratio, maximum drawdown, and turnover, net of costs.

## Quick start

```bash
git clone <your-repo-url>
cd volatility-project
pip install -r requirements.txt

python src/01_download_data.py        # add --extended for TLT/GLD/USO/BTC-USD
python src/00_validate_data.py        # read this report before trusting anything
python run_all.py
```

Results land in `results/`:

| File | Contents |
|---|---|
| `forecast_metrics.csv`, `dm_test.txt` | out-of-sample forecast comparison |
| `regimes.parquet`, `transition_matrix.csv` | daily labels, probabilities, dynamics |
| `early_warning_auc.csv` | criticality vs volatility-only AUC |
| `backtest_metrics.csv`, `equity_curves.parquet` | economic evaluation |
| `feature_importance.csv` | permutation importance |
| `factor_ic.csv`, `factor_ic_orth.csv` | IC decay, raw and orthogonalized vs HAR+VIX |
| `factor_quantiles.csv`, `feature_corr.csv` | quantile monotonicity; feature redundancy |
| `pnl_attribution.csv` | strategy Sharpe and PnL share by volatility regime |
| `fig1`–`fig8` `.png` | signals, regime timeline, transition heatmap, importance, equity curves, event study, IC decay, feature correlation |

## Repository structure

```
volatility-project/
├── data/
│   ├── raw/                 downloaded OHLCV CSVs (gitignored)
│   └── processed/           aligned panels, returns, RV, features
├── src/
│   ├── common.py                  shared paths, constants, loaders
│   ├── 00_validate_data.py        data-quality report (hard/soft checks)
│   ├── 01_download_data.py        Yahoo Finance download with retries + diagnostics
│   ├── 02_clean_data.py           calendar alignment, log returns, summary stats
│   ├── 03_realized_volatility.py  Parkinson, Garman–Klass, Rogers–Satchell,
│   │                              Yang–Zhang, daily RV proxy
│   ├── 04_features.py             HAR components + model matrix (t -> t+1, lag-safe)
│   ├── 05_baseline_models.py      walk-forward forecasts, QLIKE, Diebold–Mariano
│   ├── 06_statistical_physics.py  physics features + criticality score
│   ├── 07_regime_detection.py     HMM/GMM regimes, early-warning AUC
│   ├── 08_backtest.py             volatility-targeting strategy
│   ├── 09_visualizations.py       figures + permutation importance
│   └── 10_factor_analysis.py      alpha-research-style IC / quantile / redundancy evaluation
├── tests/                   offline unit tests (no market data needed)
├── results/                 metrics, figures (generated)
├── paper/                   abstract / write-up
├── requirements.txt
└── run_all.py
```

## Testing

The test suite runs **without any market data** — it validates the math on
constructed inputs:

```bash
pytest tests/ -v
```

What is covered and why:

- **Estimator correctness** — close-to-close matches `numpy` exactly; a
  deterministic Parkinson case is checked in closed form; all range-based
  estimators recover the true σ of a simulated GBM with intraday sub-steps;
  constant prices give zero vol; inconsistent High/Low rows don't produce NaNs.
- **Metric properties** — QLIKE is zero at the truth, positive elsewhere, and
  penalizes under-prediction more than over-prediction; the Diebold–Mariano
  statistic is antisymmetric and insignificant under the null.
- **Physics features** — Hurst ≈ 0.5 for white noise and > 0.5 for persistent
  series; entropy stays in [0, 1] and orders concentrated vs diffuse
  distributions correctly; susceptibility recovers a known AR slope; coupled
  assets show higher correlation length / lower eigen-entropy than independent
  ones.
- **No look-ahead** — rewriting *future* observations must not change any
  feature, EWMA forecast, trailing quantile bucket, orthogonalization
  residual, Hawkes threshold, or event-label threshold in the past. This is
  the single most important test family in a forecasting project.
- **Factor machinery** — the IC statistic detects a planted signal and stays
  near zero on noise; the forward-change target responds only to the future;
  walk-forward residualization actually strips the baseline information.
- **Regime detection** — the HMM recovers three planted, well-separated states
  with correct ordering, persistent dynamics, and >85% label accuracy.

Separately, `src/00_validate_data.py` checks the *data*: schema, positive
prices, High/Low consistency, duplicate dates (hard failures), plus calendar
coverage, gaps, extreme returns, and cross-asset calendar overlap (warnings).

## Methodology guarantees

- **No look-ahead anywhere.** Row *t* of the model matrix contains only
  information available at the close of day *t*; all rolling statistics,
  z-scores, and exceedance thresholds use past-only windows; scalers and
  elastic-net hyperparameters are fit inside each training window; models
  (including the HMM used for causal P(High) and the early-warning
  classifiers) are refit walk-forward on expanding windows. Enforced by
  tests, not just intended.
- **No survivorship bias.** The cross-section is a fixed, investable sector
  ETF universe with point-in-time entry for later listings — never a
  backward-projected list of today's index members.
- **Common evaluation sample.** All models are scored on the identical set of
  out-of-sample days.
- **HMM caveat.** Regime labels fitted on the full sample are in-sample by
  construction; the early-warning evaluation therefore trains on the first half
  and scores AUC only on the second half.
- **RV proxy.** The daily Garman–Klass estimate stands in for intraday realized
  variance. If you have 5-minute data, swap the target in
  `03_realized_volatility.py` — nothing else changes.

## Data sources (all free)

- **Yahoo Finance** daily OHLCV via `yfinance`: SPY, QQQ, IWM, ^GSPC, ^VIX,
  optionally TLT, GLD, USO, BTC-USD. Note that yfinance relies on unofficial
  endpoints that change from time to time; if downloads fail, upgrade the
  package first.
- **FRED** for Treasury yields (optional extension, free API key).
- External realized-volatility libraries (e.g., the Oxford-Man realized
  library) can replace the RV proxy if you have access — availability of that
  dataset has changed over time, so verify it yourself before planning around it.

## Other physics domains worth borrowing from

Two are already implemented here: **random matrix theory** (Marchenko–Pastur
noise edge separating genuine collective modes from sampling noise) and
**seismology / self-organized criticality** (the Hawkes branching ratio as a
distance-to-criticality gauge — volatility clustering as an
aftershock cascade, with n → 1 meaning shocks become self-sustaining).
Natural next candidates: **turbulence / multifractal cascades** (the width of
the multifractal spectrum of returns as a feature — MF-DFA is a small
extension of the DFA code already in `06_statistical_physics.py`) and the
**fluctuation–dissipation theorem** (comparing the market's spontaneous
fluctuation level with its measured response to shocks; a growing violation
would signal a system far from equilibrium).

## Roadmap / stretch goals

Rough Bergomi and rough Heston forecasts, fractional Brownian motion
calibration of the Hurst estimates, Lévy-jump models, and neural operators —
as *comparison models*, not the centerpiece.

## References

Standard, widely cited papers behind the methods (verify exact citation
details before using them in a paper):

- Parkinson (1980) — extreme-value (high–low) variance estimation
- Garman & Klass (1980) — OHLC variance estimators
- Rogers & Satchell (1991) — drift-independent range estimator
- Yang & Zhang (2000) — estimator handling overnight jumps and drift
- Corsi (2009) — the HAR-RV model
- Diebold & Mariano (1995) — comparing predictive accuracy

## Disclaimer

Research code, not investment advice. Backtests overstate live performance;
results depend on data quality, sample period, and cost assumptions.
