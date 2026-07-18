"""Shared paths, constants, and loaders used across the pipeline."""
import os

import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE, "data", "raw")
PROC_DIR = os.path.join(BASE, "data", "processed")
RES_DIR = os.path.join(BASE, "results")

ANN = 252            # trading days per year
TARGET = "SPY"       # primary asset for the study
ROLL_WINDOW = 63     # ~3 months, used by physics features

CORE_TICKERS = ["SPY", "QQQ", "IWM", "^GSPC", "^VIX"]

# Cross-sectional universe for correlation / collective-mode features.
# Sector ETFs are used deliberately to avoid SURVIVORSHIP BIAS: they are a
# fixed, investable universe that has existed since late 1998 (XLRE since
# 2015, XLC since 2018). Selecting today's index constituents and extending
# them backwards would leak the survivors' identities into the past.
# XLRE/XLC enter the universe only from their listing dates (point-in-time).
SECTOR_TICKERS = ["XLK", "XLF", "XLE", "XLV", "XLI",
                  "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]

EXTENSION_TICKERS = ["TLT", "GLD", "USO", "BTC-USD"]  # cross-asset tests
MIN_ASSETS = 6  # minimum live assets required for cross-sectional features

# The theoretical narrative: markets as interacting systems approaching a
# critical transition. Every CORE feature maps to one predicted mechanism.
# Auxiliary features are kept for robustness but are not part of the story
# (hurst_vol measures persistence of log-vol, from the rough-volatility
# literature; temperature is a designed index without a sharp physics
# interpretation).
MECHANISM_MAP = {
    "collective behaviour": ["corr_length", "eig_max_frac",
                             "frac_eig_above_mp"],
    "critical slowing down": ["acf1", "var_of_vol", "susceptibility"],
    "self-excitation": ["hawkes_branching"],
    "changing disorder": ["entropy_eigen", "entropy_returns"],
    "composite": ["criticality"],
}
CORE_FEATURES = [f for v in MECHANISM_MAP.values() for f in v]
AUX_FEATURES = ["hurst_vol", "temperature"]

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def ticker_to_fname(t: str) -> str:
    return t.replace("^", "").replace("-", "_") + ".csv"


def load_processed(name: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(PROC_DIR, name))


def ensure_dirs():
    for d in (RAW_DIR, PROC_DIR, RES_DIR):
        os.makedirs(d, exist_ok=True)


def high_vol_event_labels(rv, horizon=5, trail=252, q=0.85):
    """Model-free high-volatility-event labels.

    Label at day t: mean daily RV over t+1..t+horizon exceeds the trailing
    `trail`-day q-quantile of daily RV known at the close of day t.
    - The LABEL uses future data (it is the prediction target).
    - The THRESHOLD uses strictly past data (shifted trailing quantile),
      so no forecasting feature ever sees it as an input.
    Returns (labels, event_starts): labels is a 0/1/NaN Series; event_starts
    marks days where the label switches 0 -> 1 (onset of a new episode).
    """
    import numpy as _np
    fwd = rv.rolling(horizon).mean().shift(-horizon)          # future window
    thr = rv.rolling(trail).quantile(q).shift(1)              # past-only
    labels = _np.where(fwd.isna() | thr.isna(), _np.nan,
                       (fwd > thr).astype(float))
    labels = pd.Series(labels, index=rv.index, name="event")
    prev = labels.shift(1)
    starts = (labels == 1) & (prev == 0)
    return labels, starts
