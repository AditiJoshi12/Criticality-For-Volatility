"""
02_clean_data.py
Align all tickers on a common trading calendar, compute log returns,
and save one tidy panel per field plus a returns matrix.
"""
import glob
import os

import numpy as np
import pandas as pd

from common import PROC_DIR, RAW_DIR, TARGET


def load_raw():
    frames = {}
    for path in sorted(glob.glob(os.path.join(RAW_DIR, "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        # coerce numerics (yfinance sometimes writes stray header rows)
        df = df.apply(pd.to_numeric, errors="coerce")
        frames[name] = df
    return frames


def main():
    os.makedirs(PROC_DIR, exist_ok=True)
    frames = load_raw()
    if not frames:
        raise SystemExit("No raw data found. Run 01_download_data.py first.")

    equity_names = [n for n in frames if n != "VIX"]
    if TARGET not in frames:
        raise SystemExit(f"Target asset {TARGET} missing from data/raw/.")

    # POINT-IN-TIME CALENDAR: use the TARGET asset's trading days, not the
    # intersection across all names. Intersecting would silently truncate
    # the whole sample to the youngest asset's history (e.g. XLC lists in
    # 2018), destroying two decades of data. Assets that list later simply
    # have NaNs before their inception; downstream cross-sectional features
    # select the live universe at each date.
    idx = frames[TARGET].index

    panels = {}
    for field in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        panels[field] = pd.DataFrame(
            {n: frames[n].reindex(idx)[field] for n in equity_names})

    close = panels["Adj Close"]
    # NOTE: no forward-fill of prices across an asset's pre-inception period;
    # returns are NaN until the asset actually trades.
    rets = np.log(close / close.shift(1))
    rets = rets[rets[TARGET].notna()]

    for field, df in panels.items():
        df.to_parquet(os.path.join(PROC_DIR, f"{field.lower().replace(' ', '_')}.parquet"))
    rets.to_parquet(os.path.join(PROC_DIR, "log_returns.parquet"))

    if "VIX" in frames:
        vix = frames["VIX"]["Close"].reindex(idx).ffill()
        vix.to_frame("VIX").to_parquet(os.path.join(PROC_DIR, "vix.parquet"))

    print(f"Cleaned panel: {rets.shape[0]} days x {rets.shape[1]} assets "
          f"({rets.index.min().date()} to {rets.index.max().date()}), "
          f"calendar = {TARGET} trading days")
    inception = rets.apply(lambda c: c.first_valid_index())
    late = inception[inception > rets.index.min()]
    if len(late):
        print("Assets entering the universe after sample start "
              "(point-in-time, expected for XLRE/XLC):")
        for n, d in late.items():
            print(f"  {n}: {d.date()}")
    print("\nPer-asset daily log-return summary (sanity-check against "
          "known history — e.g. SPY annualized vol is typically ~15-20%):")
    summary = pd.DataFrame({
        "mean_ann": rets.mean() * 252,
        "vol_ann": rets.std() * np.sqrt(252),
        "skew": rets.skew(),
        "kurtosis": rets.kurt(),
        "worst_day": rets.min(),
        "best_day": rets.max(),
        "n_missing": rets.isna().sum(),
    })
    print(summary.round(4))


if __name__ == "__main__":
    main()
