"""
00_validate_data.py
Data-quality report for the raw CSVs in data/raw/. Run this immediately
after 01_download_data.py and read the output before trusting anything
downstream. Exits with a nonzero code if any HARD check fails.

Checks per ticker
  HARD (pipeline will be wrong if these fail):
    - required columns present, parseable dates, numeric values
    - prices strictly positive
    - High >= max(Open, Close) and Low <= min(Open, Close), within tolerance
    - no duplicated dates
  SOFT (warnings you should investigate):
    - calendar coverage vs. approximate expected trading days
    - large gaps (> 7 calendar days) inside the sample
    - extreme daily moves (|log return| > 20%) listed for manual review
    - NaN counts, zero-volume days

Cross-ticker:
    - overlap of trading calendars (the pipeline uses the intersection)
    - date range that will survive alignment
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

from common import RAW_DIR, REQUIRED_COLS, SECTOR_TICKERS, ticker_to_fname

TOL = 1e-6          # tolerance for OHLC consistency (float noise in feeds)
EXTREME_RET = 0.20  # |log return| threshold for manual review


def check_ticker(name: str, df: pd.DataFrame):
    hard, soft = [], []

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        hard.append(f"missing columns: {missing}")
        return hard, soft

    df = df.apply(pd.to_numeric, errors="coerce")
    if df.index.duplicated().any():
        hard.append(f"{df.index.duplicated().sum()} duplicated dates")

    px = df[["Open", "High", "Low", "Close"]].dropna()
    if (px <= 0).any(axis=None):
        hard.append(f"{int((px <= 0).any(axis=1).sum())} rows with non-positive prices")

    bad_hi = (px["High"] + TOL < px[["Open", "Close"]].max(axis=1)).sum()
    bad_lo = (px["Low"] - TOL > px[["Open", "Close"]].min(axis=1)).sum()
    if bad_hi:
        hard.append(f"{int(bad_hi)} rows where High < max(Open, Close)")
    if bad_lo:
        hard.append(f"{int(bad_lo)} rows where Low > min(Open, Close)")

    # soft checks
    nans = df[REQUIRED_COLS].isna().sum()
    if nans.sum():
        soft.append(f"NaNs: {nans[nans > 0].to_dict()}")

    gaps = df.index.to_series().diff().dt.days
    big_gaps = gaps[gaps > 7]
    if len(big_gaps):
        worst = big_gaps.sort_values(ascending=False).head(3)
        soft.append(f"{len(big_gaps)} gaps > 7 calendar days; largest: "
                    + ", ".join(f"{d.date()} ({int(g)}d)" for d, g in worst.items()))

    r = np.log(df["Adj Close"] / df["Adj Close"].shift(1)).dropna()
    extreme = r[r.abs() > EXTREME_RET]
    if len(extreme):
        soft.append(f"{len(extreme)} days with |log return| > {EXTREME_RET:.0%}: "
                    + ", ".join(f"{d.date()} ({v:+.1%})"
                                for d, v in extreme.head(5).items())
                    + (" ..." if len(extreme) > 5 else "")
                    + " — verify these against another source (real crash days "
                      "look like this, but so do bad ticks and unadjusted splits)")

    span_days = (df.index.max() - df.index.min()).days
    approx_expected = span_days * 252 / 365
    cov = len(df) / max(approx_expected, 1)
    if cov < 0.95:
        soft.append(f"coverage ~{cov:.0%} of approx expected trading days "
                    f"— possible missing history")
    return hard, soft


def main():
    paths = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not paths:
        sys.exit("No raw CSVs found in data/raw/. Run 01_download_data.py first.")

    any_hard = False
    calendars = {}
    print("=" * 64)
    print("DATA QUALITY REPORT")
    print("=" * 64)
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
        df = df.apply(pd.to_numeric, errors="coerce")
        print(f"\n[{name}]  rows={len(df)}  "
              f"{df.index.min().date()} -> {df.index.max().date()}")
        hard, soft = check_ticker(name, df)
        for h in hard:
            print(f"  HARD FAIL: {h}")
        for s in soft:
            print(f"  warn: {s}")
        if not hard and not soft:
            print("  ok")
        any_hard |= bool(hard)
        if name != "VIX":
            calendars[name] = df.index

    target_name = "SPY"
    if len(calendars) >= 2 and target_name in calendars:
        tcal = calendars[target_name]
        print("\n" + "-" * 64)
        print(f"Pipeline calendar = {target_name} trading days: {len(tcal)} "
              f"({tcal.min().date()} -> {tcal.max().date()})")
        print("Per-asset inception (later listings enter the cross-sectional "
              "universe point-in-time; expected for XLRE ~2015, XLC ~2018):")
        for name, c in calendars.items():
            if name == target_name:
                continue
            late_start = c.min() > tcal.min()
            # within the asset's own lifetime, how many target days it misses
            lifetime = tcal[(tcal >= c.min()) & (tcal <= c.max())]
            missing = len(lifetime.difference(c))
            note = []
            if late_start:
                note.append(f"starts {c.min().date()}")
            if missing > 0.02 * len(lifetime):
                note.append(f"missing {missing} target days within its "
                            f"lifetime ({missing / len(lifetime):.0%}) — "
                            f"check for data holes or a different exchange "
                            f"calendar (e.g. BTC-USD trades 7 days/week)")
            if note:
                print(f"  {name}: " + "; ".join(note))

    missing_sectors = [t for t in SECTOR_TICKERS
                       if not os.path.exists(os.path.join(
                           RAW_DIR, ticker_to_fname(t)))]
    if missing_sectors:
        print("\n" + "-" * 64)
        print(f"warn: {len(missing_sectors)} sector ETF files missing: "
              f"{missing_sectors}")
        print("      Cross-sectional physics features (correlation length, "
              "order parameter,\n      spectral entropy, Marchenko-Pastur) "
              "need this universe. Re-run:\n      python src/01_download_data.py")

    print("\n" + "=" * 64)
    if any_hard:
        print("RESULT: HARD FAILURES — fix the data before running the pipeline.")
        sys.exit(1)
    print("RESULT: no hard failures. Review warnings, then run: python run_all.py")


if __name__ == "__main__":
    main()
