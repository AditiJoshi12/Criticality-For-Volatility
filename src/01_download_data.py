"""
01_download_data.py
Download daily OHLCV data from Yahoo Finance via the `yfinance` package.

Usage:
    python src/01_download_data.py               # core tickers
    python src/01_download_data.py --extended    # + TLT, GLD, USO, BTC-USD
    python src/01_download_data.py --start 2005-01-01

Notes:
- Requires internet access. yfinance scrapes Yahoo's endpoints, which
  change occasionally; if downloads fail, upgrade yfinance first
  (`pip install -U yfinance`) before debugging anything else.
- After downloading, run `python src/00_validate_data.py` to get a full
  data-quality report before proceeding with the pipeline.
"""
import argparse
import os
import time

import pandas as pd

from common import (CORE_TICKERS, EXTENSION_TICKERS, RAW_DIR, REQUIRED_COLS,
                    SECTOR_TICKERS, ensure_dirs, ticker_to_fname)

DEFAULT_START = "2000-01-01"
MAX_RETRIES = 3


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance sometimes returns a MultiIndex (field, ticker); flatten it."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def print_ticker_report(ticker: str, df: pd.DataFrame) -> None:
    """Per-ticker diagnostics printed at download time."""
    n = len(df)
    span_days = (df.index.max() - df.index.min()).days
    approx_expected = span_days * 252 / 365  # rough trading-day count
    nan_counts = df[REQUIRED_COLS].isna().sum()
    zero_vol = int((df["Volume"] == 0).sum()) if "Volume" in df else -1
    nonpos = int((df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1).sum())
    hl_violation = int((df["High"] < df["Low"]).sum())

    print(f"  rows: {n}   range: {df.index.min().date()} -> {df.index.max().date()}")
    print(f"  approx expected trading days in range: {approx_expected:,.0f} "
          f"(coverage ~{n / max(approx_expected, 1):.0%}; verify if far below 95%)")
    if nan_counts.sum():
        print(f"  NaNs by column: {nan_counts[nan_counts > 0].to_dict()}")
    if zero_vol > 0:
        print(f"  days with zero volume: {zero_vol} "
              f"(normal for indices like ^GSPC/^VIX; suspicious for ETFs)")
    if nonpos:
        print(f"  WARNING: {nonpos} rows with non-positive prices")
    if hl_violation:
        print(f"  WARNING: {hl_violation} rows where High < Low")


def download_one(ticker: str, start: str) -> pd.DataFrame | None:
    import yfinance as yf

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = yf.download(ticker, start=start, auto_adjust=False,
                             progress=False)
            df = flatten_columns(df)
            if not df.empty:
                return df
            print(f"  attempt {attempt}: empty frame returned")
        except Exception as e:  # noqa: BLE001 - report and retry
            print(f"  attempt {attempt} failed: {e}")
        time.sleep(2 * attempt)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extended", action="store_true",
                    help="also download TLT/GLD/USO/BTC-USD")
    ap.add_argument("--start", default=DEFAULT_START)
    args = ap.parse_args()

    ensure_dirs()
    tickers = CORE_TICKERS + SECTOR_TICKERS \
        + (EXTENSION_TICKERS if args.extended else [])
    failed = []
    for t in tickers:
        print(f"\nDownloading {t} ...")
        df = download_one(t, args.start)
        if df is None:
            failed.append(t)
            print(f"  FAILED after {MAX_RETRIES} attempts")
            continue
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            print(f"  WARNING: missing columns {missing} — "
                  f"yfinance schema may have changed")
        fname = ticker_to_fname(t)
        df.to_csv(os.path.join(RAW_DIR, fname))
        print(f"  saved data/raw/{fname}")
        print_ticker_report(t, df)

    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED tickers: {failed}")
        print("Try: pip install -U yfinance, check your connection, "
              "or rerun (Yahoo rate-limits occasionally).")
    else:
        print("All tickers downloaded.")
    print("Next: python src/00_validate_data.py")


if __name__ == "__main__":
    main()
