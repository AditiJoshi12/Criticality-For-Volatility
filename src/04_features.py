"""
04_features.py
Assemble the model matrix: HAR components of realized variance plus
(optionally) the statistical-physics features from 06_statistical_physics.py.

Everything is lagged so that row t contains only information available
at the close of day t, used to predict RV at day t+1.
"""
import os

import numpy as np
import pandas as pd

from common import ANN, PROC_DIR, RES_DIR, TARGET



def build():
    rv = pd.read_parquet(os.path.join(PROC_DIR, "daily_rv.parquet"))["rv"]
    log_rv = np.log(rv)

    X = pd.DataFrame(index=rv.index)
    X["rv_d"] = log_rv                                  # daily
    X["rv_w"] = log_rv.rolling(5).mean()                # weekly
    X["rv_m"] = log_rv.rolling(21).mean()               # monthly

    # Multi-horizon targets: log of mean daily RV over the next h days.
    # rv.rolling(h).mean().shift(-h) at index t covers exactly t+1 .. t+h.
    targets = {}
    for h in (1, 5, 22):
        targets[f"target_h{h}"] = np.log(
            rv.rolling(h).mean().shift(-h)).rename(f"target_h{h}")
    y = pd.concat(targets.values(), axis=1)

    phys_path = os.path.join(PROC_DIR, "physics_features.parquet")
    phys = pd.read_parquet(phys_path) if os.path.exists(phys_path) else None

    vix_path = os.path.join(PROC_DIR, "vix.parquet")
    if os.path.exists(vix_path):
        X["log_vix"] = np.log(pd.read_parquet(vix_path)["VIX"]).reindex(X.index)

    df = X.join(y)
    if phys is not None:
        df = df.join(phys.drop(columns=["n_assets"], errors="ignore"))

    all_nan = [c for c in df.columns if df[c].isna().all()]
    if all_nan:
        print(f"WARNING: dropping all-NaN feature columns {all_nan} — "
              f"usually caused by missing sector ETF data "
              f"(re-run src/01_download_data.py).")
        df = df.drop(columns=all_nan)
    df = df.dropna()
    if len(df) == 0:
        raise SystemExit(
            "Model matrix is EMPTY after dropna(). Check the feature "
            "coverage printed by 06_statistical_physics.py and the raw "
            "data in data/raw/. Aborting so downstream steps don't emit "
            "NaN results.")
    df.to_parquet(os.path.join(PROC_DIR, "model_matrix.parquet"))
    print(f"Model matrix: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


if __name__ == "__main__":
    build()
