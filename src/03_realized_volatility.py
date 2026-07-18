"""
03_realized_volatility.py  (Phase 1)
Range-based and return-based daily volatility estimators, annualized.

Estimators: close-to-close rolling, Parkinson (1980), Garman-Klass (1980),
Rogers-Satchell (1991), Yang-Zhang (2000).
"""
import os

import numpy as np
import pandas as pd

from common import ANN, PROC_DIR, RES_DIR, TARGET



def _load(field):
    return pd.read_parquet(os.path.join(PROC_DIR, f"{field}.parquet"))


def close_to_close(rets, window=21):
    return rets.rolling(window).std() * np.sqrt(ANN)


def parkinson(h, l, window=21):
    x = (np.log(h / l) ** 2) / (4 * np.log(2))
    return np.sqrt(x.rolling(window).mean() * ANN)


def garman_klass(o, h, l, c, window=21):
    x = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    return np.sqrt(x.clip(lower=0).rolling(window).mean() * ANN)


def rogers_satchell(o, h, l, c, window=21):
    x = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    return np.sqrt(x.clip(lower=0).rolling(window).mean() * ANN)


def yang_zhang(o, h, l, c, window=21):
    log_oc_prev = np.log(o / c.shift(1))       # overnight
    log_co = np.log(c / o)                     # open-to-close
    rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    var_o = log_oc_prev.rolling(window).var()
    var_c = log_co.rolling(window).var()
    var_rs = rs.rolling(window).mean()
    return np.sqrt((var_o + k * var_c + (1 - k) * var_rs).clip(lower=0) * ANN)


def daily_rv_proxy(o, h, l, c):
    """One-day Garman-Klass variance: the daily 'realized variance' proxy
    used as the forecasting target (no intraday data needed)."""
    x = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    return x.clip(lower=1e-10)


def main():
    o, h = _load("open"), _load("high")
    l, c = _load("low"), _load("close")
    rets = _load("log_returns")

    est = pd.DataFrame({
        "close_to_close": close_to_close(rets[TARGET]),
        "parkinson": parkinson(h[TARGET], l[TARGET]),
        "garman_klass": garman_klass(o[TARGET], h[TARGET], l[TARGET], c[TARGET]),
        "rogers_satchell": rogers_satchell(o[TARGET], h[TARGET], l[TARGET], c[TARGET]),
        "yang_zhang": yang_zhang(o[TARGET], h[TARGET], l[TARGET], c[TARGET]),
    }).dropna()
    est.to_parquet(os.path.join(PROC_DIR, "vol_estimators.parquet"))

    rv = daily_rv_proxy(o[TARGET], h[TARGET], l[TARGET], c[TARGET]).rename("rv")
    np.sqrt(rv * ANN).rename("rv_ann").to_frame().join(rv).dropna() \
        .to_parquet(os.path.join(PROC_DIR, "daily_rv.parquet"))

    print("Estimator correlation matrix:")
    print(est.corr().round(3))
    print("\nMean annualized vol by estimator:")
    print(est.mean().round(4))


if __name__ == "__main__":
    main()
