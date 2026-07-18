"""
08_backtest.py  (Phase 7)

Practical evaluation: a volatility-targeting strategy on the target asset.
Position_t = min(cap, target_vol / forecast_vol_t), applied to return t+1.
One strategy per forecasting model, compared on Sharpe, max drawdown,
realized vol vs target, and turnover. Simple transaction cost applied.
"""
import os

import numpy as np
import pandas as pd

from common import ANN, PROC_DIR, RES_DIR, TARGET


TARGET_VOL = 0.10   # 10% annualized
LEV_CAP = 2.0
COST_BPS = 1.0      # per unit turnover, one-way


def stats(returns, benchmark=None):
    mu = returns.mean() * ANN
    sd = returns.std() * np.sqrt(ANN)
    sharpe = mu / sd if sd > 0 else np.nan
    curve = (1 + returns).cumprod()
    dd = (curve / curve.cummax() - 1).min()
    win = (returns > 0).mean()
    calmar = mu / abs(dd) if dd < 0 else np.nan
    ir = np.nan
    if benchmark is not None:
        active = (returns - benchmark.reindex(returns.index)).dropna()
        te = active.std() * np.sqrt(ANN)
        ir = active.mean() * ANN / te if te > 0 else np.nan
    return {"AnnRet": mu, "AnnVol": sd, "Sharpe": sharpe, "MaxDD": dd,
            "Calmar": calmar, "WinRate": win, "IR_vs_BH": ir}


def main():
    fc = pd.read_parquet(os.path.join(RES_DIR, "forecasts.parquet"))
    rets = pd.read_parquet(os.path.join(PROC_DIR, "log_returns.parquet"))["SPY"]
    r_next = rets.shift(-1).reindex(fc.index)  # position at t earns return t+1

    bh = r_next.dropna()
    rows, curves, strat_rets = {}, {}, {}
    for model in fc.columns:
        vol_fc = np.sqrt(np.exp(fc[model]) * ANN)  # annualized vol forecast
        w = (TARGET_VOL / vol_fc).clip(upper=LEV_CAP)
        turnover = w.diff().abs().fillna(0)
        strat = (w * r_next - COST_BPS / 1e4 * turnover).dropna()
        row = stats(strat, benchmark=bh)
        row.update({"AvgLev": w.mean(), "Turnover/yr": turnover.mean() * ANN})
        rows[model] = row
        curves[model] = (1 + strat).cumprod()
        strat_rets[model] = strat

    row = stats(bh)
    row.update({"AvgLev": 1.0, "Turnover/yr": 0.0})
    rows["Buy&Hold"] = row
    curves["Buy&Hold"] = (1 + bh).cumprod()
    strat_rets["Buy&Hold"] = bh

    table = pd.DataFrame(rows).T.sort_values("Sharpe", ascending=False)
    table.to_csv(os.path.join(RES_DIR, "backtest_metrics.csv"))
    pd.DataFrame(curves).to_parquet(os.path.join(RES_DIR, "equity_curves.parquet"))
    print("=== Vol-targeting backtest (net of costs) ===")
    print(table.round(3))

    # ---- PnL attribution by volatility regime (EX-POST DIAGNOSTIC) ----
    # Uses the descriptive full-sample HMM labels purely to attribute
    # realized PnL after the fact -- labels are never a trading input.
    reg_path = os.path.join(RES_DIR, "regimes.parquet")
    if os.path.exists(reg_path):
        reg = pd.read_parquet(reg_path)["regime_hmm"]
        attr = {}
        for model, sr in strat_rets.items():
            lab = reg.reindex(sr.index)
            attr[model] = {
                f"Sharpe_{name}": (sr[lab == name].mean() * ANN
                                   / (sr[lab == name].std() * np.sqrt(ANN)))
                if (lab == name).sum() > 50 else np.nan
                for name in ["Low", "Medium", "High"]}
            attr[model]["PnLshare_High"] = (
                sr[lab == "High"].sum() / sr.sum() if sr.sum() != 0 else np.nan)
        attr = pd.DataFrame(attr).T
        attr.to_csv(os.path.join(RES_DIR, "pnl_attribution.csv"))
        print("\n=== PnL attribution by (ex-post) volatility regime ===")
        print(attr.round(3))


if __name__ == "__main__":
    main()
