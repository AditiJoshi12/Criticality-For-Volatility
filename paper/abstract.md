# Abstract

Financial markets exhibit abrupt transitions between calm and turbulent states
that standard econometric models struggle to anticipate. We test whether daily
features inspired by statistical physics — average cross-asset correlation
("correlation length"), the dominant eigenvalue of the correlation matrix as an
order parameter, entropy of the return distribution and of the eigenvalue
spectrum, a designed market temperature, a shock-response susceptibility,
critical-slowing-down diagnostics, and a rolling Hurst exponent — carry
measurable information about volatility regimes beyond established baselines
(EWMA, GARCH/EGARCH, HAR-RV). Using only free daily OHLCV data, we (i) forecast
next-day realized variance under walk-forward evaluation with QLIKE and
Diebold–Mariano tests, (ii) label Low/Medium/High regimes with a Gaussian HMM
and evaluate a composite criticality score as an out-of-sample early-warning
signal for entering the High regime, and (iii) translate forecasting gains into
economic terms via a cost-adjusted volatility-targeting backtest. The
contribution is not a new volatility model but a clean, reproducible answer to
one scientific question: do statistical-physics measures add value for
volatility regime detection and forecasting over established econometric
models?
