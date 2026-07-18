"""Run the full pipeline in order.

Prerequisite: real market data downloaded via
    python src/01_download_data.py
"""
import subprocess
import sys

STEPS = [
    "src/00_validate_data.py",         # abort early on bad data
    "src/02_clean_data.py",
    "src/03_realized_volatility.py",
    "src/06_statistical_physics.py",   # features must exist before 04
    "src/04_features.py",
    "src/05_baseline_models.py",
    "src/07_regime_detection.py",
    "src/08_backtest.py",
    "src/09_visualizations.py",
    "src/10_factor_analysis.py",
    "src/11_critical_transitions.py",
]

for s in STEPS:
    print(f"\n========== {s} ==========")
    r = subprocess.run([sys.executable, s])
    if r.returncode != 0:
        sys.exit(f"Step failed: {s}")
print("\nPipeline complete. See results/")
