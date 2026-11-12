#!/usr/bin/env bash
# Reproduce every table and figure of the paper, from raw data to Markdown tables.
# Run from the repository root:  bash run_all.sh
# Downloads about 2 GB of one-minute candles; the full run takes roughly 10 hours on one core.
set -euo pipefail
cd "$(dirname "$0")"
py() { echo "== $(date +%H:%M:%S) $*"; python3 "src/$@"; }

# 1. Data
py download_klines.py                 # 41 USDT pairs, Aug 2017 - Oct 2026
py download_funding.py                # perpetual funding rates of the main coins
for sample in main replication point_in_time; do
    py build_windows.py "$sample"
done

# 2. Skewness by trend state and robustness
py skew_by_state.py 1999              # Tables 1-3, A3-A4; Figure 1
py state_rules.py 999                 # Table A2; Figure A1
py quantile_regression.py             # Tables 5, A7
py quantile_regression.py --no-controls
py quantile_regression.py --bootcheck 24h
py quantile_regression.py --bootcheck 15h
py robustness.py 999                  # Table 5
py inference_checks.py 999            # Table 5
py time_of_day.py 999                 # Table A12
py episodes.py                        # Tables 1, 5, A1; Figures 2-4
py mechanism.py 999                   # Tables 3-4, A11-A12; Figure A2
py scaling_checks.py                  # Tables 3-5, A3, A8, A12; Figure 1
py extra_checks.py                    # Tables 3, 5, A3, A5, A8

# 3. Forecasting, Value-at-Risk and payoffs
py oos_forecasts.py                   # Table 6
py var_and_payoffs.py                 # Tables 6-8, A9, A10

# 4. Monte Carlo size of the bootstrap test (Table A6)
py monte_carlo.py nodrift 1000
py monte_carlo.py drift 1000
py monte_carlo.py asymvol 500 199 0.05
py monte_carlo.py statevol 500 199 0.08

# 5. Figures and tables
py figures.py
py tables.py
echo "== done: results/figures, results/tables"
