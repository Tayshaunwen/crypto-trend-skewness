# Trend-dependent skewness in cryptocurrency returns

Replication code for the paper *Trend-Dependent Skewness in Cryptocurrency Returns* (Tianxiao Wen, 2026).

The paper asks whether the asymmetry of cryptocurrency returns depends on the preceding trend.
Non-overlapping windows from one hour to one week are classified as following an uptrend, a range
or a downtrend by the t-statistic of the previous 45 daily returns, and skewness is compared across
states. At the 15- and 24-hour horizons, returns are right-skewed after strong uptrends and close to
symmetric after downtrends.

All data are public: one-minute spot candles and perpetual funding rates from
[data.binance.vision](https://data.binance.vision). Nothing in this repository requires an API key.

## Quick start

```bash
git clone https://github.com/Tayshaunwen/crypto-trend-skewness.git
cd crypto-trend-skewness
python3 -m pip install -r requirements.txt
bash run_all.sh
```

`run_all.sh` downloads about 2 GB of data and runs every step in order; on one CPU core the whole
run takes roughly ten hours, most of it in the bootstrap and Monte Carlo steps. Tables are written
as Markdown to `results/tables/` and figures to `results/figures/`.

To keep the data or the outputs elsewhere, set `CTS_DATA` or `CTS_RESULTS` before running.
Python 3.10 or later is required.

## Layout

```
src/config.py               paths, coin lists, sample period
src/download_klines.py      one-minute candles of 41 USDT pairs
src/download_funding.py     perpetual funding rates
src/build_windows.py        returns, realized measures and trend states per window
src/skew_by_state.py        skewness by trend state; shared statistics and bootstrap
...                         one module per group of results (see below)
src/figures.py              Figures 1-4, A1-A2
src/tables.py               Tables 1-8, A1-A12 and the numbers quoted in the text
run_all.sh                  the full pipeline in order
```

## Where each result comes from

| Paper | Module(s) |
|---|---|
| Tables 1–2, A1, A4 | `skew_by_state.py`, `episodes.py` |
| Table 3, Figure 1 | `skew_by_state.py`, `scaling_checks.py`, `extra_checks.py` |
| Table 4 (alternative explanations) | `mechanism.py`, `scaling_checks.py` |
| Table 5 (robustness) | `robustness.py`, `inference_checks.py`, `quantile_regression.py`, `episodes.py`, `scaling_checks.py`, `extra_checks.py` |
| Table 6 (density forecasts) | `oos_forecasts.py`, `var_and_payoffs.py` |
| Tables 7–8, A9–A10 (Value-at-Risk, payoffs) | `var_and_payoffs.py` |
| Figures 2–4 | `episodes.py` |
| Table A2, Figure A1 | `state_rules.py` |
| Tables A3, A5, A8 | `extra_checks.py`, `scaling_checks.py` |
| Table A6 | `monte_carlo.py` |
| Table A7 | `quantile_regression.py` |
| Tables A11–A12, Figure A2 | `mechanism.py`, `time_of_day.py`, `scaling_checks.py` |

## Reproducibility

Bootstrap and simulation seeds are fixed. Several modules draw from one random stream defined in
`skew_by_state.py`, in a fixed order; running the modules as in `run_all.sh` reproduces the
published numbers exactly. Changing the order of parts within a module, or the number of
replications, changes the bootstrap draws but not the point estimates.

The point-in-time universe uses a fixed list of 41 candidate pairs (`config.UNIVERSE_COINS`).
Binance occasionally removes files of delisted pairs from the public repository; if a pair can no
longer be downloaded, the corresponding results will differ slightly.

## License

MIT, see `LICENSE`.
