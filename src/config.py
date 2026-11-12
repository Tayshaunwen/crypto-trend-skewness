"""Paths, coin lists and shared settings.

All scripts read their locations from here. By default everything lives under
``data/`` and ``results/`` in the repository root; set ``CTS_DATA`` or
``CTS_RESULTS`` to put the (large) data or the outputs elsewhere.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("CTS_DATA", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("CTS_RESULTS", ROOT / "results"))

KLINE_DIR = DATA_DIR / "klines"  # one-minute candles, <COIN>_1m.npz
FUNDING_DIR = DATA_DIR / "funding"  # perpetual-futures funding rates
PANEL_DIR = DATA_DIR / "panels"  # window panels, <sample>/panel_<horizon>.csv.gz
FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

# Main sample: ten large coins (Section 3.1 of the paper).
MAIN_COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC"]
# Replication sample: processed only after the specification was fixed.
REPLICATION_COINS = ["TRX", "XLM", "ETC", "BCH", "EOS", "ATOM", "XTZ", "VET", "ALGO", "DOT"]
# Further candidates for the point-in-time universe (some later delisted or collapsed).
EXTRA_COINS = [
    "LUNA",
    "FTT",
    "MATIC",
    "UNI",
    "FIL",
    "ICP",
    "NEAR",
    "SHIB",
    "XMR",
    "NEO",
    "IOTA",
    "DASH",
    "ZEC",
    "ONT",
    "QTUM",
    "AAVE",
    "APT",
    "ARB",
    "OP",
    "SUI",
    "WAVES",
]
UNIVERSE_COINS = MAIN_COINS + REPLICATION_COINS + EXTRA_COINS

SAMPLES = {"main": MAIN_COINS, "replication": REPLICATION_COINS, "point_in_time": UNIVERSE_COINS}
SAMPLE_START, SAMPLE_END = "2017-08", "2026-10"
HORIZONS = ["1h", "4h", "15h", "24h", "7d"]


def panel_path(horizon, sample="main"):
    return PANEL_DIR / sample / f"panel_{horizon}.csv.gz"


def result_path(name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / name
