"""Download Binance USD-margined perpetual funding rates from data.binance.vision.

Reads:  monthly archives https://data.binance.vision/data/futures/um/monthly/fundingRate/<SYM>/
Writes: config.FUNDING_DIR/funding.csv with columns coin, ts (ms), rate.
Supports: Table A11 (funding rates and the skewness gap, via mechanism.py).

Months before a contract was listed return 404 and are skipped.

Usage:
    python src/download_funding.py [--coins BTC,ETH,...] [--start 2020-01] [--end 2026-10]
"""

import argparse
import io
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import config

BASE = "https://data.binance.vision/data/futures/um/monthly/fundingRate/{s}/{s}-fundingRate-{ym}.zip"
FUNDING_FILE = config.FUNDING_DIR / "funding.csv"


def months(start, end):
    y, m = map(int, start.split("-"))
    y1, m1 = map(int, end.split("-"))
    while (y, m) <= (y1, m1):
        yield f"{y}-{m:02d}"
        m += 1
        if m > 12:
            y, m = y + 1, 1


def fetch(sym, ym):
    try:
        raw = urllib.request.urlopen(BASE.format(s=sym, ym=ym), timeout=60).read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    z = zipfile.ZipFile(io.BytesIO(raw))
    d = pd.read_csv(z.open(z.namelist()[0]))
    if "calc_time" not in d.columns:  # early files have no header row
        d = pd.read_csv(
            z.open(z.namelist()[0]),
            header=None,
            names=["calc_time", "funding_interval_hours", "last_funding_rate"],
        )
    return d[["calc_time", "last_funding_rate"]].rename(
        columns={"calc_time": "ts", "last_funding_rate": "rate"}
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--coins", default=",".join(config.MAIN_COINS), help="comma-separated coin list")
    ap.add_argument("--start", default="2020-01", help="first month, YYYY-MM")
    ap.add_argument("--end", default=config.SAMPLE_END, help="last month, YYYY-MM")
    args = ap.parse_args()
    yms = list(months(args.start, args.end))
    out = []
    for coin in args.coins.split(","):
        with ThreadPoolExecutor(8) as ex:
            parts = [x for x in ex.map(fetch, [coin + "USDT"] * len(yms), yms) if x is not None]
        d = pd.concat(parts)
        d["coin"] = coin
        out.append(d)
        print(coin, len(d), pd.to_datetime(d.ts.min(), unit="ms").date(), flush=True)
    config.FUNDING_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(out)[["coin", "ts", "rate"]].sort_values(["coin", "ts"]).to_csv(FUNDING_FILE, index=False)


if __name__ == "__main__":
    main()
