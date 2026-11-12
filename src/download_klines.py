"""Download one-minute Binance spot candles (USDT pairs) from data.binance.vision.

Reads:  monthly archives https://data.binance.vision/data/spot/monthly/klines/<SYM>/1m/
Writes: config.KLINE_DIR/<COIN>_1m.npz with arrays ts (ms), o, h, l, c, v, sorted by time and deduplicated.
Supports: all tables and figures (raw input of build_windows.py).

Each month is fetched in a thread pool, checked against the official .CHECKSUM (sha256) and parsed in
memory. Months before a coin was listed return 404 and are skipped; coins with no data at all are skipped.
Coins whose file already exists are not downloaded again (delete the file to refresh it).

Usage:
    python src/download_klines.py [--coins BTC,ETH,...] [--start 2017-08] [--end 2026-10]
"""

import argparse
import hashlib
import io
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import config

BASE = "https://data.binance.vision/data/spot/monthly/klines/{s}/1m/{s}-1m-{ym}.zip"
WORKERS = 16


def months(start, end):
    y, m = map(int, start.split("-"))
    y1, m1 = map(int, end.split("-"))
    while (y, m) <= (y1, m1):
        yield f"{y}-{m:02d}"
        m += 1
        if m > 12:
            y, m = y + 1, 1


def get(url, tries=4):
    for k in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def fetch(sym, ym):
    url = BASE.format(s=sym, ym=ym)
    try:
        raw = urllib.request.urlopen(url, timeout=60).read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raw = get(url)
    checksum = get(url + ".CHECKSUM").decode().split()[0]
    if hashlib.sha256(raw).hexdigest() != checksum:
        raise ValueError(f"checksum mismatch: {url}")
    z = zipfile.ZipFile(io.BytesIO(raw))
    a = np.loadtxt(
        io.TextIOWrapper(z.open(z.namelist()[0])), delimiter=",", usecols=(0, 1, 2, 3, 4, 5), ndmin=2
    )
    ts = a[:, 0].astype(np.int64)
    # Spot archives switched to microsecond timestamps in 2025.
    ts = np.where(ts > 10**14, ts // 1000, ts)
    return ts, a[:, 1:6]


def download_coin(coin, yms):
    path = config.KLINE_DIR / f"{coin}_1m.npz"
    if path.exists():
        print(f"{coin}: exists, skipped", flush=True)
        return
    t_start = time.time()
    sym = coin + "USDT"
    with ThreadPoolExecutor(WORKERS) as ex:
        parts = [x for x in ex.map(lambda ym: fetch(sym, ym), yms) if x is not None]
    if not parts:
        print(f"{coin}: no data, skipped", flush=True)
        return
    ts = np.concatenate([p[0] for p in parts])
    px = np.concatenate([p[1] for p in parts])
    order = np.argsort(ts, kind="stable")
    ts, px = ts[order], px[order]
    keep = np.r_[True, np.diff(ts) > 0]
    ts, px = ts[keep], px[keep]
    np.savez_compressed(path, ts=ts, o=px[:, 0], h=px[:, 1], l=px[:, 2], c=px[:, 3], v=px[:, 4])
    gaps = int(np.sum(np.diff(ts) > 60_000))
    print(f"{coin}: {len(ts):,} rows, {gaps} gaps, {time.time() - t_start:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--coins", default=",".join(config.UNIVERSE_COINS), help="comma-separated coin list")
    ap.add_argument("--start", default=config.SAMPLE_START, help="first month, YYYY-MM")
    ap.add_argument("--end", default=config.SAMPLE_END, help="last month, YYYY-MM")
    args = ap.parse_args()
    config.KLINE_DIR.mkdir(parents=True, exist_ok=True)
    yms = list(months(args.start, args.end))
    for coin in args.coins.split(","):
        download_coin(coin, yms)


if __name__ == "__main__":
    main()
