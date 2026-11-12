"""Build non-overlapping return windows at several horizons from one-minute candles.

Reads:  config.KLINE_DIR/<COIN>_1m.npz (download_klines.py)
Writes: config.panel_path(h, sample) for h in 1h, 4h, 15h, 24h, 9h, 7d
Supports: all tables and figures (every analysis module reads these panels).

Usage:
    python src/build_windows.py [main|replication|point_in_time]     (default: main)

Windows (UTC, left-closed [T0, T1)):
    1h   every hour; BV from 1-minute returns; windows with more than 30% zero returns are dropped
    4h   starting 00/04/08/12/16/20; BV from 5-minute returns (as all horizons below)
    15h  17:00 to 08:00 next day (the overnight holding period of daily options)
    24h  starting 00:00 (main results)
    9h   08:00 to 17:00, the complement of the 15h window (time-of-day split)
    7d   starting Monday 00:00
The price at time T is the close of the candle that starts at T - 1 minute. Exchange outages are
forward-filled, and windows containing a run of more than 10 missing minutes are dropped.

Columns (window statistics):
    r            log(P(T1) / P(T0))
    bv           bipower variation (pi/2) sum |r_i||r_{i-1}| of intraday returns
    r_std        r / sqrt(bv), the standardized return z
    rv           realized variance sum r_i^2
    zfrac        share of intraday intervals with a zero return
    lev          corr(r_i, r_{i+1}^2) within the window (not an ex-ante variable)
    rs_pos/neg   upside / downside realized semivariance
    ac1          first-order autocorrelation of intraday returns (path smoothness)
    bv5          (1h only) BV from 5-minute returns, for microstructure robustness
State variables (prices up to T0 only):
    snr{W}       sum of past W daily returns / (std of daily returns * sqrt(W)), W in 30, 45, 90
    ret45/vol45  past 45-day log return / std of daily returns (the two parts of snr45)
    sgn{N}       sign of the past N-day log return, N in 30, 60, 90
    ma{N}        log(P(T0) / mean daily price over the past N days), N in 30, 60
Controls (before T0 only):
    lbv_lag      log bv of the previous window
    lrv1d/lrv7d  log 5-minute realized variance over the past day / past 7 days (daily average)
    lvol_rel     log(past-day quote volume / past 30-day average daily quote volume)
    r1d          past-day log return
    rstd_lag     r_std of the previous window
    dow, hour, year  calendar of T0 (dow 0 = Monday)
"""

import argparse
import datetime as dt
import math

import numpy as np
import pandas as pd

import config

KDIR = config.KLINE_DIR
COINS = config.MAIN_COINS
MIN = 60_000
DAY = 1440
# name: (length in minutes, offset of the first start within the UTC day, sampling step, spacing of starts)
HORIZONS = {
    "1h": (60, 0, 1, 60),
    "4h": (240, 0, 5, 240),
    "15h": (900, 17 * 60, 5, DAY),
    "24h": (DAY, 0, 5, DAY),
    "9h": (540, 8 * 60, 5, DAY),
    "7d": (7 * DAY, 0, 5, 7 * DAY),
}
WARM = 95 * DAY  # look-back needed by the state and control variables (90 days plus margin)
MAXGAP = 10  # longest run of missing minutes allowed inside a window


def load(coin):
    """Minute grid from the first UTC midnight: (t0 in ms, log prices, quote volume, missing-run length)."""
    z = np.load(KDIR / f"{coin}_1m.npz")
    ts, c, v = z["ts"], z["c"], z["v"] * z["c"]
    t0 = ts[0] - ts[0] % (DAY * MIN)
    n = int((ts[-1] - t0) // MIN) + 1
    k = ((ts - t0) // MIN).astype(np.int64)
    C = np.full(n, np.nan)
    C[k] = c
    V = np.zeros(n)
    V[k] = v
    miss = np.isnan(C)
    run = np.zeros(n, np.int32)
    for i in np.flatnonzero(miss):
        run[i] = run[i - 1] + 1 if i > 0 else 1
    s = pd.Series(C).ffill().bfill().to_numpy()
    # P[m] is the price at t0 + m minutes, i.e. the close of candle m - 1.
    P = np.empty(n + 1)
    P[0] = s[0]
    P[1:] = s
    return t0, np.log(P), V, run


def rv5_cum(lp):
    """S[m]: cumulative sum of squared 5-minute returns up to minute m (defined where m is a multiple of 5)."""
    S = np.zeros(len(lp))
    idx = np.arange(5, len(lp), 5)
    S[idx] = np.cumsum((lp[idx] - lp[idx - 5]) ** 2)
    return S


def daily_dt(t0):
    return dt.datetime.fromtimestamp(t0 / 1000, dt.UTC)


def _row_corr(a, b):
    """Row-wise Pearson correlation of a and b; NaN where either row has zero variance."""
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(den > 0, (a * b).sum(1) / np.where(den > 0, den, 1), np.nan)


def build(coin, hname, spec, lp, V, run, t0):
    """Panel of windows of one coin at one horizon; spec = (length, offset, step, spacing) in minutes."""
    L, off, step, gapm = spec
    n = len(lp) - 1
    if hname == "7d":  # first Monday
        wd = daily_dt(t0).weekday()
        off = ((7 - wd) % 7) * DAY
    starts = np.arange(off, n - L, gapm, dtype=np.int64)
    starts = starts[starts >= WARM]
    g = starts[:, None] + np.arange(0, L + 1, step)[None, :]
    lg = lp[g]
    ri = np.diff(lg, axis=1)
    r = lg[:, -1] - lg[:, 0]
    bv = (math.pi / 2) * np.sum(np.abs(ri[:, 1:]) * np.abs(ri[:, :-1]), axis=1)
    # Window [s, s+L) uses candles s-1 .. s+L-1.
    runmax = pd.Series(run).rolling(L + 1, min_periods=1).max().to_numpy()
    gap = runmax[np.minimum(starts + L - 1, len(run) - 1)] > MAXGAP
    out = {
        "coin": coin,
        "t0": t0 + starts * MIN,
        "r": r,
        "bv": bv,
        "rv": np.sum(ri**2, axis=1),
        "zfrac": np.mean(ri == 0, axis=1),
        "lev": _row_corr(ri[:, :-1], ri[:, 1:] ** 2),
        # Realized semivariances (Barndorff-Nielsen, Kinnebrock and Shephard 2010).
        "rs_pos": np.sum(np.where(ri > 0, ri**2, 0.0), axis=1),
        "rs_neg": np.sum(np.where(ri < 0, ri**2, 0.0), axis=1),
        "ac1": _row_corr(ri[:, :-1], ri[:, 1:]),
    }
    if step == 1:
        r5w = np.diff(lp[starts[:, None] + np.arange(0, L + 1, 5)[None, :]], axis=1)
        out["bv5"] = (math.pi / 2) * np.sum(np.abs(r5w[:, 1:]) * np.abs(r5w[:, :-1]), axis=1)

    # Daily log prices before T0: column k is k days earlier; dr column k is the return from day k+1 to day k.
    kd = np.arange(0, 91)
    dl = lp[np.maximum(starts[:, None] - kd[None, :] * DAY, 0)]
    dr = dl[:, :-1] - dl[:, 1:]
    for W in (30, 45, 90):
        x = dr[:, :W]
        out[f"snr{W}"] = x.sum(1) / (x.std(1, ddof=1) * math.sqrt(W))
    out["ret45"] = dl[:, 0] - dl[:, 45]
    out["vol45"] = dr[:, :45].std(1, ddof=1)
    for N in (30, 60, 90):
        out[f"sgn{N}"] = np.sign(dl[:, 0] - dl[:, N])
    for N in (30, 60):
        out[f"ma{N}"] = -np.log(np.mean(np.exp(dl[:, 1 : N + 1] - dl[:, :1]), axis=1))

    S = rv5_cum(lp)  # every window start is a multiple of 5 minutes
    out["lrv1d"] = np.log(S[starts] - S[np.maximum(starts - DAY, 0)] + 1e-12)
    out["lrv7d"] = np.log((S[starts] - S[np.maximum(starts - 7 * DAY, 0)]) / 7 + 1e-12)
    cv = np.concatenate([[0.0], np.cumsum(V)])
    v1 = cv[starts] - cv[np.maximum(starts - DAY, 0)]
    v30 = (cv[starts] - cv[np.maximum(starts - 30 * DAY, 0)]) / 30
    out["lvol_rel"] = np.log((v1 + 1) / (v30 + 1))
    out["r1d"] = lp[starts] - lp[np.maximum(starts - DAY, 0)]

    df = pd.DataFrame(out)
    df["r_std"] = df.r / np.sqrt(df.bv)
    df["lbv_lag"] = np.log(df.bv).shift(1)
    df["rstd_lag"] = df.r_std.shift(1)
    tt = pd.to_datetime(df.t0, unit="ms", utc=True)
    df["dow"] = tt.dt.weekday
    df["hour"] = tt.dt.hour
    df["year"] = tt.dt.year
    df = df[(~gap) & (df.bv > 0)].iloc[1:]
    if hname == "1h":
        # Near-flat hours have BV close to zero, which blows up r_std.
        df = df[df.zfrac <= 0.30]
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sample", nargs="?", default="main", choices=list(config.SAMPLES))
    args = ap.parse_args()
    acc = {h: [] for h in HORIZONS}
    for coin in config.SAMPLES[args.sample]:
        if not (KDIR / f"{coin}_1m.npz").exists():
            print(f"{coin}: no kline file, skipped", flush=True)
            continue
        t0, lp, V, run = load(coin)
        for h, spec in HORIZONS.items():
            d = build(coin, h, spec, lp, V, run, t0)
            acc[h].append(d)
            print(
                f"{coin} {h}: {len(d):,} windows {d.year.min()}-{d.year.max()} "
                f"r_std mean {d.r_std.mean():+.3f} sd {d.r_std.std():.3f}",
                flush=True,
            )
        del lp, V, run
    for h, parts in acc.items():
        path = config.panel_path(h, args.sample)
        path.parent.mkdir(parents=True, exist_ok=True)
        p = pd.concat(parts, ignore_index=True)
        p.to_csv(path, index=False, float_format="%.6g")
        print(f"{path.name} ({args.sample}): {len(p):,} rows", flush=True)


if __name__ == "__main__":
    main()
