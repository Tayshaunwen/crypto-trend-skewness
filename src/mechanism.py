"""Mechanism checks for the uptrend-minus-downtrend skewness gap (main specification, 45-day blocks).

    expost_*            ex-post states from the 45 days right after the window, from 45 days
                        centred on the window, and from the 45 days starting 90 days after it
    BV channel          by state: corr(r, log BV) and corr(r, BV surprise), where the surprise is
                        the residual of log BV on lbv_lag, lrv1d and lrv7d; gap within terciles of
                        the surprise, weighted by tercile size
    signed jumps        SJ = (RS+ - RS-) / RV by state; jump windows = (RV - BV) / RV in a coin's
                        top 10%; gap without and with only jump windows
    event weeks         gap without COVID (2020-03), May 2021, LUNA (2022-05) and FTX (2022-11)
    funding             average perpetual funding rate over the 7 days before the window; gap for
                        windows below and above the coin median, and low-minus-high skewness in uptrends
    1h sampling         1h gap with BV from 5-minute instead of 1-minute returns
    clock               8-hour windows starting at 00, 04, ..., 20 UTC (built from the minute klines);
                        and the 15h-window gap minus the 9h-window gap on common dates

Reads:  main-sample panels (1h, 4h, 9h, 15h, 24h), funding.csv, one-minute klines of the main coins.
Writes: mechanism_checks.csv
Paper:  Table 4, online Tables A11 and A12, Figure A2; numbers quoted in the text.

Usage:  python src/mechanism.py [bootstrap replications, default 999]
"""

import argparse

import numpy as np
import pandas as pd

import build_windows as W
import config
import episodes as ep
import skew_by_state as sk

DAYMS = 86_400_000
EVENTS = [
    ("2020-03-09", "2020-03-20"),
    ("2021-05-12", "2021-05-26"),
    ("2022-05-05", "2022-05-18"),
    ("2022-11-06", "2022-11-16"),
]
WINDOW_DAYS = {"4h": 4 / 24, "15h": 15 / 24, "24h": 1.0}
STATES = ((1, "up"), (0, "rg"), (-1, "dn"))


def rec(rows, h, check, est, bs=None, extra=None):
    d = dict(h=h, check=check, est=est)
    if bs is not None:
        lo, hi = sk.ci(bs)
        d.update(lo=lo, hi=hi, p=sk.pval(bs))
    if extra:
        d.update(extra)
    rows.append(d)
    tail = f" [{d['lo']:+.4f},{d['hi']:+.4f}] p={d['p']:.3f}" if bs is not None else ""
    print(f"{h:>6} {check:<36} {est:+.4f}{tail}", flush=True)


def gap_boot(p, col, h, gcol="g"):
    R, G = ep.mats(p.assign(gg=p[gcol]), [col, "gg"])
    return ep.gap_fn(sk.g1)(R, G), ep.boot(ep.gap_fn(sk.g1), [R, G], sk.block_for(h))


def strat_gap(R, G, Q):
    """Up-minus-down gap within each stratum of Q, averaged with stratum sizes as weights."""
    r, g, q = ep.flat(R, G, Q)
    tot = w = 0.0
    for k in np.unique(q):
        s = q == k
        if (s & (g == 1)).sum() > 30 and (s & (g == -1)).sum() > 30:
            tot += s.sum() * (sk.g1(r[s & (g == 1)]) - sk.g1(r[s & (g == -1)]))
            w += s.sum()
    return tot / w


def future_snr(p, start_days, length_days=45):
    """SNR of daily returns over [t0 + start_days, t0 + start_days + length_days].

    Approximated by snr45 of the window that starts at the end of that period (exact for 45 days).
    """
    out = np.full(len(p), np.nan)
    for _, d in p.groupby("coin"):
        s, t = d.snr45.to_numpy(), d.t0.to_numpy()
        i = np.searchsorted(t, t + int((start_days + length_days) * DAYMS))
        ok = i < len(t)
        v = np.full(len(d), np.nan)
        v[ok] = s[i[ok]]
        out[d.index.to_numpy()] = v
    return out


def funding_7d(p, fr):
    """Mean funding rate over the 7 days before each window (NaN with fewer than 15 fixings)."""
    out = pd.Series(np.nan, index=p.index)
    for c, d in p.groupby("coin"):
        f = fr[fr.coin == c].sort_values("ts")
        if f.empty:
            continue
        cs = np.concatenate([[0.0], np.cumsum(f.rate.to_numpy())])
        tt = f.ts.to_numpy()
        i1 = np.searchsorted(tt, d.t0.to_numpy(), "left")
        i0 = np.searchsorted(tt, d.t0.to_numpy() - 7 * DAYMS, "left")
        n = i1 - i0
        out.loc[d.index] = np.where(n >= 15, (cs[i1] - cs[i0]) / np.maximum(n, 1), np.nan)
    return out


def horizon_checks(rows, h, fr):
    p = ep.load(h)

    for name, start in (
        ("expost_immediate", WINDOW_DAYS[h]),
        ("expost_centered", -22.5 + WINDOW_DAYS[h] / 2),
        ("expost_plus90", 90),
    ):
        p["gf"] = ep.tri(future_snr(p, start))
        e, bs = gap_boot(p.dropna(subset=["gf"]).reset_index(drop=True), "r_std", h, "gf")
        rec(rows, h, name, e, bs)

    X = np.column_stack([np.ones(len(p)), p.lbv_lag, p.lrv1d, p.lrv7d])
    y = np.log(p.bv)
    ok = np.all(np.isfinite(X), 1) & np.isfinite(y)
    beta = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
    p["bvs"] = np.where(ok, y - X @ beta, np.nan)
    for gv, nm in STATES:
        d = p[(p.g == gv) & ok]
        rec(rows, h, f"corr_r_logBV_{nm}", np.corrcoef(d.r, np.log(d.bv))[0, 1])
        rec(rows, h, f"corr_r_BVsurprise_{nm}", np.corrcoef(d.r, d.bvs)[0, 1])
    p["q"] = pd.qcut(p.bvs, 3, labels=False).astype(float)
    R, G, Q = ep.mats(p, ["r_std", "g", "q"])
    rec(
        rows,
        h,
        "gap_within_BVsurprise_terciles",
        strat_gap(R, G, Q),
        ep.boot(strat_gap, [R, G, Q], sk.block_for(h)),
    )

    p["sj"] = (p.rs_pos - p.rs_neg) / p.rv
    p["jr"] = (p.rv - p.bv).clip(lower=0) / p.rv
    p["jump"] = p.jr > p.groupby("coin").jr.transform(lambda x: x.quantile(0.9))
    for gv, nm in STATES:
        d = p[p.g == gv]
        rec(
            rows,
            h,
            f"signed_jump_mean_{nm}",
            d.sj.mean(),
            extra=dict(up_jump_share=np.mean(d.jump & (d.r > 0)), dn_jump_share=np.mean(d.jump & (d.r < 0))),
        )
    e, bs = gap_boot(p[~p.jump].reset_index(drop=True), "r_std", h)
    rec(rows, h, "gap_excluding_jump_windows", e, bs)
    e, bs = gap_boot(p[p.jump].reset_index(drop=True), "r_std", h)
    rec(rows, h, "gap_jump_windows_only", e, bs)

    ts = pd.to_datetime(p.t0, unit="ms")
    in_event = np.zeros(len(p), bool)
    for a, b in EVENTS:
        in_event |= (ts >= a) & (ts <= b)
    e, bs = gap_boot(p[~in_event].reset_index(drop=True), "r_std", h)
    rec(rows, h, "gap_excluding_event_weeks", e, bs)

    p["fr7"] = funding_7d(p, fr)
    q = p.dropna(subset=["fr7"]).reset_index(drop=True)
    q["frlow"] = (q.fr7 < q.groupby("coin").fr7.transform("median")).astype(float)
    for label, sel in (("low_funding", q.frlow == 1), ("high_funding", q.frlow == 0)):
        e, bs = gap_boot(q[sel].reset_index(drop=True), "r_std", h)
        rec(rows, h, f"gap_{label}", e, bs)

    def low_minus_high(R, F):
        r, f = ep.flat(R, F)
        return sk.g1(r[f == 1]) - sk.g1(r[f == 0])

    up = q[q.g == 1]
    R, F = ep.mats(up.assign(f=up.frlow), ["r_std", "f"])
    rec(
        rows,
        h,
        "uptrend_skew_low_minus_high_funding",
        low_minus_high(R, F),
        ep.boot(low_minus_high, [R, F], sk.block_for(h)),
    )


def sampling_checks(rows):
    p = ep.load("1h")
    p["z5"] = p.r / np.sqrt(p.bv5)
    e, bs = gap_boot(p[np.isfinite(p.z5)].reset_index(drop=True), "z5", "1h")
    rec(rows, "1h", "gap_bv_5min_sampling", e, bs)
    e, bs = gap_boot(p, "r_std", "1h")
    rec(rows, "1h", "gap_bv_1min_sampling", e, bs)


def clock_checks(rows):
    segs = {f"8h@{s:02d}": (480, s * 60, 5, W.DAY) for s in (0, 4, 8, 12, 16, 20)}
    parts = {k: [] for k in segs}
    for coin in config.MAIN_COINS:
        t0, lp, V, run = W.load(coin)
        for k, spec in segs.items():
            parts[k].append(W.build(coin, k, spec, lp, V, run, t0))
    for k in segs:
        p = pd.concat(parts[k], ignore_index=True)
        p["g"] = ep.tri(p.snr45.to_numpy())
        e, bs = gap_boot(p, "r_std", "24h")
        rec(rows, k, "clock_gap", e, bs, extra=dict(g1_all=sk.g1(p.r_std.to_numpy())))

    # 15h (17:00-08:00) gap minus 9h (08:00-17:00) gap; both indexed by calendar day of t0 and
    # resampled jointly by day.
    a, b = ep.load("15h"), ep.load("9h")
    a["d"] = a.t0 // DAYMS
    b["d"] = b.t0 // DAYMS
    days = np.intersect1d(a.d.unique(), b.d.unique())
    Ra, Ga = ep.mats(a[a.d.isin(days)].assign(t0=lambda x: x.d), ["r_std", "g"])
    Rb, Gb = ep.mats(b[b.d.isin(days)].assign(t0=lambda x: x.d), ["r_std", "g"])

    def gap_difference(Ra, Ga, Rb, Gb):
        return ep.gap_fn(sk.g1)(Ra, Ga) - ep.gap_fn(sk.g1)(Rb, Gb)

    rec(
        rows,
        "15h-9h",
        "difference_in_gaps",
        gap_difference(Ra, Ga, Rb, Gb),
        ep.boot(gap_difference, [Ra, Ga, Rb, Gb], 45),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("reps", nargs="?", type=int, default=999, help="bootstrap replications")
    ep.NB = ap.parse_args().reps
    fr = pd.read_csv(config.FUNDING_DIR / "funding.csv")
    rows = []
    # Order matters: all bootstraps draw from one generator in skew_by_state.
    for h in ["24h", "15h", "4h"]:
        horizon_checks(rows, h, fr)
    sampling_checks(rows)
    clock_checks(rows)
    pd.DataFrame(rows).to_csv(config.result_path("mechanism_checks.csv"), index=False)


if __name__ == "__main__":
    main()
