"""Uptrend-minus-downtrend skewness gap under other scalings, plus related mechanism checks.

Trend states are the three snr45 states (threshold 0.75); inference uses the stationary
bootstrap with 45-day blocks of dates, resampled jointly across coins.

Parts
  S  gap in g1, Bowley and Q10/90 skewness for four scalings of returns: own-window BV
     (baseline z), previous-window BV, past-day RV rescaled to the window (ex-ante),
     and the coin's full-sample standard deviation                    -> scalings.csv
  A  path smoothness: intra-window autocorrelation by state and the gap within
     autocorrelation terciles                                         -> smoothness.csv
  T  past return vs past volatility: 3x3 double sort, return gap within volatility
     terciles, gap after removing coin x state x month means          -> trend_vs_vol.csv
  H  replication sample (ten further coins, specification frozen)     -> replication.csv
  Y  year and half-year clusters (Ibragimov-Mueller, excluding 2026) and a grouped
     jackknife over BTC trend episodes                                -> year_clusters.csv
  C  nested daytime windows 08-16 and 08-17 UTC                       -> nested_daytime.csv

Reads panels from data/panels/ (main and replication samples; part C also rebuilds the
08-16 UTC windows from the one-minute klines). Writes the files above to results/.
Supports Tables 3, 4, 5, A3, A8 and A12 and Figure 1.

The published numbers come from one run of all parts in the order above (the parts share
one random stream).

Usage: python src/scaling_checks.py [--reps 999] [--parts S,A,T,H,Y,C]
"""

import argparse
import math

import numpy as np
import pandas as pd
from scipy import stats

import build_windows as W
import config
import episodes as ep
import skew_by_state as sk

NB = 999
BOOT_SEED = 20261111  # seed of the bootstrap stream in skew_by_state
PARTS = ["S", "A", "T", "H", "Y", "C"]
DAYMS = 86_400_000
LMIN = {"1h": 60, "4h": 240, "15h": 900, "24h": 1440, "9h": 540}
g1 = sk.g1


def bow(x):
    return sk.qskew(x, 25)


def q10(x):
    return sk.qskew(x, 10)


MEAS = {"g1": g1, "bowley": bow, "q10": q10}


def load(h, sample="main"):
    p = pd.read_csv(config.panel_path(h, sample))
    p["g"] = ep.tri(p.snr45.to_numpy())
    return p


def exante_z(p, h):
    """Return scaled by the past 24-hour realized variance, rescaled to the window length."""
    return p.r / np.sqrt(np.exp(p.lrv1d) * LMIN[h] / 1440)


def gap_stat(fn):
    def f(R, G):
        r, g = ep.flat(R, G)
        return fn(r[g == 1]) - fn(r[g == -1])

    return f


def gap_ci(p, col, h, fn, nb=None):
    """Up-minus-down gap of fn(col) with bootstrap interval and p-value."""
    R, G = ep.mats(p.assign(v=p[col]), ["v", "g"])
    est = gap_stat(fn)(R, G)
    bs = ep.boot(gap_stat(fn), [R, G], sk.block_for(h), nb or NB)
    lo, hi = sk.ci(bs)
    r, g = ep.flat(R, G)
    return dict(up=fn(r[g == 1]), dn=fn(r[g == -1]), gap=est, lo=lo, hi=hi, p=sk.pval(bs))


def state_gap(d, fn=g1):
    return fn(d.r_std[d.g == 1].to_numpy()) - fn(d.r_std[d.g == -1].to_numpy())


def fmt_gap(d):
    return f"gap {d['gap']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}] p={d['p']:.3f}"


def part_S():
    rows = []
    for h in ["24h", "15h", "4h", "1h"]:
        p = load(h)
        p["z_lagbv"] = p.r / np.sqrt(np.exp(p.lbv_lag))
        p["z_exante"] = exante_z(p, h)
        p["z_raw"] = p.r / p.groupby("coin").r.transform("std")
        scalings = (
            ("BV (baseline)", "r_std"),
            ("lagged BV", "z_lagbv"),
            ("ex-ante", "z_exante"),
            ("unstandardized", "z_raw"),
        )
        for sc, col in scalings:
            q = p[np.isfinite(p[col])].reset_index(drop=True)
            for mn, fn in MEAS.items():
                # At one hour only the quantile measure is computed for the alternative scalings.
                if h == "1h" and mn != "q10" and sc != "BV (baseline)":
                    continue
                d = gap_ci(q, col, h, fn)
                rows.append(dict(h=h, scaling=sc, measure=mn, **d))
                print(f"S {h:>4} {sc:<15} {mn:<6} {fmt_gap(d)}", flush=True)
    pd.DataFrame(rows).to_csv(config.result_path("scalings.csv"), index=False)


def part_A():
    rows = []
    for h in ["24h", "15h", "4h"]:
        p = load(h).dropna(subset=["ac1"]).reset_index(drop=True)
        b = sk.block_for(h)
        for gv, nm in ((1, "up"), (0, "rg"), (-1, "dn")):
            d = p[p.g == gv]
            rows.append(dict(h=h, check=f"ac1_mean_{nm}", est=d.ac1.mean()))
            rows.append(dict(h=h, check=f"Ez_rpos_{nm}", est=d.r_std[d.r > 0].mean()))
            rows.append(dict(h=h, check=f"Ez_rneg_{nm}", est=d.r_std[d.r < 0].mean()))
            rows.append(dict(h=h, check=f"ac1_mean_{nm}_rpos", est=d.ac1[d.r > 0].mean()))
            rows.append(dict(h=h, check=f"ac1_mean_{nm}_rneg", est=d.ac1[d.r < 0].mean()))
        p["q"] = pd.qcut(p.ac1, 3, labels=False).astype(float)
        R, G, Q = ep.mats(p, ["r_std", "g", "q"])

        def within_terciles(R, G, Q):
            """Window-weighted average of the gap within autocorrelation terciles."""
            r, g, q = ep.flat(R, G, Q)
            t = w = 0.0
            for k in (0, 1, 2):
                s = q == k
                if (s & (g == 1)).sum() > 30 and (s & (g == -1)).sum() > 30:
                    t += s.sum() * (g1(r[s & (g == 1)]) - g1(r[s & (g == -1)]))
                    w += s.sum()
            return t / w

        est = within_terciles(R, G, Q)
        bs = ep.boot(within_terciles, [R, G, Q], b)
        lo, hi = sk.ci(bs)
        rows.append(dict(h=h, check="gap_within_ac1_terciles", est=est, lo=lo, hi=hi, p=sk.pval(bs)))
        for k in (0, 1, 2):
            rows.append(dict(h=h, check=f"gap_ac1_tercile{k + 1}", est=state_gap(p[p.q == k])))
        print(
            f"A {h}: ac1 up {p.ac1[p.g == 1].mean():+.4f} down {p.ac1[p.g == -1].mean():+.4f}; "
            f"gap within ac1 terciles {est:+.3f} [{lo:+.3f},{hi:+.3f}] p={sk.pval(bs):.3f}",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(config.result_path("smoothness.csv"), index=False)


def part_T():
    rows = []
    labels = ["low", "mid", "high"]
    for h in ["24h", "15h", "4h"]:
        p = load(h).dropna(subset=["ret45", "vol45"]).reset_index(drop=True)
        p["rt"] = pd.qcut(p.ret45, 3, labels=labels)
        p["vt"] = pd.qcut(p.vol45, 3, labels=labels)
        for (a, c), d in p.groupby(["rt", "vt"], observed=True):
            rows.append(
                dict(
                    h=h,
                    check="sort3x3",
                    ret_tercile=a,
                    vol_tercile=c,
                    n=len(d),
                    g1=g1(d.r_std.to_numpy()),
                    q10=q10(d.r_std.to_numpy()),
                )
            )
        # High-minus-low past-return tercile, within each past-volatility tercile.
        for c in labels:
            d = p[p.vt == c].assign(
                g=lambda x: np.where(x.rt == "high", 1.0, np.where(x.rt == "low", -1.0, 0.0))
            )
            res = gap_ci(d.reset_index(drop=True), "r_std", h, g1)
            rows.append(
                dict(
                    h=h,
                    check="ret_gap_within_vol",
                    vol_tercile=c,
                    g1=res["gap"],
                    lo=res["lo"],
                    hi=res["hi"],
                    p=res["p"],
                )
            )
            print(
                f"T {h} {c} volatility: high - low return gap {res['gap']:+.3f} p={res['p']:.3f}", flush=True
            )
        p["ym"] = pd.to_datetime(p.t0, unit="ms").dt.to_period("M").astype(str)
        p["zd"] = p.r_std - p.groupby(["coin", "g", "ym"]).r_std.transform("mean")
        res = gap_ci(p, "zd", h, g1)
        rows.append(
            dict(h=h, check="demean_coin_state_month", g1=res["gap"], lo=res["lo"], hi=res["hi"], p=res["p"])
        )
        print(f"T {h} demeaned by coin x state x month: gap {res['gap']:+.3f} p={res['p']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(config.result_path("trend_vs_vol.csv"), index=False)


def part_H():
    rows = []
    for h in ["24h", "15h", "4h", "1h"]:
        p = load(h, "replication")
        d = gap_ci(p, "r_std", h, g1)
        dq = gap_ci(p, "r_std", h, q10, 499)
        sign = [state_gap(x) > 0 for _, x in p.groupby("coin")]
        rows.append(
            dict(
                h=h,
                n=len(p),
                coins=p.coin.nunique(),
                **d,
                q10_gap=dq["gap"],
                q10_p=dq["p"],
                coins_pos=int(np.sum(sign)),
            )
        )
        print(
            f"H {h}: {len(p):,} windows, g1 {fmt_gap(d)}; Q10 gap {dq['gap']:+.3f} p={dq['p']:.3f}; "
            f"positive in {int(np.sum(sign))}/{len(sign)} coins",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(config.result_path("replication.csv"), index=False)


def im_test(gaps):
    """Ibragimov-Mueller t-test with clusters as observations."""
    t = gaps.mean() / (gaps.std(ddof=1) / math.sqrt(len(gaps)))
    return t, 2 * (1 - stats.t.cdf(abs(t), len(gaps) - 1))


def part_Y():
    rows = []
    rng = np.random.default_rng(7007)
    btc = load("24h")
    btc = btc[btc.coin == "BTC"].sort_values("t0")
    episodes = [e for e in ep.runs(btc.g.to_numpy(), btc.t0.to_numpy(), 1) if e[0] != 0]
    for h in ["24h", "15h", "4h"]:
        p = load(h)
        ts = pd.to_datetime(p.t0, unit="ms")
        clusterings = (
            ("year_ex2026", ts.dt.year.where(ts.dt.year < 2026)),
            ("halfyear", ts.dt.year.astype(str) + "H" + ((ts.dt.month > 6) + 1).astype(str)),
        )
        for lab, key in clusterings:
            gaps = []
            for _, d in p.assign(k=key).dropna(subset=["k"]).groupby("k"):
                if (d.g == 1).sum() > 30 and (d.g == -1).sum() > 30:
                    gaps.append(state_gap(d))
            gaps = np.array(gaps)
            tt, pv = im_test(gaps)
            rows.append(
                dict(
                    h=h,
                    check=f"IM_{lab}",
                    est=gaps.mean(),
                    t=tt,
                    p=pv,
                    n_clusters=len(gaps),
                    n_pos=int((gaps > 0).sum()),
                )
            )
            print(
                f"Y {h} IM {lab}: {len(gaps)} clusters, {int((gaps > 0).sum())} positive, t={tt:+.2f} p={pv:.3f}",
                flush=True,
            )
        # Grouped jackknife: each draw drops a random 20% of BTC trend episodes from all coins.
        jk = []
        for _ in range(200):
            drop = rng.choice(len(episodes), int(0.2 * len(episodes)), replace=False)
            keep = np.ones(len(p), bool)
            for k in drop:
                _, a, b, _ = episodes[k]
                keep &= ~((p.t0 >= a) & (p.t0 <= b + DAYMS)).to_numpy()
            jk.append(state_gap(p[keep]))
        jk = np.array(jk)
        # The t and p columns hold the 2.5% and 97.5% percentiles of the jackknife gaps.
        rows.append(
            dict(
                h=h,
                check="grouped_jackknife_20pct",
                est=float(np.median(jk)),
                t=float(np.percentile(jk, 2.5)),
                p=float(np.percentile(jk, 97.5)),
                n_clusters=200,
                n_pos=int((jk > 0).sum()),
            )
        )
        print(
            f"Y {h} grouped jackknife: median {np.median(jk):+.3f}, "
            f"95% range [{np.percentile(jk, 2.5):+.3f},{np.percentile(jk, 97.5):+.3f}], "
            f"positive {int((jk > 0).sum())}/200",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(config.result_path("year_clusters.csv"), index=False)


def part_C():
    rows = []
    parts = []
    for coin in config.MAIN_COINS:
        t0, lp, V, run = W.load(coin)
        parts.append(W.build(coin, "8h@08", (480, 480, 5, W.DAY), lp, V, run, t0))
    p8 = pd.concat(parts, ignore_index=True)
    p8["g"] = ep.tri(p8.snr45.to_numpy())
    p9 = load("9h")
    for lab, p in (("8h_08_16", p8), ("9h_08_17", p9)):
        for mn in ("g1", "q10"):
            d = gap_ci(p, "r_std", "24h", MEAS[mn])
            rows.append(dict(window=lab, measure=mn, **d))
            print(f"C {lab} {mn}: up {d['up']:+.3f} down {d['dn']:+.3f} {fmt_gap(d)}", flush=True)
    a = p8.assign(d=p8.t0 // DAYMS)[["coin", "d", "r_std", "g"]]
    b = p9.assign(d=p9.t0 // DAYMS)[["coin", "d", "r_std"]]
    both = a.merge(b, on=["coin", "d"], suffixes=("_8", "_9"))
    rows.append(
        dict(window="corr_z8_z9", measure="corr", gap=float(np.corrcoef(both.r_std_8, both.r_std_9)[0, 1]))
    )
    pd.DataFrame(rows).to_csv(config.result_path("nested_daytime.csv"), index=False)


def main():
    global NB
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=NB, help="bootstrap replications (default 999)")
    ap.add_argument("--parts", default=",".join(PARTS), help="comma-separated subset of S,A,T,H,Y,C")
    args = ap.parse_args()
    NB = args.reps
    ep.NB = NB
    sk.rng = np.random.default_rng(BOOT_SEED)
    run = {"S": part_S, "A": part_A, "T": part_T, "H": part_H, "Y": part_Y, "C": part_C}
    for k in args.parts.split(","):
        run[k]()


if __name__ == "__main__":
    main()
