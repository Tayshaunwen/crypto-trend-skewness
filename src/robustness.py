"""Robustness of the uptrend-minus-downtrend skewness gap to subsamples, scalings and state rules.

Pooled panel of the ten main coins; statistic = moment skewness g1 in uptrend and downtrend
windows and their difference, with stationary block-bootstrap intervals. Variants:

    base        main specification (snr45 three states, threshold 0.75, z = r / sqrt(BV))
    from2023    2023 onwards only
    ex2021      without 2021
    std_rv      scaled by realized variance instead of bipower variation
    std_exante  scaled by ex-ante volatility: past-24h realized variance times window length
    raw         unscaled log returns, divided by each coin's full-sample standard deviation
    adaptive    thresholds at each coin's expanding 25% / 75% quantiles of snr45, lagged once
    majors      BTC and ETH only
    alts        the other eight coins
    lookahead   ex-post states from the signal-to-noise ratio of the 45 days after the window
                start (uses future information; a contrast, not a test)

Reads:  main-sample panels for 24h, 15h and 4h windows.
Writes: robustness_subsamples.csv
Paper:  Table 5.

Usage:  python src/robustness.py [bootstrap replications, default 999]
"""

import argparse

import numpy as np
import pandas as pd

import config
import skew_by_state as sk

HS = ["24h", "15h", "4h"]
LMIN = {"4h": 240, "15h": 900, "24h": 1440}
TH = 0.75
DAYMS = 86_400_000


def tri(s):
    return np.where(s > TH, 1, np.where(s < -TH, -1, 0)).astype(float)


def future_snr(p):
    """Per coin: snr45 of the window 45 days after t0, i.e. the SNR of the 45 days after the start."""
    out = np.full(len(p), np.nan)
    for _, d in p.groupby("coin"):
        s = pd.Series(d.snr45.to_numpy(), index=d.t0.to_numpy())
        idx = np.searchsorted(s.index.to_numpy(), d.t0.to_numpy() + 45 * DAYMS)
        ok = idx < len(s)
        v = np.full(len(d), np.nan)
        v[ok] = s.to_numpy()[idx[ok]]
        out[d.index.to_numpy()] = v
    return out


def adaptive(p):
    """States from each coin's expanding quantiles of snr45 (lagged one window, no look-ahead)."""
    g = np.full(len(p), np.nan)
    for _, d in p.groupby("coin"):
        s = d.snr45
        hi = s.expanding(min_periods=60).quantile(0.75).shift(1)
        lo = s.expanding(min_periods=60).quantile(0.25).shift(1)
        lab = np.where(s > hi, 1, np.where(s < lo, -1, 0)).astype(float)
        lab[hi.isna().to_numpy()] = np.nan
        g[d.index.to_numpy()] = lab
    return g


def evaluate(p, val, grp, b, nb):
    q = p.assign(v=val, g=grp).dropna(subset=["v", "g"])
    coins = list(q.coin.unique())
    R = q.pivot(index="t0", columns="coin", values="v")[coins].to_numpy()
    G = q.pivot(index="t0", columns="coin", values="g")[coins].to_numpy()
    r, g = R.ravel(), G.ravel()
    ok = ~np.isnan(r) & ~np.isnan(g)
    r, g = r[ok], g[ok]
    # Missing states are coded 9 so that they fall in no state inside the bootstrap.
    bs = sk.boot_regime(R, np.nan_to_num(G, nan=9), b, nb)
    up, dn = sk.g1(r[g == 1]), sk.g1(r[g == -1])
    lo, hi = sk.ci(bs["diff"])
    return dict(
        n=len(r),
        up_g1=up,
        dn_g1=dn,
        diff=up - dn,
        diff_lo=lo,
        diff_hi=hi,
        diff_p=sk.pval(bs["diff"]),
        up_q10=sk.qskew(r[g == 1], 10),
        dn_q10=sk.qskew(r[g == -1], 10),
    )


def variants(p, h):
    base_g = tri(p.snr45.to_numpy())
    sd = p.groupby("coin").r.transform("std")
    exante = p.r / np.sqrt(np.exp(p.lrv1d) * LMIN[h] / 1440)
    return {  # name: (values, states, row mask or None)
        "base": (p.r_std, base_g, None),
        "from2023": (p.r_std, base_g, p.year >= 2023),
        "ex2021": (p.r_std, base_g, p.year != 2021),
        "std_rv": (p.r / np.sqrt(p.rv), base_g, None),
        "std_exante": (exante, base_g, None),
        "raw": (p.r / sd, base_g, None),
        "adaptive": (p.r_std, adaptive(p), None),
        "majors": (p.r_std, base_g, p.coin.isin(["BTC", "ETH"])),
        "alts": (p.r_std, base_g, ~p.coin.isin(["BTC", "ETH"])),
        "lookahead": (p.r_std, tri(future_snr(p)), None),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("reps", nargs="?", type=int, default=999, help="bootstrap replications")
    nb = ap.parse_args().reps
    rows = []
    for h in HS:
        p = pd.read_csv(config.panel_path(h))
        b = sk.block_for(h)
        for name, (val, grp, mask) in variants(p, h).items():
            pp = p if mask is None else p[mask.to_numpy()]
            keep = pp.index.to_numpy()
            res = evaluate(pp.reset_index(drop=True), np.asarray(val)[keep], np.asarray(grp)[keep], b, nb)
            rows.append(dict(h=h, variant=name, **res))
            print(
                f"{h:>4} {name:<11} n={res['n']:>7,} up {res['up_g1']:+.3f} down {res['dn_g1']:+.3f} "
                f"gap {res['diff']:+.3f} [{res['diff_lo']:+.2f},{res['diff_hi']:+.2f}] "
                f"p={res['diff_p']:.3f} "
                f"| Q10 up {res['up_q10']:+.3f} down {res['dn_q10']:+.3f}",
                flush=True,
            )
    pd.DataFrame(rows).to_csv(config.result_path("robustness_subsamples.csv"), index=False)


if __name__ == "__main__":
    main()
