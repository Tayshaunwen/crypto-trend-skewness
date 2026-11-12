"""Skewness by trend state in intraday windows: 08:00-17:00 UTC against 15h (17:00-08:00) and 24h.

For each horizon: moment skewness g1 of z pooled and by state, and the uptrend-minus-downtrend
gap with a stationary block bootstrap (45-day blocks).

Reads:  main-sample panels for 9h, 15h and 24h windows (and, only to size the random-number
        replay described in replay_earlier_draws, the 1h, 4h, 15h and 24h panels).
Writes: time_of_day.csv (rows C_gap, C_pooled_g1, C_g1_up, C_g1_rg, C_g1_dn)
Paper:  online Table A12.

Usage:  python src/time_of_day.py [bootstrap replications, default 999]
"""

import argparse

import numpy as np
import pandas as pd

import config
import skew_by_state as sk

PER_DAY = {"1h": 24, "4h": 6, "9h": 1, "15h": 1, "24h": 1}
CHUNK = 50


def tri(s):
    return np.where(s > 0.75, 1, np.where(s < -0.75, -1, 0)).astype(float)


def mats(p, cols):
    coins = sorted(p.coin.unique())
    return [p.pivot(index="t0", columns="coin", values=c)[coins].to_numpy() for c in cols]


def boot(fn, arrs, per, nb):
    T = arrs[0].shape[0]
    out = np.empty(nb)
    for c0 in range(0, nb, CHUNK):
        idx = sk.sb_index(T, 45 * per, min(CHUNK, nb - c0))
        for j, ii in enumerate(idx):
            out[c0 + j] = fn(*[a[ii] for a in arrs])
    return out


def flat(*arrs):
    v = [a.ravel() for a in arrs]
    ok = np.all([~np.isnan(x) for x in v], axis=0)
    return [x[ok] for x in v]


def gap(R, G):
    r, g = flat(R, G)
    return sk.g1(r[g == 1]) - sk.g1(r[g == -1])


def replay_earlier_draws(nb):
    """Advance the shared bootstrap generator past the draws of checks that preceded this one.

    The published intervals were computed in a run in which ten other bootstrap checks (three each
    on the 15h, 24h and 4h panels, then one on the 1h panel) drew from the same generator first.
    Drawing the same index matrices again reproduces those intervals exactly; the draws depend
    only on the number of time points and the replication count.
    """
    for h, n_checks in (("15h", 3), ("24h", 3), ("4h", 3), ("1h", 1)):
        p = pd.read_csv(config.panel_path(h), usecols=["t0", "zfrac"])
        if h == "1h":
            p = p[p.zfrac <= 0.30]
        T = p.t0.nunique()
        for _ in range(n_checks):
            for c0 in range(0, nb, CHUNK):
                sk.sb_index(T, 45 * PER_DAY[h], min(CHUNK, nb - c0))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("reps", nargs="?", type=int, default=999, help="bootstrap replications")
    nb = ap.parse_args().reps
    replay_earlier_draws(nb)
    rows = []
    for h in ["9h", "15h", "24h"]:
        p = pd.read_csv(config.panel_path(h))
        p["g"] = tri(p.snr45.to_numpy())
        r, g = p.r_std.to_numpy(), p.g.to_numpy()
        R, G = mats(p, ["r_std", "g"])
        bs = boot(gap, [R, G], PER_DAY[h], nb)
        est = gap(R, G)
        lo, hi = sk.ci(bs)
        rows.append(dict(h=h, check="C_gap", est=est, lo=lo, hi=hi, p=sk.pval(bs)))
        rows.append(dict(h=h, check="C_pooled_g1", est=sk.g1(r)))
        for gv, nm in ((1, "up"), (0, "rg"), (-1, "dn")):
            rows.append(dict(h=h, check=f"C_g1_{nm}", est=sk.g1(r[g == gv])))
        print(
            f"{h:>4} pooled g1 {sk.g1(r):+.3f}; up {sk.g1(r[g == 1]):+.3f} range {sk.g1(r[g == 0]):+.3f} "
            f"down {sk.g1(r[g == -1]):+.3f}; gap {est:+.4f} [{lo:+.4f},{hi:+.4f}] p={rows[-5]['p']:.3f}",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(config.result_path("time_of_day.csv"), index=False)


if __name__ == "__main__":
    main()
