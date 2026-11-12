"""Further inference checks on the uptrend-minus-downtrend skewness gap (pooled main panel).

    block45d_g1, block90d_g1   stationary bootstrap with expected block lengths of 45 and 90 days
    shift_placebo_g1           the state series of all coins is rotated by the same random shift
                               (at least 90 days from either end), which keeps the cross-section but
                               breaks the link between states and returns; p is the share of placebo
                               gaps at least as large in absolute value as the observed gap
    base_q10, exante_q10,      Q10/90 quantile-skewness gap under BV scaling, ex-ante scaling
    raw_q10                    (past-24h realized variance) and no scaling, 45-day blocks
    lookahead_t0+90d_g1        ex-post states from the 45 days that start 90 days after the
                               window, so that the future period never overlaps the window

Reads:  main-sample panels for 24h, 15h and 4h windows.
Writes: inference_checks.csv
Paper:  Table 5.

Usage:  python src/inference_checks.py [bootstrap and placebo replications, default 999]
"""

import argparse

import numpy as np
import pandas as pd

import config
import skew_by_state as sk

PER_DAY = {"24h": 1, "15h": 1, "4h": 6}
LMIN = {"4h": 240, "15h": 900, "24h": 1440}
DAYMS = 86_400_000


def tri(s):
    return np.where(s > 0.75, 1, np.where(s < -0.75, -1, 0)).astype(float)


def mats(p, val, grp):
    q = p.assign(v=val, g=grp)
    coins = sorted(q.coin.unique())
    R = q.pivot(index="t0", columns="coin", values="v")[coins].to_numpy()
    G = q.pivot(index="t0", columns="coin", values="g")[coins].to_numpy()
    return R, G


def diff_stat(R, G, fn):
    r, g = R.ravel(), G.ravel()
    ok = ~np.isnan(r) & ~np.isnan(g)
    r, g = r[ok], g[ok]
    return fn(r[g == 1]) - fn(r[g == -1])


def boot(R, G, fn, b, nb):
    T = R.shape[0]
    out = np.empty(nb)
    for c0 in range(0, nb, 50):
        idx = sk.sb_index(T, b, min(50, nb - c0))
        for j, ii in enumerate(idx):
            out[c0 + j] = diff_stat(R[ii], G[ii], fn)
    return out


def q10(x):
    return sk.qskew(x, 10)


def interval_row(h, check, est, bs):
    lo, hi = sk.ci(bs)
    return dict(h=h, check=check, est=est, lo=lo, hi=hi, p=sk.pval(bs))


def future_states(p, start_days):
    """States from snr45 of the window (start_days + 45) days ahead: SNR of the 45 days after start_days."""
    fut = np.full(len(p), np.nan)
    for _, d in p.groupby("coin"):
        s, t = d.snr45.to_numpy(), d.t0.to_numpy()
        i = np.searchsorted(t, t + (start_days + 45) * DAYMS)
        ok = i < len(t)
        v = np.full(len(d), np.nan)
        v[ok] = s[i[ok]]
        fut[d.index.to_numpy()] = v
    return tri(fut)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("reps", nargs="?", type=int, default=999, help="bootstrap replications")
    nb = ap.parse_args().reps
    rng = np.random.default_rng(2026)  # placebo shifts; bootstrap draws come from skew_by_state
    rows = []
    for h, per in PER_DAY.items():
        p = pd.read_csv(config.panel_path(h))
        g0 = tri(p.snr45.to_numpy())
        R, G = mats(p, p.r_std.to_numpy(), g0)
        obs = diff_stat(R, G, sk.g1)
        for days in (45, 90):
            rows.append(interval_row(h, f"block{days}d_g1", obs, boot(R, G, sk.g1, days * per, nb)))

        T = R.shape[0]
        placebo = []
        for _ in range(nb):
            k = int(rng.integers(90 * per, T - 90 * per))
            placebo.append(diff_stat(R, np.roll(G, k, axis=0), sk.g1))
        placebo = np.array(placebo)
        rows.append(
            dict(
                h=h,
                check="shift_placebo_g1",
                est=obs,
                lo=np.percentile(placebo, 2.5),
                hi=np.percentile(placebo, 97.5),
                p=float(np.mean(np.abs(placebo) >= abs(obs))),
                sd=placebo.std(),
            )
        )

        sd = p.groupby("coin").r.transform("std").to_numpy()
        exante = (p.r / np.sqrt(np.exp(p.lrv1d) * LMIN[h] / 1440)).to_numpy()
        scalings = (
            ("base_q10", p.r_std.to_numpy()),
            ("exante_q10", exante),
            ("raw_q10", p.r.to_numpy() / sd),
        )
        for name, v in scalings:
            Rv, Gv = mats(p, v, g0)
            rows.append(interval_row(h, name, diff_stat(Rv, Gv, q10), boot(Rv, Gv, q10, 45 * per, nb)))

        Rf, Gf = mats(p, p.r_std.to_numpy(), future_states(p, 90))
        est = diff_stat(Rf, Gf, sk.g1)
        # Missing states are coded 9 so that they fall in no state inside the bootstrap.
        bs = boot(Rf, np.nan_to_num(Gf, nan=9), sk.g1, 45 * per, nb)
        rows.append(interval_row(h, "lookahead_t0+90d_g1", est, bs))

        for r in rows[-7:]:
            print(
                f"{h:>4} {r['check']:<22} est {r['est']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}] p={r['p']:.3f}",
                flush=True,
            )
    pd.DataFrame(rows).to_csv(config.result_path("inference_checks.csv"), index=False)


if __name__ == "__main__":
    main()
