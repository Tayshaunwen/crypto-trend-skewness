"""Skewness of standardized returns, pooled and by trend state, with stationary-bootstrap inference.

Reads:  config.panel_path(h, "main") for h in 1h, 4h, 24h, 7d, 15h (build_windows.py)
Writes: pooled_skewness.csv, skew_by_state.csv
Supports: Tables 1, 2 and A4; Figure 1. Also provides the shared statistics and bootstrap used by
          most other modules (g1, qskew, sb_index, boot_regime, block_for, ci, pval, ...).

Usage:
    python src/skew_by_state.py [n_bootstrap=1999]

Definitions:
    z = r_std = r / sqrt(BV).
    g1: moment skewness (biased estimator, as scipy.stats.skew); bowley: 25/50/75 quantile skewness;
    q10: 10/50/90 quantile skewness; sv: demeaned downside over upside sum of squares.
    States from snr45: uptrend > +0.75, downtrend < -0.75, range otherwise (prices before T0 only).
    Inference: stationary bootstrap (Politis and Romano 1994) with an expected block of 45 days.
    The panel is resampled by whole time points, which keeps the cross-coin correlation.
    The seed is fixed per horizon (20261111 + horizon index). Coin-level up-minus-down p-values
    are Holm-adjusted within each horizon.
"""

import argparse

import numpy as np
import pandas as pd

import config

HS = ["1h", "4h", "24h", "7d", "15h"]
TH = 0.75
rng = np.random.default_rng(20261111)


def g1(x):
    m = x.mean()
    d = x - m
    s2 = (d * d).mean()
    return (d**3).mean() / s2**1.5


def qskew(x, a):
    lo, md, hi = np.percentile(x, [a, 50, 100 - a])
    return (hi + lo - 2 * md) / (hi - lo)


def sv(x):
    d = x - x.mean()
    return np.sum(d[d < 0] ** 2) / np.sum(d[d > 0] ** 2)


def stat_all(x):
    return dict(n=len(x), mean=x.mean(), g1=g1(x), bowley=qskew(x, 25), q10=qskew(x, 10), sv=sv(x))


def sb_index(T, b, nb):
    """Stationary-bootstrap index matrix [nb, T]; block lengths are geometric with mean b."""
    p = 1.0 / b
    start = rng.integers(0, T, (nb, T))
    newb = rng.random((nb, T)) < p
    newb[:, 0] = True
    t = np.arange(T)
    pos = np.maximum.accumulate(np.where(newb, t, 0), axis=1)  # start position of the current block
    st = np.take_along_axis(start, pos, axis=1)
    return (st + (t - pos)) % T


# A 45-day expected block covers the persistence of the 45-day state variable itself.
BLOCK_DAYS = 45
PER_DAY = {"1h": 24, "4h": 6, "9h": 1, "15h": 1, "24h": 1, "7d": 1 / 7}


def block_for(h):
    """Expected block length in windows: 45 days at horizon h (at least 2)."""
    return max(2.0, BLOCK_DAYS * PER_DAY[h])


def regime(s):
    return np.where(s > TH, 1, np.where(s < -TH, -1, 0))


def boot_regime(R, G, b, nb):
    """Bootstrap distributions of g1 overall, by state, and of the up-minus-down gap.

    R, G: [T, K] arrays of r_std and state (K coins, NaN for missing).
    """
    T = R.shape[0]
    out = {k: np.empty(nb) for k in ("all", "up", "rg", "dn", "diff")}
    chunk = 100
    for c0 in range(0, nb, chunk):
        idx = sb_index(T, b, min(chunk, nb - c0))
        for j, ii in enumerate(idx):
            r = R[ii].ravel()
            g = G[ii].ravel()
            ok = ~np.isnan(r)
            r, g = r[ok], g[ok]
            u, d, m = g1(r[g == 1]), g1(r[g == -1]), g1(r[g == 0])
            k = c0 + j
            out["all"][k] = g1(r)
            out["up"][k] = u
            out["dn"][k] = d
            out["rg"][k] = m
            out["diff"][k] = u - d
    return out


def ci(a):
    return np.percentile(a, 2.5), np.percentile(a, 97.5)


def pval(a):
    """Two-sided p-value: twice the share of the bootstrap distribution on the far side of zero."""
    return min(1.0, 2 * min(np.mean(a <= 0), np.mean(a >= 0)))


def holm(p):
    p = np.asarray(p)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1, run)
    return adj


def analyse_horizon(h, nb, pooled_rows, state_rows):
    global rng
    p = pd.read_csv(config.panel_path(h), usecols=["coin", "t0", "r_std", "snr45"])
    p["g"] = regime(p.snr45.to_numpy())
    coins = list(p.coin.unique())
    R = p.pivot(index="t0", columns="coin", values="r_std")[coins]
    G = p.pivot(index="t0", columns="coin", values="g")[coins]
    b = block_for(h)
    print(f"\n{h}: {len(p):,} windows, block {b:.1f}, {nb} replications", flush=True)
    # A fixed seed per horizon keeps each horizon's results independent of the others.
    rng = np.random.default_rng(20261111 + HS.index(h))
    units = [(c, R[[c]].to_numpy(), G[[c]].to_numpy()) for c in coins] + [
        ("PANEL", R.to_numpy(), G.to_numpy())
    ]
    coin_rows, coin_p = [], []
    for name, Rm, Gm in units:
        ok = ~np.isnan(Rm.ravel())
        r = Rm.ravel()[ok]
        g = Gm.ravel()[ok]
        bs = boot_regime(Rm, Gm, b, nb)
        sa = stat_all(r)
        lo, hi = ci(bs["all"])
        pooled_rows.append(dict(h=h, coin=name, **sa, g1_lo=lo, g1_hi=hi, block=b))
        row = dict(h=h, coin=name)
        for key, gv in (("up", 1), ("rg", 0), ("dn", -1)):
            s = stat_all(r[g == gv])
            l2, h2 = ci(bs[key])
            row.update(
                {
                    f"{key}_n": s["n"],
                    f"{key}_g1": s["g1"],
                    f"{key}_lo": l2,
                    f"{key}_hi": h2,
                    f"{key}_bowley": s["bowley"],
                    f"{key}_q10": s["q10"],
                    f"{key}_sv": s["sv"],
                    f"{key}_mean": s["mean"],
                }
            )
        d = row["up_g1"] - row["dn_g1"]
        l3, h3 = ci(bs["diff"])
        row.update(diff=d, diff_lo=l3, diff_hi=h3, diff_p=pval(bs["diff"]))
        state_rows.append(row)
        if name != "PANEL":
            coin_rows.append(row)
            coin_p.append(row["diff_p"])
        print(
            f"  {name:<6} all {sa['g1']:+.3f} | up {row['up_g1']:+.3f} range {row['rg_g1']:+.3f} "
            f"down {row['dn_g1']:+.3f} | gap {d:+.3f} [{l3:+.2f}, {h3:+.2f}] p={row['diff_p']:.3f}",
            flush=True,
        )
    adj = holm(coin_p)
    for row, a in zip(coin_rows, adj):
        row["diff_p_holm"] = a
    print(
        f"  coins with a significant gap at 5%: {sum(adj < 0.05)}/{len(adj)} Holm-adjusted, "
        f"{sum(np.array(coin_p) < 0.05)}/{len(coin_p)} unadjusted",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("n_bootstrap", nargs="?", type=int, default=1999)
    nb = ap.parse_args().n_bootstrap
    pooled_rows, state_rows = [], []
    for h in HS:
        analyse_horizon(h, nb, pooled_rows, state_rows)
    pd.DataFrame(pooled_rows).to_csv(config.result_path("pooled_skewness.csv"), index=False)
    pd.DataFrame(state_rows).to_csv(config.result_path("skew_by_state.csv"), index=False)


if __name__ == "__main__":
    main()
