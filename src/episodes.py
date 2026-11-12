"""Descriptive statistics, trend episodes and the time variation of the skewness gap.

Main specification throughout: z = r / sqrt(BV), states from snr45 (threshold 0.75), stationary
block bootstrap with 45-day blocks. Also provides the panel helpers (load, mats, flat, gap_fn,
boot, runs, hill) used by the mechanism and later checks.

Reads:  main-sample panels for all horizons.
Writes:
    describe_by_coin.csv     per coin and horizon: n, mean, sd, skewness, kurtosis of z; state shares
    state_spells.csv         per coin (24h): number and mean length in days of up / down / range spells
    state_transitions.csv    24h state transition matrix, pooled over coins (rows = today)
    episode_jackknife.csv    market trend episodes from BTC 24h states; gap with each episode left out
    gap_by_year.csv          panel gap per year with bootstrap interval, plus a t test with years as
                             clusters (Ibragimov and Mueller, 2010; row year = "IM_test")
    snr_deciles.csv          g1 and Q10/90 skewness by decile of snr45, with bootstrap intervals
    rolling_gap.csv          gap over rolling 12-month windows
Paper:  Table 1, online Table A1; Figures 2, 3 and 4; numbers quoted in the text.

Usage:  python src/episodes.py   (fixed replication counts: 499 per year, 299 per decile)
"""

import math

import numpy as np
import pandas as pd
from scipy import stats

import config
import skew_by_state as sk

NB = 999  # default replications for boot(); importing modules may override it
HS = ["1h", "4h", "15h", "24h", "7d"]
H3 = ["24h", "15h", "4h"]
DAYMS = 86_400_000


def tri(s):
    return np.where(s > 0.75, 1, np.where(s < -0.75, -1, 0)).astype(float)


def load(h, sample="main"):
    p = pd.read_csv(config.panel_path(h, sample))
    p["g"] = tri(p.snr45.to_numpy())
    return p


def mats(p, cols):
    """Pivot columns of a panel into [time, coin] matrices (coins sorted)."""
    coins = sorted(p.coin.unique())
    return [p.pivot(index="t0", columns="coin", values=c)[coins].to_numpy() for c in cols]


def flat(*a):
    """Flatten matrices and keep the cells that are non-missing in all of them."""
    v = [x.ravel() for x in a]
    ok = np.all([~np.isnan(x) for x in v], axis=0)
    return [x[ok] for x in v]


def gap_fn(fn):
    """Statistic fn(uptrend) - fn(downtrend) on a pair of return and state matrices."""

    def f(R, G):
        r, g = flat(R, G)
        return fn(r[g == 1]) - fn(r[g == -1])

    return f


def boot(stat, arrs, b, nb=None):
    """Stationary bootstrap of stat(*arrs), resampling whole time rows of all matrices together."""
    nb = nb or NB
    T = arrs[0].shape[0]
    out = np.empty(nb)
    for c0 in range(0, nb, 50):
        idx = sk.sb_index(T, b, min(50, nb - c0))
        for j, ii in enumerate(idx):
            out[c0 + j] = stat(*[a[ii] for a in arrs])
    return out


def hill(x, frac=0.02):
    """Hill tail index from the largest frac of the positive values (at least 10 order statistics)."""
    x = np.sort(x[x > 0])[::-1]
    k = max(10, int(len(x) * frac))
    return 1.0 / np.mean(np.log(x[:k] / x[k]))


def runs(states, t0, day):
    """Spells of a constant state: list of (state, first t0, last t0, length in days)."""
    out = []
    s0 = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[s0]:
            out.append((states[s0], t0[s0], t0[i - 1], (t0[i - 1] - t0[s0]) / DAYMS + day))
            s0 = i
    return out


def describe():
    desc, spells = [], []
    for h in HS:
        p = load(h)
        for c, d in p.groupby("coin"):
            z = d.r_std.to_numpy()
            desc.append(
                dict(
                    h=h,
                    coin=c,
                    n=len(z),
                    start=pd.to_datetime(d.t0.min(), unit="ms").date(),
                    mean=z.mean(),
                    sd=z.std(ddof=1),
                    skew=sk.g1(z),
                    kurt=stats.kurtosis(z, fisher=False),
                    up=np.mean(d.g == 1),
                    rg=np.mean(d.g == 0),
                    dn=np.mean(d.g == -1),
                )
            )
            if h == "24h":
                rr = runs(d.g.to_numpy(), d.t0.to_numpy(), 1)
                for gv, nm in ((1, "up"), (-1, "dn"), (0, "rg")):
                    L = [x[3] for x in rr if x[0] == gv]
                    spells.append(
                        dict(coin=c, state=nm, n_spells=len(L), mean_days=np.mean(L) if L else np.nan)
                    )
        if h == "24h":
            tr = []
            for _, d in p.groupby("coin"):
                g = d.g.to_numpy()
                tr.append(pd.crosstab(g[:-1], g[1:]))
            T = sum(tr)
            T = T.div(T.sum(1), axis=0)
            T.to_csv(config.result_path("state_transitions.csv"))
            print("24h state transitions (rows = today, columns = next day)\n", T.round(3))
    pd.DataFrame(desc).to_csv(config.result_path("describe_by_coin.csv"), index=False)
    S = pd.DataFrame(spells)
    S.to_csv(config.result_path("state_spells.csv"), index=False)
    print(S.groupby("state")[["n_spells", "mean_days"]].agg(["sum", "mean"]).round(1))


def episode_jackknife():
    """Market trend episodes = non-range spells of BTC's 24h state; drop one episode at a time."""
    btc = load("24h")
    btc = btc[btc.coin == "BTC"].sort_values("t0")
    eps = [e for e in runs(btc.g.to_numpy(), btc.t0.to_numpy(), 1) if e[0] != 0]
    n_up, n_dn = sum(e[0] == 1 for e in eps), sum(e[0] == -1 for e in eps)
    print(f"BTC 24h trend episodes: {n_up} up, {n_dn} down")
    rows = []
    for h in H3:
        p = load(h)
        full = gap_fn(sk.g1)(*mats(p, ["r_std", "g"]))
        jk = []
        for _, a, b, _ in eps:
            q = p[(p.t0 < a) | (p.t0 > b + DAYMS)]
            jk.append(gap_fn(sk.g1)(*mats(q, ["r_std", "g"])))
        jk = np.array(jk)
        n_below = int((jk < full / 2).sum())
        rows.append(
            dict(
                h=h,
                check="jackknife",
                full=full,
                n_episodes=len(eps),
                n_up_ep=n_up,
                n_dn_ep=n_dn,
                jk_min=jk.min(),
                jk_median=np.median(jk),
                jk_max=jk.max(),
                n_below_half=n_below,
            )
        )
        print(
            f"{h} leave-one-episode-out: full gap {full:+.3f}, range [{jk.min():+.3f}, {jk.max():+.3f}], "
            f"below half in {n_below}/{len(eps)}"
        )
    pd.DataFrame(rows).to_csv(config.result_path("episode_jackknife.csv"), index=False)


def gap_by_year():
    rows = []
    for h in H3:
        p = load(h)
        b = sk.block_for(h)
        for y, d in p.groupby("year"):
            R, G = mats(d, ["r_std", "g"])
            r, g = flat(R, G)
            if (g == 1).sum() < 30 or (g == -1).sum() < 30:
                continue
            e = gap_fn(sk.g1)(R, G)
            bs = boot(gap_fn(sk.g1), [R, G], b, 499)
            lo, hi = sk.ci(bs)
            rows.append(
                dict(h=h, year=y, gap=e, lo=lo, hi=hi, n_up=int((g == 1).sum()), n_dn=int((g == -1).sum()))
            )
        # Ibragimov-Mueller: t test on the yearly gaps. In this row lo = t statistic, hi = p value,
        # n_up = number of years and n_dn = number of positive years.
        d = pd.DataFrame([x for x in rows if x["h"] == h])
        t = d.gap.mean() / (d.gap.std(ddof=1) / math.sqrt(len(d)))
        p_im = 2 * (1 - stats.t.cdf(abs(t), len(d) - 1))
        rows.append(
            dict(
                h=h, year="IM_test", gap=d.gap.mean(), lo=t, hi=p_im, n_up=len(d), n_dn=int((d.gap > 0).sum())
            )
        )
        print(
            f"{h} yearly gaps: {', '.join(f'{int(a)}:{v:+.2f}' for a, v in zip(d.year, d.gap))}; "
            f"IM t={t:+.2f} p={p_im:.3f}, positive {int((d.gap > 0).sum())}/{len(d)}"
        )
    pd.DataFrame(rows).to_csv(config.result_path("gap_by_year.csv"), index=False)


def snr_deciles():
    def q10(x):
        return sk.qskew(x, 10)

    def g1_valid(V):
        return sk.g1(V[~np.isnan(V)])

    def q10_valid(V):
        return q10(V[~np.isnan(V)])

    rows = []
    for h in H3:
        p = load(h)
        p["dec"] = pd.qcut(p.snr45, 10, labels=False)
        b = sk.block_for(h)
        for k in range(10):
            d = p.assign(v=np.where(p.dec == k, p.r_std, np.nan))
            V = mats(d, ["v"])[0]
            b1 = boot(g1_valid, [V], b, 299)
            b2 = boot(q10_valid, [V], b, 299)
            z = d.v.dropna().to_numpy()
            rows.append(
                dict(
                    h=h,
                    decile=k + 1,
                    snr_mid=p.snr45[p.dec == k].median(),
                    g1=sk.g1(z),
                    g1_lo=np.percentile(b1, 2.5),
                    g1_hi=np.percentile(b1, 97.5),
                    q10=q10(z),
                    q10_lo=np.percentile(b2, 2.5),
                    q10_hi=np.percentile(b2, 97.5),
                )
            )
        print(f"{h} g1 by decile:", " ".join(f"{x['g1']:+.2f}" for x in rows if x["h"] == h))
    pd.DataFrame(rows).to_csv(config.result_path("snr_deciles.csv"), index=False)


def rolling_gap():
    rows = []
    for h in ["24h", "15h"]:
        p = load(h)
        ts = pd.to_datetime(p.t0, unit="ms")
        for end in pd.date_range("2018-09-01", ts.max(), freq="MS"):
            d = p[(ts >= end - pd.DateOffset(months=12)) & (ts < end)]
            if (d.g == 1).sum() > 100 and (d.g == -1).sum() > 100:
                rows.append(
                    dict(
                        h=h,
                        end=end.date(),
                        gap=gap_fn(sk.g1)(*mats(d, ["r_std", "g"])),
                        gap_q10=gap_fn(lambda x: sk.qskew(x, 10))(*mats(d, ["r_std", "g"])),
                    )
                )
    pd.DataFrame(rows).to_csv(config.result_path("rolling_gap.csv"), index=False)


def main():
    # Order matters: the bootstrap draws of gap_by_year and snr_deciles share one generator.
    describe()
    episode_jackknife()
    gap_by_year()
    snr_deciles()
    rolling_gap()


if __name__ == "__main__":
    main()
