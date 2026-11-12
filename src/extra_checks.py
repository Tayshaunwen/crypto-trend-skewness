"""Further checks: pooled 20-coin panel, year clusters, tails, start hour, point-in-time universe,
replication of the primary statistic and the joint test over fifteen statistics.

Trend states are the three snr45 states (threshold 0.75); inference uses the stationary
bootstrap with 45-day blocks of dates.

Parts
  H2   correlation of main and replication states; gaps on the pooled 20-coin panel
                                                                   -> pooled_20coins.csv
  I    Ibragimov-Mueller test with calendar years (2026 included) -> year_clusters_all.csv
  T    Hill tail indices (top 2%) and the gap in skewness winsorized at 0.5% / 99.5%,
       for the four scalings of scaling_checks                    -> tails.csv
  P    24-hour windows starting at 00, 04, ..., 20 UTC (rebuilt from klines) -> start_hour.csv
  U    point-in-time universe: each month the top 10 (and 20) of 41 candidate USDT pairs
       by trailing 30-day quote volume, delisted coins included   -> point_in_time.csv
  H3   primary statistic (Q10/90 gap of past-day-RV-scaled returns) on the replication
       sample                                                     -> replication_primary.csv
  J15  joint test: 5 horizons x {g1 of z, Q10/90 of z, Q10/90 of past-day-RV-scaled
       returns}, all resampled with the same blocks of dates; studentized, unadjusted
       and Romano-Wolf step-down p-values                         -> joint_test.csv

Reads panels from data/panels/ (main, replication and point_in_time samples) and, for P and
U, the one-minute klines. Writes the files above to results/.
Supports Tables 3, 5, A3, A5 and A8.

Random draws: H2, I, T and P share one random stream; U, H3 and J15 each start a fresh
stream from the bootstrap seed. This is how the published numbers were produced.
J15 uses 1,999 replications, the other parts 999.

Usage: python src/extra_checks.py [--reps 999] [--joint-reps 1999] [--parts H2,I,T,P,U,H3,J15]
"""

import argparse

import numpy as np
import pandas as pd

import build_windows as W
import config
import episodes as ep
import scaling_checks as sc
import skew_by_state as sk

NB = 999
NB_JOINT = 1999
DAYMS = 86_400_000
LMIN = {"1h": 60, "4h": 240, "15h": 900, "24h": 1440, "7d": 10080}
# Parts that share one random stream; each group starts from a fresh seed.
GROUPS = [["H2", "I", "T", "P"], ["U"], ["H3"], ["J15"]]
g1 = sk.g1
q10 = sc.q10
load = sc.load


def winsorized_g1(x, a=0.005):
    lo, hi = np.percentile(x, [100 * a, 100 * (1 - a)])
    return sk.g1(np.clip(x, lo, hi))


def exante_z(p, h):
    return p.r / np.sqrt(np.exp(p.lrv1d) * LMIN[h] / 1440)


def part_H2():
    rows = []
    for h in ["24h", "15h", "4h", "1h"]:
        a, b = load(h), load(h, "replication")
        mean_a = a.groupby("t0").g.mean()
        mean_b = b.groupby("t0").g.mean()
        joined = pd.concat([mean_a, mean_b], axis=1, join="inner").dropna()
        corr = float(np.corrcoef(joined.iloc[:, 0], joined.iloc[:, 1])[0, 1])
        both = pd.concat([a, b], ignore_index=True)
        d = sc.gap_ci(both, "r_std", h, g1)
        dq = sc.gap_ci(both, "r_std", h, q10)
        rows.append(
            dict(
                h=h,
                state_corr=corr,
                comb_g1=d["gap"],
                comb_g1_p=d["p"],
                comb_q10=dq["gap"],
                comb_q10_p=dq["p"],
                n=len(both),
            )
        )
        print(
            f"H2 {h}: state corr {corr:.2f}; pooled g1 gap {d['gap']:+.3f} p={d['p']:.3f}, "
            f"Q10 gap {dq['gap']:+.3f} p={dq['p']:.3f}",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(config.result_path("pooled_20coins.csv"), index=False)


def part_I():
    rows = []
    for h in ["24h", "15h", "4h"]:
        p = load(h)
        gaps = []
        for _, d in p.groupby("year"):
            if (d.g == 1).sum() > 30 and (d.g == -1).sum() > 30:
                gaps.append(sc.state_gap(d))
        gaps = np.array(gaps)
        t, pv = sc.im_test(gaps)
        rows.append(dict(h=h, n_years=len(gaps), n_pos=int((gaps > 0).sum()), t=t, p=pv))
        print(
            f"I {h}: {int((gaps > 0).sum())} of {len(gaps)} years positive, IM t={t:+.2f} p={pv:.3f}",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(config.result_path("year_clusters_all.csv"), index=False)


def part_T():
    rows = []
    for h in ["24h", "15h", "4h"]:
        p = load(h)
        scalings = {
            "BV (baseline)": p.r_std,
            "lagged BV": p.r / np.sqrt(np.exp(p.lbv_lag)),
            "ex-ante": exante_z(p, h),
            "unstandardized": p.r / p.groupby("coin").r.transform("std"),
        }
        for name, z in scalings.items():
            q = p.assign(zz=z)[np.isfinite(z)].reset_index(drop=True)
            zz = q.zz.to_numpy()
            hill_up, hill_lo = ep.hill(zz), ep.hill(-zz)
            d = sc.gap_ci(q, "zz", h, winsorized_g1)
            rows.append(
                dict(
                    h=h,
                    scaling=name,
                    hill_upper=hill_up,
                    hill_lower=hill_lo,
                    wins_gap=d["gap"],
                    wins_lo=d["lo"],
                    wins_hi=d["hi"],
                    wins_p=d["p"],
                )
            )
            print(
                f"T {h} {name:<15} Hill up {hill_up:.1f} low {hill_lo:.1f}; "
                f"winsorized g1 gap {d['gap']:+.3f} p={d['p']:.3f}",
                flush=True,
            )
    pd.DataFrame(rows).to_csv(config.result_path("tails.csv"), index=False)


def part_P():
    rows = []
    parts = {s: [] for s in (0, 4, 8, 12, 16, 20)}
    for coin in config.MAIN_COINS:
        t0, lp, V, run = W.load(coin)
        for s in parts:
            parts[s].append(W.build(coin, f"24h@{s:02d}", (1440, s * 60, 5, W.DAY), lp, V, run, t0))
    for s, ps in parts.items():
        p = pd.concat(ps, ignore_index=True)
        p["g"] = ep.tri(p.snr45.to_numpy())
        d = sc.gap_ci(p, "r_std", "24h", g1)
        dq = sc.gap_ci(p, "r_std", "24h", q10)
        rows.append(dict(start=s, g1_gap=d["gap"], g1_p=d["p"], q10_gap=dq["gap"], q10_p=dq["p"]))
        print(
            f"P start {s:02d}:00 UTC: g1 gap {d['gap']:+.3f} p={d['p']:.3f}; "
            f"Q10 gap {dq['gap']:+.3f} p={dq['p']:.3f}",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(config.result_path("start_hour.csv"), index=False)


def monthly_volume():
    """Trailing 30-day quote volume of each candidate coin on the first day of each month."""
    vol = {}
    for c in sorted(config.UNIVERSE_COINS):
        z = np.load(config.KLINE_DIR / f"{c}_1m.npz")
        ts, qv = z["ts"], z["v"] * z["c"]
        daily = pd.Series(qv, index=pd.to_datetime(ts, unit="ms")).resample("D").sum()
        vol[c] = daily.rolling(30, min_periods=20).sum().resample("MS").first()
    return pd.DataFrame(vol)


def part_U():
    volume = monthly_volume()
    rows = []
    for topn in (10, 20):
        member = volume.rank(axis=1, ascending=False) <= topn
        print(f"U top {topn}: {int(member.any().sum())} coins ever admitted", flush=True)
        for h in ["24h", "15h", "4h"]:
            p = load(h, "point_in_time")
            month = pd.to_datetime(p.t0, unit="ms").dt.to_period("M").dt.to_timestamp()
            ok = np.array(
                [
                    bool(member.at[mm, c]) if (mm in member.index and c in member.columns) else False
                    for mm, c in zip(month, p.coin)
                ]
            )
            q = p[ok].reset_index(drop=True)
            q["z_ex"] = exante_z(q, h)
            d = sc.gap_ci(q, "r_std", h, g1)
            dq = sc.gap_ci(q, "r_std", h, q10)
            de = sc.gap_ci(q[np.isfinite(q.z_ex)].reset_index(drop=True), "z_ex", h, q10)
            rows.append(
                dict(
                    topn=topn,
                    h=h,
                    n=len(q),
                    coins=q.coin.nunique(),
                    g1_gap=d["gap"],
                    g1_p=d["p"],
                    q10_gap=dq["gap"],
                    q10_p=dq["p"],
                    exq10_gap=de["gap"],
                    exq10_p=de["p"],
                )
            )
            print(
                f"U top {topn} {h}: {len(q):,} windows ({q.coin.nunique()} coins); "
                f"g1 gap {d['gap']:+.3f} p={d['p']:.3f}; Q10 gap {dq['gap']:+.3f} p={dq['p']:.3f}; "
                f"ex-ante Q10 gap {de['gap']:+.3f} p={de['p']:.3f}",
                flush=True,
            )
    pd.DataFrame(rows).to_csv(config.result_path("point_in_time.csv"), index=False)


def part_H3():
    rows = []
    for h in ["24h", "15h", "4h", "1h"]:
        b = load(h, "replication")
        b["z_ex"] = exante_z(b, h)
        b = b[np.isfinite(b.z_ex)].reset_index(drop=True)
        d = sc.gap_ci(b, "z_ex", h, q10)
        rows.append(dict(h=h, exq10_gap=d["gap"], exq10_lo=d["lo"], exq10_hi=d["hi"], exq10_p=d["p"]))
        print(f"H3 {h}: ex-ante Q10 gap {d['gap']:+.3f} p={d['p']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(config.result_path("replication_primary.csv"), index=False)


def part_J15(nb):
    hs = ["1h", "4h", "15h", "24h", "7d"]
    data = {}
    for h in hs:
        p = load(h)
        data[h] = (p.r_std.to_numpy(), exante_z(p, h).to_numpy(), p.g.to_numpy(), (p.t0 // DAYMS).to_numpy())
    days = np.unique(np.concatenate([d[3] for d in data.values()]))
    D = len(days)
    # Rows of each panel grouped by day, so that a resampled sequence of days selects rows in every panel.
    rows_by_day = {}
    for h, (_, _, _, day) in data.items():
        di = np.searchsorted(days, day)
        order = np.argsort(di, kind="stable")
        rows_by_day[h] = (order, np.searchsorted(di[order], np.arange(D + 1)))
    measures = [("bv", "g1"), ("bv", "q10"), ("exante", "q10")]
    keys = [(h, s, mn) for h in hs for s, mn in measures]
    fn = {"g1": g1, "q10": q10}

    def stat(h, scaling, mn, idx):
        zb, ze, g, _ = data[h]
        z = (zb if scaling == "bv" else ze)[idx]
        gg = g[idx]
        ok = np.isfinite(z)
        return fn[mn](z[ok & (gg == 1)]) - fn[mn](z[ok & (gg == -1)])

    obs = {k: stat(*k, np.arange(len(data[k[0]][0]))) for k in keys}
    bs = {k: np.empty(nb) for k in keys}
    for c0 in range(0, nb, 20):
        idx = sk.sb_index(D, 45, min(20, nb - c0))
        for j, ii in enumerate(idx):
            for h in hs:
                order, cuts = rows_by_day[h]
                sel = np.concatenate([order[cuts[k] : cuts[k + 1]] for k in ii])
                for s, mn in measures:
                    bs[(h, s, mn)][c0 + j] = stat(h, s, mn, sel)
        print(f"J15 {min(c0 + 20, nb)}/{nb}", flush=True)
    sd = {k: bs[k].std(ddof=1) for k in keys}
    t = {k: obs[k] / sd[k] for k in keys}
    cen = np.column_stack([(bs[k] - obs[k]) / sd[k] for k in keys])
    raw = {k: float(np.mean(np.abs(cen[:, keys.index(k)]) >= abs(t[k]))) for k in keys}
    # Romano-Wolf step-down: largest |t| first, monotone in the order of testing.
    order = sorted(keys, key=lambda k: -abs(t[k]))
    remaining = list(order)
    adj = {}
    prev = 0.0
    for k in order:
        cols = [keys.index(x) for x in remaining]
        mx = np.abs(cen[:, cols]).max(1)
        prev = max(prev, float(np.mean(mx >= abs(t[k]))))
        adj[k] = prev
        remaining.remove(k)
    rows = [
        dict(h=k[0], scaling=k[1], measure=k[2], gap=obs[k], t=t[k], p_raw=raw[k], p_rw=adj[k]) for k in keys
    ]
    pd.DataFrame(rows).to_csv(config.result_path("joint_test.csv"), index=False)
    for k in keys:
        print(
            f"J15 {'-'.join(k)}: gap {obs[k]:+.3f} t={t[k]:+.2f} p={raw[k]:.3f} RW p={adj[k]:.3f}", flush=True
        )


def main():
    global NB
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=NB, help="bootstrap replications (default 999)")
    ap.add_argument("--joint-reps", type=int, default=NB_JOINT, help="replications for J15 (default 1999)")
    ap.add_argument(
        "--parts",
        default=",".join(p for g in GROUPS for p in g),
        help="comma-separated subset of H2,I,T,P,U,H3,J15",
    )
    args = ap.parse_args()
    NB = args.reps
    ep.NB = NB
    sc.NB = NB
    selected = args.parts.split(",")
    run = {
        "H2": part_H2,
        "I": part_I,
        "T": part_T,
        "P": part_P,
        "U": part_U,
        "H3": part_H3,
        "J15": lambda: part_J15(args.joint_reps),
    }
    for group in GROUPS:
        todo = [p for p in group if p in selected]
        if todo:
            sk.rng = np.random.default_rng(sc.BOOT_SEED)
            for p in todo:
                run[p]()


if __name__ == "__main__":
    main()
