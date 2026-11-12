"""Panel quantile regressions of standardized returns on the trend state, with controls.

Model, pooled over the ten main coins (range state is the base group):

    Q_tau(z | X) = a_tau + b_up,tau * up + b_dn,tau * down + controls + coin fixed effects

Controls use only information available before the window opens: lbv_lag, lrv1d, lrv7d,
lvol_rel, r1d, rstd_lag and a weekend dummy. The statistic of interest is the change in
quantile asymmetry,

    QS_up = b_up(0.9) + b_up(0.1) - 2 b_up(0.5),   QS_dn likewise,   DIFF = QS_up - QS_dn,

also reported for tau = 0.05 / 0.95.

Standard errors are analytic and cluster-robust jointly across quantiles (in the spirit of
Parente and Santos Silva, 2016):

    V(t, t') = A_t^-1 B(t, t') A_t'^-1,      A_t = sum_i K_h(u_i) x_i x_i'
    B(t, t') = sum_g s_gt s_gt',             s_gt = sum_{i in g} (t - 1{u_i < 0}) x_i

with a uniform (Powell) kernel and Hall-Sheather bandwidth. Clusters are blocks of 45
consecutive days, which absorb both cross-coin correlation and short-run serial correlation.
A cluster bootstrap of the whole fit takes hours, so it is run only as a check (--bootcheck).

Reads:  main-sample panels for 24h, 15h and 4h windows.
Writes: quantile_regression.csv, quantile_regression_noctrl.csv (--no-controls),
        quantile_regression_bootcheck_<h>.csv (--bootcheck).
Paper:  Table 5 and online Table A7.

Usage:
    python src/quantile_regression.py [--no-controls]
    python src/quantile_regression.py --bootcheck 24h [--reps 99]
"""

import argparse
import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.regression.quantile_regression import QuantReg

import config

warnings.filterwarnings("ignore")

TH = 0.75
TAUS = [0.05, 0.1, 0.5, 0.9, 0.95]
CTRL = ["lbv_lag", "lrv1d", "lrv7d", "lvol_rel", "r1d", "rstd_lag", "wkd"]
HS = ["24h", "15h", "4h"]
# Cluster size in windows: 45 consecutive days, the block length used for all inference.
BLOCK = {"24h": 45, "15h": 45, "4h": 270, "1h": 1080, "7d": 7}


def load_panel(h):
    p = pd.read_csv(config.panel_path(h))
    p["wkd"] = (p.dow >= 5).astype(float)
    return p.dropna(subset=CTRL + ["snr45", "r_std"]).reset_index(drop=True)


def time_clusters(p, h):
    times = np.sort(p.t0.unique())
    return np.searchsorted(times, p.t0.to_numpy()) // BLOCK[h]


def design(p, controls=True):
    X = pd.DataFrame({"const": 1.0, "up": (p.snr45 > TH).astype(float), "dn": (p.snr45 < -TH).astype(float)})
    for c in CTRL if controls else []:
        X[c] = p[c].to_numpy()
    for c in sorted(p.coin.unique())[1:]:
        X[f"fe_{c}"] = (p.coin == c).astype(float).to_numpy()
    return X


def hs_bw(n, tau, alpha=0.05):
    """Hall-Sheather bandwidth (in probability units)."""
    z = norm.ppf(1 - alpha / 2)
    q = norm.ppf(tau)
    return n ** (-1 / 3) * z ** (2 / 3) * (1.5 * norm.pdf(q) ** 2 / (2 * q**2 + 1)) ** (1 / 3)


def fit_horizon(h, controls):
    """Fit all quantiles for one horizon; return {stat: (estimate, standard error)} and n."""
    p = load_panel(h)
    y = p.r_std.to_numpy()
    Xd = design(p, controls)
    X = Xd.to_numpy()
    n = X.shape[0]
    cl = time_clusters(p, h)
    beta, score, a_inv = {}, {}, {}
    for t in TAUS:
        b = QuantReg(y, Xd).fit(q=t, max_iter=2000).params.to_numpy()
        beta[t] = b
        u = y - X @ b
        kappa = min(u.std(), (np.percentile(u, 75) - np.percentile(u, 25)) / 1.34)
        hb = hs_bw(n, t)
        hu = kappa * (norm.ppf(min(t + hb, 0.999)) - norm.ppf(max(t - hb, 0.001)))
        w = (np.abs(u) < hu) / (2 * hu)
        a_inv[t] = np.linalg.inv((X * w[:, None]).T @ X)
        psi = t - (u < 0)
        score[t] = pd.DataFrame(X * psi[:, None]).groupby(cl).sum().to_numpy()  # [clusters, k]
        print(f"  {h} tau={t} done", flush=True)

    def cov(t1, t2):
        return a_inv[t1] @ (score[t1].T @ score[t2]) @ a_inv[t2]

    def combine(tau_weights, col_weights):
        """Estimate and s.e. of sum over (tau, column) of weight * beta[tau][column]."""
        est = sum(wt * sum(cw * beta[t][c] for c, cw in col_weights) for t, wt in tau_weights)
        var = 0.0
        for t1, w1 in tau_weights:
            for t2, w2 in tau_weights:
                C = cov(t1, t2)
                for c1, a1 in col_weights:
                    for c2, a2 in col_weights:
                        var += w1 * w2 * a1 * a2 * C[c1, c2]
        return est, math.sqrt(max(var, 0))

    iu, idn = Xd.columns.get_loc("up"), Xd.columns.get_loc("dn")
    out = {}
    for lo, hi, tag in ((0.1, 0.9, "10"), (0.05, 0.95, "05")):
        W = [(hi, 1), (lo, 1), (0.5, -2)]
        out[f"QS{tag}_up"] = combine(W, [(iu, 1)])
        out[f"QS{tag}_dn"] = combine(W, [(idn, 1)])
        out[f"QS{tag}_diff"] = combine(W, [(iu, 1), (idn, -1)])
    for t in TAUS:
        out[f"bup_{t}"] = combine([(t, 1)], [(iu, 1)])
        out[f"bdn_{t}"] = combine([(t, 1)], [(idn, 1)])
    print(f"\n{h}: n={n:,}, clusters={score[0.5].shape[0]}")
    return out, n


def run_regressions(controls):
    rows = []
    for h in HS:
        out, n = fit_horizon(h, controls)
        for key, (e, se) in out.items():
            z = e / se if se > 0 else float("nan")
            pv = 2 * (1 - norm.cdf(abs(z)))
            rows.append(dict(h=h, stat=key, est=e, se=se, z=z, p=pv, lo=e - 1.96 * se, hi=e + 1.96 * se, n=n))
            print(f"  {key:<10} {e:+.4f}  se {se:.4f}  z={z:+.2f}  p={pv:.3f}", flush=True)
    name = "quantile_regression.csv" if controls else "quantile_regression_noctrl.csv"
    pd.DataFrame(rows).to_csv(config.result_path(name), index=False)


def qs_stats(y, X):
    b = {t: QuantReg(y, X).fit(q=t, max_iter=2000).params for t in TAUS}
    out = {}
    for lo, hi, tag in ((0.1, 0.9, "10"), (0.05, 0.95, "05")):
        for g in ("up", "dn"):
            out[f"QS{tag}_{g}"] = b[hi][g] + b[lo][g] - 2 * b[0.5][g]
        out[f"QS{tag}_diff"] = out[f"QS{tag}_up"] - out[f"QS{tag}_dn"]
    return out


def bootcheck(h, reps):
    """Resample whole time clusters with replacement and compare bootstrap with analytic s.e."""
    rng = np.random.default_rng(11)
    p = load_panel(h)
    y = p.r_std.to_numpy()
    X = design(p)
    cl = time_clusters(p, h)
    groups = [np.flatnonzero(cl == g) for g in range(cl.max() + 1)]
    rows = []
    for k in range(reps):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[g] for g in pick])
        rows.append(qs_stats(y[idx], X.iloc[idx].reset_index(drop=True)))
        if (k + 1) % 10 == 0:
            print(f"{h} {k + 1}/{reps}", flush=True)
    B = pd.DataFrame(rows)
    B.to_csv(config.result_path(f"quantile_regression_bootcheck_{h}.csv"), index=False)

    A = pd.read_csv(config.result_path("quantile_regression.csv"))
    A = A[A.h == h].set_index("stat")
    print(f"\n{h}: analytic vs cluster-bootstrap standard errors ({reps} replications)")
    for c in B.columns:
        sd = B[c].std(ddof=1)
        print(
            f"  {c:<10} est {A.at[c, 'est']:+.4f}  analytic {A.at[c, 'se']:.4f}  bootstrap {sd:.4f}"
            f"  ratio {sd / A.at[c, 'se']:.2f}"
            f"  95% CI [{np.percentile(B[c], 2.5):+.3f}, {np.percentile(B[c], 97.5):+.3f}]"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-controls", action="store_true", help="state dummies and coin fixed effects only")
    ap.add_argument("--bootcheck", metavar="H", help="cluster-bootstrap check for horizon H (24h or 15h)")
    ap.add_argument("--reps", type=int, default=99, help="bootstrap replications for --bootcheck")
    args = ap.parse_args()
    if args.bootcheck:
        bootcheck(args.bootcheck, args.reps)
    else:
        run_regressions(controls=not args.no_controls)


if __name__ == "__main__":
    main()
