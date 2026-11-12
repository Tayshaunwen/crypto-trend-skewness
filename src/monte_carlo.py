"""Monte Carlo size and bias checks of the uptrend-minus-downtrend skewness gap.

Simulates ten coins over 3,200 days of 288 five-minute returns, builds daily returns,
bipower variation (BV), z = r / sqrt(BV) and the trend state from the simulated paths
(past 45-day SNR, threshold 0.75), and applies the same stationary block bootstrap test
as the main analysis (45-day blocks).

Data-generating process:
  log daily volatility  AR(1) around log 0.03 with a response to the previous absolute
                        standardized return, a common factor and coin-specific shocks
                        (symmetric: it depends on |return|, never on its sign)
  five-minute returns   sigma_d / sqrt(288) * (sqrt(rho) f + sqrt(1 - rho) e), with f, e
                        variance-one Student t(6) innovations (finite sixth moment), plus
                        symmetric jumps (probability 0.002, size N(0, (4 sigma_5min)^2))
Scenarios:
  nodrift   symmetric innovations, no drift: size of the test
  drift     daily drift kappa_s * sigma_d, with kappa_s the mean of r / sigma_hat by state in
            the real 24-hour panel; the state comes from the simulated SNR (with feedback)
  asymvol   intraday volatility rises with the window's cumulative return, same in all states
  statevol  as asymvol, but the sign depends on the state (positive in uptrends, negative in
            downtrends), matching the sign pattern of corr(r, log BV) in the data

Reads:  panel_24h.csv.gz (main sample; drift scenario only, for calibration)
Writes: mc_<scenario>.csv (skewness gap and bootstrap p-value of each replication)
Supports Table A6.

Usage: python src/monte_carlo.py {nodrift,drift,asymvol,statevol} [n_mc=1000] [n_boot=199] [beta=0.3]
"""

import argparse
import math

import numpy as np
import pandas as pd
from scipy import stats

import config
import skew_by_state as sk

SCENARIOS = {"nodrift": 1601, "drift": 1602, "asymvol": 1603, "statevol": 1604}  # scenario -> seed
K, D, N = 10, 3200, 288  # coins, days, five-minute returns per day
MU, PHI, ALPHA = math.log(0.03), 0.97, 0.06
SIG_C, SIG_I, RHO, NU = 0.12, 0.10, 0.6, 6.0
BURN_IN = 95  # initial days dropped while the SNR and volatility process settle
E_ABS_NORMAL = math.sqrt(2 / math.pi)


def tstd(rng, size):
    """Student t(NU) draws scaled to unit variance."""
    return stats.t.rvs(NU, size=size, random_state=rng) / math.sqrt(NU / (NU - 2))


def calibrate_kappa():
    """Mean standardized daily return by trend state in the real 24-hour panel."""
    p = pd.read_csv(config.panel_path("24h"), usecols=["r", "lrv1d", "snr45"])
    zh = p.r / np.sqrt(np.exp(p.lrv1d))
    g = np.where(p.snr45 > 0.75, 1, np.where(p.snr45 < -0.75, -1, 0))
    return {gv: float(zh[g == gv].mean()) for gv in (-1, 0, 1)}


def simulate(rng, scenario, kappa, beta):
    """One simulated panel: z, state, daily return and BV, each [days, coins] after burn-in."""
    lsig = np.full(K, MU)
    daily = np.empty((D, K))
    bv = np.empty((D, K))
    snr = np.full((D, K), np.nan)
    state = np.zeros((D, K))
    prev_abs = np.full(K, E_ABS_NORMAL)
    for d in range(D):
        lsig = (
            MU
            + PHI * (lsig - MU)
            + ALPHA * (prev_abs - E_ABS_NORMAL)
            + SIG_C * rng.standard_normal()
            + SIG_I * rng.standard_normal(K)
        )
        s = np.exp(lsig)
        if d >= 46:
            dr = daily[d - 45 : d]
            snr[d] = dr.sum(0) / (dr.std(0, ddof=1) * math.sqrt(45))
            state[d] = np.where(snr[d] > 0.75, 1, np.where(snr[d] < -0.75, -1, 0))
        drift = np.array([kappa[int(x)] for x in state[d]]) * s
        f = tstd(rng, N)
        e = tstd(rng, (N, K))
        eps = math.sqrt(RHO) * f[:, None] + math.sqrt(1 - RHO) * e
        mult = 1.0
        if scenario in ("asymvol", "statevol"):
            b = beta if scenario == "asymvol" else beta * state[d]
            c_prev = np.vstack([np.zeros((1, K)), np.cumsum(eps, 0)[:-1]]) / math.sqrt(N)
            # Not renormalised per day, so the window's BV level moves with the direction of its return.
            mult = np.exp(b * c_prev)
        x = eps * mult * s / math.sqrt(N) + drift / N
        jmp = rng.random((N, K)) < 0.002
        x = x + jmp * rng.standard_normal((N, K)) * 4 * s / math.sqrt(N)
        daily[d] = x.sum(0)
        bv[d] = (math.pi / 2) * np.sum(np.abs(x[1:]) * np.abs(x[:-1]), 0)
        prev_abs = np.abs(daily[d]) / s
    ok = np.arange(D) >= BURN_IN
    return (daily / np.sqrt(bv))[ok], state[ok], daily[ok], bv[ok]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scenario", nargs="?", default="nodrift", choices=list(SCENARIOS))
    ap.add_argument("n_mc", nargs="?", type=int, default=1000, help="Monte Carlo replications")
    ap.add_argument("n_boot", nargs="?", type=int, default=199, help="bootstrap draws per replication")
    ap.add_argument(
        "beta",
        nargs="?",
        type=float,
        default=0.3,
        help="response of intraday volatility to the cumulative window return",
    )
    args = ap.parse_args()

    rng = np.random.default_rng(SCENARIOS[args.scenario])
    kappa = calibrate_kappa() if args.scenario == "drift" else {-1: 0.0, 0: 0.0, 1: 0.0}
    out = []
    for i in range(args.n_mc):
        Z, G, R, BV = simulate(rng, args.scenario, kappa, args.beta)
        gap = sk.g1(Z[G == 1]) - sk.g1(Z[G == -1])
        corr_up = np.corrcoef(R[G == 1], np.log(BV[G == 1]))[0, 1]
        corr_dn = np.corrcoef(R[G == -1], np.log(BV[G == -1]))[0, 1]
        bs = sk.boot_regime(Z, G, 45, args.n_boot)
        out.append(
            dict(
                rep=i,
                gap=gap,
                corr_up=corr_up,
                corr_dn=corr_dn,
                p=sk.pval(bs["diff"]),
                share_up=float(np.mean(G == 1)),
                share_dn=float(np.mean(G == -1)),
            )
        )
        if (i + 1) % 50 == 0:
            d = pd.DataFrame(out)
            print(
                f"{args.scenario} {i + 1}/{args.n_mc}: mean gap {d.gap.mean():+.4f}, "
                f"5% rejection rate {np.mean(d.p < 0.05):.3f}",
                flush=True,
            )
    d = pd.DataFrame(out)
    d.to_csv(config.result_path(f"mc_{args.scenario}.csv"), index=False)
    print(
        f"{args.scenario}: {len(d)} replications, mean gap {d.gap.mean():+.4f} (sd {d.gap.std():.4f}), "
        f"rejection rate at 5% {np.mean(d.p < 0.05):.3f} (MC s.e. {math.sqrt(0.05 * 0.95 / len(d)):.3f}), "
        f"at 1% {np.mean(d.p < 0.01):.3f}; kappa = {kappa}",
        flush=True,
    )


if __name__ == "__main__":
    main()
