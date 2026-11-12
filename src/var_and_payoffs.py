"""Value-at-Risk backtests by trend state and out-of-sample payoff ratios.

Separates the role of state-dependent location and scale from state-dependent skewness.
Target: z_hat = r / sigma_hat with the ex-ante HAR volatility forecast of oos_forecasts.py
(expanding window, re-estimated monthly, out of sample from 2019-01). Skewed-t models:
  M0s      all parameters shared across states
  M1p      only the asymmetry lambda depends on the state
  Mloc     location and scale depend on the state; lambda and nu shared
  Mloc+λ   location, scale and lambda depend on the state; nu shared

Reads:  panels panel_<h>.csv.gz (main sample)
Writes:
  state_moments.csv   mean and variance of z_hat and of z = r / sqrt(BV) by state
  var_backtests.csv   hit rates by model, state and quantile (time-block bootstrap intervals),
                      Kupiec p-values, Christoffersen independence (summed over coins) and a
                      dynamic-quantile test of hit - tau on [1, up, down, lagged hit] that tests
                      only the two state coefficients
  mloc_scores.csv     Mloc+λ minus Mloc: log score, tail-weighted CRPS and quantile losses at
                      the four VaR levels, with Diebold-Mariano t statistics
  payoff_ratios.csv   nonparametric E[(z_hat - k)+] / E[(-k - z_hat)+] and P(z_hat > k) / P(z_hat < -k)
                      by state for k = 1.28 and 2.0, with bootstrap intervals; rows resid_* use
                      the Mloc residuals, i.e. shape only after removing state location and scale
Supports Tables 6, 7, 8, A9 and A10.

Usage: python src/var_and_payoffs.py [horizons, default 24h,15h]
"""

import argparse
import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

import config
import density_models as dmod
import oos_forecasts as oos
import skew_by_state as sk

TQ = [0.01, 0.05, 0.95, 0.99]
ST = dmod.STATES
STATE_NAMES = ((1, "up"), (0, "rg"), (-1, "dn"))
STRIKES = (1.28, 2.0)
N_BOOT = 499
# Model name -> (location and scale vary by state, lambda varies by state).
SPECS = {"M0s": (False, False), "M1p": (False, True), "Mloc": (True, False), "Mloc+λ": (True, True)}


def state_params(th, loc, lam):
    """Parameter vector -> {state: (mu, sigma, nu, lambda)}."""
    nloc = 3 if loc else 1
    mu = th[:nloc]
    ls = th[nloc : 2 * nloc]
    nu = 2.05 + math.exp(min(max(th[2 * nloc], -3), 6))
    la = th[2 * nloc + 1 : 2 * nloc + 1 + (3 if lam else 1)]

    def pick(v, j):
        return v[j] if len(v) == 3 else v[0]

    return {gv: (pick(mu, j), math.exp(pick(ls, j)), nu, math.tanh(pick(la, j))) for j, gv in enumerate(ST)}


def nll_states(th, x, g, loc, lam):
    try:
        tot = 0.0
        for gv, (mu, s, nu, lm) in state_params(th, loc, lam).items():
            z = (x[g == gv] - mu) / s
            tot += (dmod.sk_logpdf(z, nu, lm) - math.log(s)).sum()
        return -tot if np.isfinite(tot) else 1e18
    except (ValueError, OverflowError):
        return 1e18


def fit_states(x, g, loc, lam, th0=None):
    if th0 is None:
        nloc = 3 if loc else 1
        th0 = np.concatenate(
            [[np.median(x)] * nloc, [math.log(x.std())] * nloc, [math.log(4.0)], [0.0] * (3 if lam else 1)]
        )
    return minimize(
        nll_states,
        th0,
        args=(x, g, loc, lam),
        method="Nelder-Mead",
        options=dict(maxiter=20000, xatol=1e-5, fatol=1e-4),
    ).x


def run(h):
    """Monthly re-estimated forecasts; per test window the VaR levels of every model."""
    cols = ["coin", "t0", "r", "rv", "r_std", "lbv_lag", "lrv1d", "lrv7d", "snr45"]
    p = oos.load_oos_panel(h, cols)
    th = {}
    parts = []
    for tr, te in oos.monthly_splits(p, h):
        sig_tr, te["sig"] = oos.har_sigma(tr, te)
        x = (tr.r / sig_tr).to_numpy()
        g = tr.g.to_numpy()
        te["zh"] = te.r / te.sig
        for name, (loc, lam) in SPECS.items():
            th[name] = fit_states(x, g, loc, lam, th.get(name))
            P = state_params(th[name], loc, lam)
            if name in ("Mloc", "Mloc+λ"):
                for j, c in enumerate(("mu", "sd", "nu", "lam")):
                    te[f"{name}_{c}"] = te.g.map({gv: P[gv][j] for gv in ST})
            for t in TQ:
                te[f"{name}_v{t}"] = np.nan
                for gv in ST:
                    mu, s, nu, lm = P[gv]
                    k = te.g == gv
                    te.loc[k, f"{name}_v{t}"] = te.sig[k] * (mu + s * dmod.sk_ppf([t], nu, lm)[0])
        parts.append(te)
    return pd.concat(parts, ignore_index=True)


def block_indices(rng, T, b, n):
    """Circular moving-block bootstrap of time indices with fixed block length b."""
    return [
        np.concatenate([np.arange(s0, s0 + b) % T for s0 in rng.integers(0, T, int(np.ceil(T / b)))])[:T]
        for _ in range(n)
    ]


def kupiec_p(n, xh, tau):
    ph = xh / n
    lr = -2 * (
        (n - xh) * math.log(1 - tau)
        + xh * math.log(tau)
        - ((n - xh) * math.log(1 - ph) if ph < 1 else 0)
        - (xh * math.log(ph) if ph > 0 else 0)
    )
    return ph, 1 - stats.chi2.cdf(lr, 1)


def _bernoulli_ll(n0, n1, q):
    return (n0 * math.log(1 - q) if n0 else 0) + (n1 * math.log(q) if n1 else 0) if 0 < q < 1 else 0.0


def christoffersen_lr(hits):
    """Likelihood-ratio statistic for first-order independence of one coin's hit sequence."""
    a, b = hits[:-1], hits[1:]
    n00, n01, n10, n11 = [np.sum((a == i) & (b == j)) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    pp = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    return -2 * (
        _bernoulli_ll(n00 + n10, n01 + n11, pp) - _bernoulli_ll(n00, n01, p01) - _bernoulli_ll(n10, n11, p11)
    )


def dq_state_test(D, hit, tau, times, b):
    """Regress hit - tau on [1, up, down, lagged hit] with time-block clustered errors.

    Returns the coefficients and the Wald p-value of the two state coefficients.
    """
    Dh = D.assign(hit=hit.astype(float))
    Dh["lag"] = Dh.groupby("coin").hit.shift(1)
    Dh = Dh.dropna(subset=["lag"])
    y = (Dh.hit - tau).to_numpy()
    X = np.column_stack([np.ones(len(Dh)), Dh.g == 1, Dh.g == -1, Dh.lag]).astype(float)
    bh = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ bh
    cl = np.searchsorted(times, Dh.t0.to_numpy()) // b
    Sg = pd.DataFrame(X * u[:, None]).groupby(cl).sum().to_numpy()
    A = np.linalg.inv(X.T @ X)
    Vc = A @ (Sg.T @ Sg) @ A
    sel = [1, 2]
    w = bh[sel] @ np.linalg.solve(Vc[np.ix_(sel, sel)], bh[sel])
    return bh, 1 - stats.chi2.cdf(w, 2)


def var_backtests(D, h, tix, times, b, bidx):
    rows = []
    for name in SPECS:
        for t in TQ:
            v = D[f"{name}_v{t}"]
            hit = ((D.r < v) if t < 0.5 else (D.r > v)).to_numpy()
            tt = t if t < 0.5 else 1 - t
            for gv, nm in STATE_NAMES + ((9, "all"),):
                k = (D.g == gv).to_numpy() if gv != 9 else np.ones(len(D), bool)
                agg = pd.DataFrame({"h": hit & k, "k": k}).groupby(tix).sum().to_numpy()
                bs = [agg[bb, 0].sum() / max(agg[bb, 1].sum(), 1) for bb in bidx]
                ph, kp = kupiec_p(k.sum(), (hit & k).sum(), tt)
                rows.append(
                    dict(
                        h=h,
                        model=name,
                        tau=t,
                        state=nm,
                        hit=ph,
                        lo=np.percentile(bs, 2.5),
                        hi=np.percentile(bs, 97.5),
                        kupiec_p=kp,
                    )
                )
            lr_ind = 0.0
            for _, dd in D.assign(hit=hit).groupby("coin"):
                lr_ind += christoffersen_lr(dd.hit.to_numpy().astype(int))
            bh, dq_p = dq_state_test(D, hit, tt, times, b)
            rows.append(
                dict(
                    h=h,
                    model=name,
                    tau=t,
                    state="tests",
                    hit=np.nan,
                    lo=bh[1],
                    hi=bh[2],
                    kupiec_p=np.nan,
                    christ_p=1 - stats.chi2.cdf(lr_ind, D.coin.nunique()),
                    dq_state_p=dq_p,
                )
            )
        up95 = next(r for r in rows if r["model"] == name and r["tau"] == 0.95 and r["state"] == "up")
        print(f"{h} {name}: 95% short VaR, uptrend hit rate {up95['hit']:.4f}", flush=True)
    return rows


def call_put_sums(v, mask, k, tix):
    """Per time point, summed (v - k)+ and (-k - v)+ over the windows in mask."""
    return (
        pd.DataFrame(
            {"c": np.where(mask, np.maximum(v - k, 0), 0), "p": np.where(mask, np.maximum(-k - v, 0), 0)}
        )
        .groupby(tix)
        .sum()
        .to_numpy()
    )


def ratio(a):
    return a[:, 0].sum() / a[:, 1].sum()


def exceed_ratio(a):
    return a[:, 2].sum() / max(a[:, 3].sum(), 1)


def ratio_gap(a_up, a_dn):
    return ratio(a_up) - ratio(a_dn)


def payoff_ratios(D, h, tix, bidx):
    rows = []
    zh = D.zh.to_numpy()
    up, dn = (D.g == 1).to_numpy(), (D.g == -1).to_numpy()
    for k in STRIKES:
        for gv, nm in STATE_NAMES:
            msk = (D.g == gv).to_numpy()
            agg = (
                pd.DataFrame(
                    {
                        "c": np.where(msk, np.maximum(zh - k, 0), 0),
                        "p": np.where(msk, np.maximum(-k - zh, 0), 0),
                        "eu": np.where(msk, zh > k, 0),
                        "el": np.where(msk, zh < -k, 0),
                    }
                )
                .groupby(tix)
                .sum()
                .to_numpy()
            )

            bsr = [ratio(agg[bb]) for bb in bidx]
            bse = [exceed_ratio(agg[bb]) for bb in bidx]
            rows.append(
                dict(
                    h=h,
                    strike=k,
                    state=nm,
                    payoff_ratio=ratio(agg),
                    lo=np.percentile(bsr, 2.5),
                    hi=np.percentile(bsr, 97.5),
                    exceed_ratio=exceed_ratio(agg),
                    ex_lo=np.percentile(bse, 2.5),
                    ex_hi=np.percentile(bse, 97.5),
                    n=int(msk.sum()),
                )
            )
    for k in STRIKES:
        A_up, A_dn = call_put_sums(zh, up, k, tix), call_put_sums(zh, dn, k, tix)
        bs = np.array([ratio_gap(A_up[bb], A_dn[bb]) for bb in bidx])
        rows.append(
            dict(
                h=h,
                strike=k,
                state="up_minus_dn",
                payoff_ratio=ratio_gap(A_up, A_dn),
                lo=np.percentile(bs, 2.5),
                hi=np.percentile(bs, 97.5),
                exceed_ratio=np.nan,
                n=int(up.sum() + dn.sum()),
                p=sk.pval(bs),
            )
        )
    # Shape only: residuals after removing the state-specific location and scale of Mloc.
    e = ((D.zh - D["Mloc_mu"]) / D["Mloc_sd"]).to_numpy()
    for k in STRIKES:
        for gv, nm in STATE_NAMES:
            msk = (D.g == gv).to_numpy()
            agg = call_put_sums(e, msk, k, tix)
            bs = [ratio(agg[bb]) for bb in bidx]
            rows.append(
                dict(
                    h=h,
                    strike=k,
                    state=f"resid_{nm}",
                    payoff_ratio=ratio(agg),
                    lo=np.percentile(bs, 2.5),
                    hi=np.percentile(bs, 97.5),
                    n=int(msk.sum()),
                )
            )
        A_up, A_dn = call_put_sums(e, up, k, tix), call_put_sums(e, dn, k, tix)
        bs = np.array([ratio_gap(A_up[bb], A_dn[bb]) for bb in bidx])
        rows.append(
            dict(
                h=h,
                strike=k,
                state="resid_up_minus_dn",
                payoff_ratio=ratio_gap(A_up, A_dn),
                lo=np.percentile(bs, 2.5),
                hi=np.percentile(bs, 97.5),
                p=sk.pval(bs),
            )
        )
    return rows


def mloc_scores(D, h):
    """Incremental value of state-dependent skewness over state-dependent location and scale."""
    rows = []
    zh = D.zh.to_numpy()
    sc = {}
    for name in ("Mloc", "Mloc+λ"):
        ls = np.empty(len(D))
        tw = np.empty(len(D))
        keycols = [f"{name}_{c}" for c in ("mu", "sd", "nu", "lam")]
        for key, idx in D.groupby(keycols).groups.items():
            idx = np.asarray(idx)
            res = oos.score(zh[idx], tuple(key))
            ls[idx] = res[0]
            tw[idx] = res[2]
        sc[name] = (ls, tw)
    for j, mt in enumerate(("LS", "twCRPS")):
        diff = sc["Mloc+λ"][j] - sc["Mloc"][j]
        d = pd.Series(diff).groupby(D.t0.to_numpy()).mean().to_numpy()
        mean, t, _ = dmod.dm(d, dmod.NWLAG[h])
        rows.append(
            dict(h=h, metric=mt, diff=mean, rel=float(np.mean(diff) / np.mean(sc["Mloc"][j])), dm_t=t)
        )
    # Quantile (tick) losses in units of z_hat at the four VaR levels.
    for t in TQ:
        ql = {}
        for name in ("Mloc", "Mloc+λ"):
            u = ((D.r - D[f"{name}_v{t}"]) / D.sig).to_numpy()
            ql[name] = (t - (u < 0)) * u
        diff = ql["Mloc+λ"] - ql["Mloc"]
        d = pd.Series(diff).groupby(D.t0.to_numpy()).mean().to_numpy()
        mean, tstat, _ = dmod.dm(d, dmod.NWLAG[h])
        rows.append(
            dict(h=h, metric=f"QL{t}", diff=mean, rel=float(np.mean(diff) / np.mean(ql["Mloc"])), dm_t=tstat)
        )
    for r in rows:
        print(f"{h} Mloc+λ - Mloc {r['metric']}: {r['rel']:+.4%} t={r['dm_t']:+.2f}", flush=True)
    return rows


def main():
    warnings.filterwarnings("ignore")  # overflow warnings from the optimizer's trial points
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("horizons", nargs="?", default="24h,15h", help="comma-separated horizons")
    hs = ap.parse_args().horizons.split(",")
    moments, var_rows, payoff_rows, score_rows = [], [], [], []
    for h in hs:
        D = run(h).sort_values(["coin", "t0"]).reset_index(drop=True)
        times = np.sort(D.t0.unique())
        tix = np.searchsorted(times, D.t0.to_numpy())
        rng = np.random.default_rng(13)
        b = int(sk.block_for(h))
        bidx = block_indices(rng, len(times), b, N_BOOT)
        for gv, nm in STATE_NAMES:
            d = D[D.g == gv]
            moments.append(
                dict(
                    h=h,
                    state=nm,
                    mean_zhat=d.zh.mean(),
                    var_zhat=d.zh.var(),
                    mean_z=d.r_std.mean(),
                    var_z=d.r_std.var(),
                    n=len(d),
                )
            )
        var_rows += var_backtests(D, h, tix, times, b, bidx)
        payoff_rows += payoff_ratios(D, h, tix, bidx)
        score_rows += mloc_scores(D, h)
    pd.DataFrame(moments).to_csv(config.result_path("state_moments.csv"), index=False)
    pd.DataFrame(var_rows).to_csv(config.result_path("var_backtests.csv"), index=False)
    pd.DataFrame(payoff_rows).to_csv(config.result_path("payoff_ratios.csv"), index=False)
    pd.DataFrame(score_rows).to_csv(config.result_path("mloc_scores.csv"), index=False)


if __name__ == "__main__":
    main()
