"""Real-time out-of-sample density forecasts of standardized window returns.

Target: z = r / sigma_hat, where sigma_hat^2 is an ex-ante HAR forecast of the window's
realised variance. At the start of each month, log RV is regressed on (lbv_lag, lrv1d,
lrv7d) over all windows that ended before that date, and sigma_hat^2 = exp(fit + s^2 / 2).
Three skewed-t models are then fitted to z on the same training data:
  M0   symmetric t
  M0s  unconditional skewed t (Hansen 1994)
  M1p  skewed t whose asymmetry alone depends on the trend state
Each model is estimated on an expanding window (main) and on a rolling two-year window.

Scores: log score, CRPS, tail-weighted CRPS (weights (2 tau - 1)^2, Gneiting and Ranjan 2011)
and quantile losses at 1/5/95/99%. M1p is compared with M0 and M0s by Diebold-Mariano tests
on the cross-coin average loss difference. The approximate power of the log-score test is
computed from the expected score gain implied by the fitted M1p and M0s (a KL divergence)
and the HAC standard error of the observed loss difference.

Reads:  panels panel_<h>.csv.gz (main sample)
Writes: oos_scores.csv, oos_power.csv
Supports Table 6.

Usage: python src/oos_forecasts.py [horizons, default 24h,15h]
"""

import argparse
import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import config
import density_models as dmod

TQ = [0.01, 0.05, 0.95, 0.99]
W_TAIL = (2 * dmod.GRID - 1) ** 2
STATE_THRESHOLD = 0.75
MIN_TRAIN = 500


def load_oos_panel(h, columns):
    p = pd.read_csv(config.panel_path(h), usecols=columns).dropna().reset_index(drop=True)
    p["ts"] = pd.to_datetime(p.t0, unit="ms", utc=True)
    p["g"] = np.where(p.snr45 > STATE_THRESHOLD, 1, np.where(p.snr45 < -STATE_THRESHOLD, -1, 0))
    p["lrv"] = np.log(p.rv)
    return p


def monthly_splits(p, h, window="expanding"):
    """Yield (training, test) frames month by month from the out-of-sample start.

    Training uses only windows that have ended by the first day of the test month.
    """
    months = pd.date_range(dmod.OOS0, p.ts.max(), freq="MS", tz="UTC")
    for i, m0 in enumerate(months):
        m1 = months[i + 1] if i + 1 < len(months) else p.ts.max() + pd.Timedelta(days=1)
        tr = p[p.ts + pd.Timedelta(hours=dmod.WINDOW_HOURS[h]) <= m0]
        if window == "rolling2y":
            tr = tr[tr.ts >= m0 - pd.DateOffset(years=2)]
        te = p[(p.ts >= m0) & (p.ts < m1)].copy()
        if len(te) == 0 or len(tr) < MIN_TRAIN:
            continue
        yield tr, te


def har_sigma(tr, te):
    """Ex-ante volatility forecasts for the training and test windows from a HAR regression."""
    X = np.column_stack([np.ones(len(tr)), tr.lbv_lag, tr.lrv1d, tr.lrv7d])
    b, *_ = np.linalg.lstsq(X, tr.lrv, rcond=None)
    s2 = np.var(tr.lrv - X @ b)
    Xt = np.column_stack([np.ones(len(te)), te.lbv_lag, te.lrv1d, te.lrv7d])
    # Log-normal correction: E[RV] = exp(E[log RV] + var / 2).
    return np.sqrt(np.exp(X @ b + s2 / 2)), np.sqrt(np.exp(Xt @ b + s2 / 2))


def score(y, prm):
    """Losses of a skewed-t forecast (mu, sigma, nu, lambda) for outcomes y.

    Returns log score, CRPS, tail-weighted CRPS, quantile losses at TQ and the TQ quantiles.
    The CRPS is approximated by twice the mean quantile loss over the 99-point grid.
    """
    mu, s, nu, lam = prm
    ls = -(dmod.sk_logpdf((y - mu) / s, nu, lam) - math.log(s))
    qg = mu + s * dmod.sk_ppf(dmod.GRID, nu, lam)
    e = y[:, None] - qg[None, :]
    ql = np.maximum(dmod.GRID * e, (dmod.GRID - 1) * e)
    qt = mu + s * dmod.sk_ppf(TQ, nu, lam)
    et = y[:, None] - qt[None, :]
    qlt = np.maximum(np.array(TQ) * et, (np.array(TQ) - 1) * et)
    return ls, 2 * ql.mean(1), 2 * (ql * W_TAIL).mean(1), qlt, qt


def run(h, window):
    """Monthly re-estimated forecasts; returns per-window losses and the last fitted parameters."""
    p = load_oos_panel(h, ["coin", "t0", "r", "rv", "lbv_lag", "lrv1d", "lrv7d", "snr45"])
    th = {}
    parts = []
    for tr, te in monthly_splits(p, h, window):
        sig_tr, te["sig"] = har_sigma(tr, te)
        x = (tr.r / sig_tr).to_numpy()
        g = tr.g.to_numpy()
        y = (te.r / te.sig).to_numpy()
        th["M0"] = dmod.fitd(x, False, th.get("M0"))
        th["M0s"] = dmod.fitd(x, True, th.get("M0s"))
        th["M1p"] = dmod.fit_m1p(x, g, th.get("M1p"))
        rec = {"t0": te.t0.to_numpy(), "coin": te.coin.to_numpy(), "g": te.g.to_numpy()}
        for mn in ("M0", "M0s"):
            ls, cr, tw, qlt, _ = score(y, dmod.unpack(th[mn], mn == "M0s"))
            rec.update({f"{mn}_LS": ls, f"{mn}_CRPS": cr, f"{mn}_twCRPS": tw})
            rec.update({f"{mn}_QL{t}": qlt[:, k] for k, t in enumerate(TQ)})
        n = len(y)
        ls, cr, tw, qlt = np.empty(n), np.empty(n), np.empty(n), np.empty((n, len(TQ)))
        for j, gv in enumerate(dmod.STATES):
            k = te.g.to_numpy() == gv
            if k.any():
                ls[k], cr[k], tw[k], qlt[k], _ = score(y[k], dmod.m1p_params(th["M1p"], j))
        rec.update({"M1p_LS": ls, "M1p_CRPS": cr, "M1p_twCRPS": tw})
        rec.update({f"M1p_QL{t}": qlt[:, k] for k, t in enumerate(TQ)})
        parts.append(pd.DataFrame(rec))
    return pd.concat(parts, ignore_index=True), th


def score_tests(D, h, window):
    rows = []
    for base in ("M0", "M0s"):
        for mt in ["LS", "CRPS", "twCRPS"] + [f"QL{t}" for t in TQ]:
            d = D[f"M1p_{mt}"] - D[f"{base}_{mt}"]
            mean, t, _ = dmod.dm(d.groupby(D.t0).mean().to_numpy(), dmod.NWLAG[h])
            rows.append(
                dict(
                    h=h,
                    window=window,
                    base=base,
                    metric=mt,
                    diff=mean,
                    rel=d.mean() / D[f"{base}_{mt}"].mean(),
                    dm_t=t,
                )
            )
    return rows


def power(D, th, h):
    """Approximate power of the DM log-score test of M1p against M0s at the 5% level."""
    zg = np.linspace(-12, 12, 24001)
    dz = zg[1] - zg[0]
    m0, s0, nu0, l0 = dmod.unpack(th["M0s"], True)
    kl = 0.0
    for j, gv in enumerate(dmod.STATES):
        w = np.mean(D.g == gv)
        mu, s, nu, lam = dmod.m1p_params(th["M1p"], j)
        f1 = np.exp(dmod.sk_logpdf((zg - mu) / s, nu, lam)) / s
        f0 = np.exp(dmod.sk_logpdf((zg - m0) / s0, nu0, l0)) / s0
        kl += w * np.sum(f1 * (np.log(f1 + 1e-300) - np.log(f0 + 1e-300))) * dz
    dls = (D.M1p_LS - D.M0s_LS).groupby(D.t0).mean().to_numpy()
    mean, t, n = dmod.dm(dls, dmod.NWLAG[h])
    se = abs(mean / t) if t != 0 else np.nan
    ncoin = D.groupby("t0").size().mean()
    pw = 1 - stats.norm.cdf(1.96 - kl / se) + stats.norm.cdf(-1.96 - kl / se)
    return dict(
        h=h,
        expected_LS_gain_per_window=kl,
        se_of_mean_diff=se,
        n_time=n,
        avg_coins=ncoin,
        approx_power=pw,
        gain_needed_for_80pct_power=2.8 * se,
    )


def main():
    warnings.filterwarnings("ignore")  # overflow warnings from the optimizer's trial points
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("horizons", nargs="?", default="24h,15h", help="comma-separated horizons")
    hs = ap.parse_args().horizons.split(",")
    scores, powers = [], []
    for h in hs:
        for window in ("expanding", "rolling2y"):
            D, th = run(h, window)
            rows = score_tests(D, h, window)
            scores += rows
            t_ls = next(r["dm_t"] for r in rows if r["base"] == "M0s" and r["metric"] == "LS")
            t_tw = next(r["dm_t"] for r in rows if r["base"] == "M0s" and r["metric"] == "twCRPS")
            print(
                f"{h} {window}: M1p - M0s  log score t={t_ls:+.2f}  tail-weighted CRPS t={t_tw:+.2f}",
                flush=True,
            )
            if window == "expanding":
                powers.append(power(D, th, h))
                print(f"{h} power: {powers[-1]['approx_power']:.2f}", flush=True)
    pd.DataFrame(scores).to_csv(config.result_path("oos_scores.csv"), index=False)
    pd.DataFrame(powers).to_csv(config.result_path("oos_power.csv"), index=False)


if __name__ == "__main__":
    main()
