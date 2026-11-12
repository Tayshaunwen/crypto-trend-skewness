"""Skewed-t density models and forecast-comparison helpers.

Library module (no command-line entry point). Provides Hansen's (1994) skewed t
distribution, maximum-likelihood fits of the unconditional model and of the
parsimonious state model (only the asymmetry parameter varies with the trend
state), and the Diebold-Mariano test with Newey-West standard errors.

Used by oos_forecasts.py and var_and_payoffs.py (Tables 6-8, A9, A10).
"""

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import t as student_t

# First month of the out-of-sample period (training data start in 2017-08).
OOS0 = pd.Timestamp("2019-01-01", tz="UTC")
# Probability grid for the quantile-loss approximation of the CRPS.
GRID = np.linspace(0.005, 0.995, 99)
# Newey-West lags for the Diebold-Mariano test, by window horizon.
NWLAG = {"24h": 13, "15h": 13, "4h": 24, "1h": 37, "7d": 7}
WINDOW_HOURS = {"1h": 1, "4h": 4, "15h": 15, "24h": 24, "7d": 168}
STATES = (-1, 0, 1)


def _skewt_constants(nu, lam):
    """Constants a, b, c of Hansen's skewed t (standardized to mean 0, variance 1)."""
    c = math.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2)) / math.sqrt(math.pi * (nu - 2))
    a = 4 * lam * c * (nu - 2) / (nu - 1)
    b = math.sqrt(1 + 3 * lam**2 - a**2)
    return a, b, c


def sk_logpdf(z, nu, lam):
    """Log density of the standardized skewed t; requires nu > 2 and -1 < lam < 1."""
    a, b, c = _skewt_constants(nu, lam)
    s = np.where(z < -a / b, 1 - lam, 1 + lam)
    u = (b * z + a) / s
    return math.log(b) + math.log(c) - (nu + 1) / 2 * np.log1p(u**2 / (nu - 2))


def sk_ppf(q, nu, lam):
    """Quantile function of the standardized skewed t."""
    a, b, _ = _skewt_constants(nu, lam)
    q = np.asarray(q, float)
    sc = math.sqrt((nu - 2) / nu)
    lo = q < (1 - lam) / 2
    x = np.where(
        lo,
        (1 - lam) * sc * student_t.ppf(q / (1 - lam), nu),
        (1 + lam) * sc * student_t.ppf(0.5 + (q - (1 - lam) / 2) / (1 + lam), nu),
    )
    return (x - a) / b


def unpack(th, skew):
    """Unconstrained vector (mu, log sigma, log(nu - 2.05), atanh lambda) -> (mu, sigma, nu, lambda).

    With ``skew=False`` the asymmetry is fixed at zero (symmetric t).
    """
    mu, ls, ln = th[0], th[1], min(max(th[2], -3.0), 6.0)
    nu = 2.05 + math.exp(ln)
    lam = math.tanh(th[3]) if skew else 0.0
    return mu, math.exp(ls), nu, lam


def nll(th, x, skew):
    try:
        mu, s, nu, lam = unpack(th, skew)
        v = -(sk_logpdf((x - mu) / s, nu, lam) - math.log(s)).sum()
        return v if np.isfinite(v) else 1e18
    except (ValueError, OverflowError):
        return 1e18


def fitd(x, skew, th0=None):
    """Fit the (skewed) t by maximum likelihood; ``th0`` warm-starts from a previous fit."""
    if th0 is None:
        th0 = np.array([np.median(x), math.log(x.std()), math.log(4.0), 0.0])
    r = minimize(
        nll, th0, args=(x, skew), method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-5, fatol=1e-4)
    )
    return r.x


def nll_p(th, x, g):
    """Negative log-likelihood of the parsimonious state model M1p.

    Location, scale and degrees of freedom are shared; th[3:6] hold the asymmetry
    of the downtrend, range and uptrend states.
    """
    try:
        mu, ls, ln = th[0], th[1], min(max(th[2], -3.0), 6.0)
        s = math.exp(ls)
        nu = 2.05 + math.exp(ln)
        tot = 0.0
        for j, gv in enumerate(STATES):
            lam = math.tanh(th[3 + j])
            z = (x[g == gv] - mu) / s
            tot += (sk_logpdf(z, nu, lam) - math.log(s)).sum()
        return -tot if np.isfinite(tot) else 1e18
    except (ValueError, OverflowError):
        return 1e18


def fit_m1p(x, g, th0=None):
    if th0 is None:
        th0 = np.array([np.median(x), math.log(x.std()), math.log(4.0), 0, 0, 0])
    return minimize(
        nll_p, th0, args=(x, g), method="Nelder-Mead", options=dict(maxiter=6000, xatol=1e-5, fatol=1e-4)
    ).x


def m1p_params(th, j):
    """(mu, sigma, nu, lambda) of M1p for state index j (0 = down, 1 = range, 2 = up)."""
    return th[0], math.exp(th[1]), 2.05 + math.exp(min(max(th[2], -3), 6)), math.tanh(th[3 + j])


def dm(d, lag):
    """Diebold-Mariano test of a loss-difference series: (mean, t statistic, n).

    Newey-West variance with Bartlett weights; NaNs are dropped.
    """
    d = d[~np.isnan(d)]
    n = len(d)
    m = d.mean()
    u = d - m
    v = (u @ u) / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n
    return m, m / math.sqrt(v / n), n
