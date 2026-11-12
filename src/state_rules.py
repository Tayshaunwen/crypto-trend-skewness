"""Sensitivity of the up-minus-down skewness gap to the definition of the trend state.

Reads:  config.panel_path(h, "main") for h in 4h, 15h, 24h, 7d (build_windows.py)
Writes: state_rule_sensitivity.csv
Supports: Table A2; Figure A1.

Rules (pooled panel of all coins):
    snr{W} three states, W in 30, 45, 90, threshold in 0.5, 0.75, 1.0
    sgn{N} two states (past N-day return up / down), N in 30, 60, 90
    ma{N}  two states (price above / below its past N-day mean), N in 30, 60

Usage:
    python src/state_rules.py [n_bootstrap=999]
"""

import argparse

import numpy as np
import pandas as pd

import config
import skew_by_state as sk

HS = ["4h", "15h", "24h", "7d"]


def defs():
    """(rule name, panel column, function mapping the column to states +1 / 0 / -1)."""
    for W in (30, 45, 90):
        for th in (0.5, 0.75, 1.0):
            yield f"snr{W}_th{th}", f"snr{W}", lambda s, th=th: np.where(s > th, 1, np.where(s < -th, -1, 0))
    for N in (30, 60, 90):
        yield f"sgn{N}", f"sgn{N}", lambda s: np.sign(s).astype(int)
    for N in (30, 60):
        yield f"ma{N}", f"ma{N}", lambda s: np.sign(s).astype(int)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("n_bootstrap", nargs="?", type=int, default=999)
    nb = ap.parse_args().n_bootstrap
    rows = []
    for h in HS:
        p = pd.read_csv(config.panel_path(h))
        coins = list(p.coin.unique())
        R = p.pivot(index="t0", columns="coin", values="r_std")[coins]
        b = sk.block_for(h)
        for name, col, rule in defs():
            p["g"] = rule(p[col].to_numpy())
            G = p.pivot(index="t0", columns="coin", values="g")[coins].to_numpy()
            Rm = R.to_numpy()
            ok = ~np.isnan(Rm.ravel())
            r = Rm.ravel()[ok]
            g = G.ravel()[ok]
            bs = sk.boot_regime(Rm, G, b, nb)
            up, dn = sk.g1(r[g == 1]), sk.g1(r[g == -1])
            lo, hi = sk.ci(bs["diff"])
            row = dict(
                h=h,
                regime=name,
                up_n=int((g == 1).sum()),
                dn_n=int((g == -1).sum()),
                up_g1=up,
                dn_g1=dn,
                diff=up - dn,
                diff_lo=lo,
                diff_hi=hi,
                diff_p=sk.pval(bs["diff"]),
                up_q10=sk.qskew(r[g == 1], 10),
                dn_q10=sk.qskew(r[g == -1], 10),
            )
            rows.append(row)
            print(
                f"{h:>4} {name:<12} up {up:+.3f} down {dn:+.3f} gap {row['diff']:+.3f} "
                f"[{lo:+.2f}, {hi:+.2f}] p={row['diff_p']:.3f}",
                flush=True,
            )
    pd.DataFrame(rows).to_csv(config.result_path("state_rule_sensitivity.csv"), index=False)


if __name__ == "__main__":
    main()
