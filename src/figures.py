"""Figures of the paper and online appendix.

  fig1   (a) moment skewness g1 of z by trend state and horizon; (b) uptrend-minus-downtrend
         Q10/90 gap under own-window BV and past-day RV scaling   <- skew_by_state.csv, scalings.csv
  fig2   skewness by decile of the past 45-day SNR                  <- snr_deciles.csv
  fig3   uptrend-minus-downtrend g1 gap by calendar year            <- gap_by_year.csv
  fig4   12-month rolling gap, g1 and Q10/90                        <- rolling_gap.csv
  figA1  sensitivity of the gap to look-back window and threshold  <- state_rule_sensitivity.csv
  figA2  gap in 8-hour windows by time of day                       <- mechanism_checks.csv

Reads the result files above from results/. Writes <name>.pdf and <name>.png to
results/figures/. Supports Figures 1-4, A1 and A2.

Usage: python src/figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

# Colours: up blue, down orange, range green (first three of a validated categorical palette).
C = {"up": "#2a78d6", "dn": "#eb6834", "rg": "#1baf7a"}
INK, MUTED, GRID = "#222220", "#6b6b66", "#e4e3df"
STATES = (("dn", "Downtrend"), ("rg", "Range"), ("up", "Uptrend"))
plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
    }
)


def read(name):
    return pd.read_csv(config.result_path(name))


def horizon_title(h):
    return f"{h.replace('h', '-hour')} windows"


def save(fig, name, tight=True):
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(config.FIGURE_DIR / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig1():
    B = read("skew_by_state.csv")
    B = B[B.coin == "PANEL"].set_index("h")
    S = read("scalings.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    hs = ["1h", "4h", "15h", "24h", "7d"]
    x = np.arange(len(hs))
    w = 0.26
    for j, (k, lab) in enumerate(STATES):
        v = B.loc[hs, f"{k}_g1"].to_numpy()
        err = np.vstack([v - B.loc[hs, f"{k}_lo"], B.loc[hs, f"{k}_hi"] - v])
        a.bar(
            x + (j - 1) * w,
            v,
            w * 0.92,
            color=C[k],
            label=lab,
            yerr=err,
            capsize=2.2,
            error_kw=dict(ecolor=INK, lw=0.7),
        )
    a.axhline(0, color=INK, lw=0.8)
    a.set_xticks(x)
    a.set_xticklabels(["1 h", "4 h", "15 h", "24 h", "1 w"])
    a.set_xlabel("Horizon")
    a.set_ylabel("Moment skewness $g_1$ of z")
    a.legend(ncol=3, loc="upper left", fontsize=8)
    a.set_title("(a) Skewness of z by trend state", fontsize=9, color=INK, loc="left")
    hs2 = ["1h", "4h", "15h", "24h"]
    x2 = np.arange(len(hs2))
    w2 = 0.34
    for j, (sc, lab, col) in enumerate(
        (("BV (baseline)", "Own-window BV", C["rg"]), ("ex-ante", "Past-day RV", C["up"]))
    ):
        d = S[(S.scaling == sc) & (S.measure == "q10")].set_index("h").loc[hs2]
        b.errorbar(
            x2 + (j - 0.5) * w2,
            d.gap,
            yerr=[d.gap - d.lo, d.hi - d.gap],
            fmt="o",
            color=col,
            ms=5,
            lw=1.6,
            capsize=2.5,
            label=lab,
        )
    b.axhline(0, color=INK, lw=0.8)
    b.set_xticks(x2)
    b.set_xticklabels(["1 h", "4 h", "15 h", "24 h"])
    b.set_xlabel("Horizon")
    b.set_ylabel("Uptrend − downtrend, Q10/90")
    b.legend(loc="upper left", fontsize=8)
    b.set_title("(b) Quantile-skewness gap by scaling", fontsize=9, color=INK, loc="left")
    save(fig, "fig1")


def fig2():
    D = read("snr_deciles.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    for ax, h in zip(axes, ["15h", "24h"]):
        d = D[D.h == h]
        ax.errorbar(
            d.decile,
            d.g1,
            yerr=[d.g1 - d.g1_lo, d.g1_hi - d.g1],
            color=C["up"],
            marker="o",
            ms=4,
            lw=2,
            capsize=2,
            label="$g_1$",
        )
        ax.errorbar(
            d.decile + 0.15,
            d.q10,
            yerr=[d.q10 - d.q10_lo, d.q10_hi - d.q10],
            color=C["dn"],
            marker="s",
            ms=4,
            lw=2,
            capsize=2,
            label="Q10/90",
        )
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Decile of past 45-day SNR")
        ax.set_title(horizon_title(h), fontsize=9, color=INK)
    axes[0].set_ylabel("Skewness of standardized returns")
    axes[0].legend(loc="upper left")
    save(fig, "fig2")


def fig3():
    Y = read("gap_by_year.csv")
    Y = Y[Y.year != "IM_test"].copy()
    Y["year"] = Y.year.astype(int)
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    w = 0.38
    for j, (h, col) in enumerate((("15h", C["up"]), ("24h", C["rg"]))):
        d = Y[Y.h == h]
        xpos = d.year + (j - 0.5) * w
        ax.bar(
            xpos,
            d.gap,
            w * 0.92,
            color=col,
            label=h.replace("h", "-hour"),
            yerr=[d.gap - d.lo, d.hi - d.gap],
            capsize=2,
            error_kw=dict(ecolor=INK, lw=0.8),
        )
        ax.scatter(xpos, d.gap, s=10, color=INK, zorder=3)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("Uptrend − downtrend $g_1$")
    ax.legend(ncol=2, loc="upper left")
    save(fig, "fig3")


def fig4():
    R = read("rolling_gap.csv")
    R["end"] = pd.to_datetime(R.end)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    for ax, h in zip(axes, ["15h", "24h"]):
        d = R[R.h == h]
        ax.plot(d.end, d.gap, color=C["up"], lw=1.8, label="$g_1$")
        ax.plot(d.end, d.gap_q10, color=C["dn"], lw=1.8, label="Q10/90")
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_title(horizon_title(h), fontsize=9, color=INK)
        ax.set_xlabel("End of 12-month window")
    axes[0].set_ylabel("Uptrend − downtrend skewness")
    axes[0].legend(loc="upper left", fontsize=8)
    save(fig, "fig4")


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def figA1():
    S = read("state_rule_sensitivity.csv")
    S = S[S.regime.str.startswith("snr")]
    S["W"] = S.regime.str.extract(r"snr(\d+)").astype(int)
    S["th"] = S.regime.str.extract(r"th([\d.]+)").astype(float)
    B = read("skew_by_state.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    vmax = 1.05 * S[S.h.isin(["15h", "24h"])]["diff"].max()
    for ax, h in zip(axes, ["15h", "24h"]):
        d = S[S.h == h].pivot(index="W", columns="th", values="diff")
        pv = S[S.h == h].pivot(index="W", columns="th", values="diff_p")
        base = B[(B.coin == "PANEL") & (B.h == h)].iloc[0]
        # The baseline cell (45 days, 0.75) shows the main estimate with 1,999 replications.
        d.loc[45, 0.75], pv.loc[45, 0.75] = base["diff"], base["diff_p"]
        im = ax.imshow(d.to_numpy(), cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                val, p = d.iat[i, j], pv.iat[i, j]
                ax.text(
                    j,
                    i,
                    f"{val:+.2f}{stars(p)}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if val > 0.6 * vmax else INK,
                )
        ax.set_xticks(range(d.shape[1]))
        ax.set_xticklabels([f"{t:g}" for t in d.columns])
        ax.set_yticks(range(d.shape[0]))
        ax.set_yticklabels(d.index)
        ax.set_xlabel("SNR threshold θ")
        ax.set_title(horizon_title(h), fontsize=9, color=INK)
        ax.grid(False)
    axes[0].set_ylabel("Look-back window (days)")
    fig.colorbar(im, ax=axes, shrink=0.85, label="Uptrend − downtrend $g_1$")
    # No tight_layout: it does not support the colorbar spanning both axes.
    save(fig, "figA1", tight=False)


def figA2():
    M = read("mechanism_checks.csv")
    d = M[M.check == "clock_gap"].copy()
    d["start"] = d.h.str.extract(r"@(\d+)").astype(int)
    d = d.sort_values("start")
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    ax.bar(
        d.start,
        d.est,
        3.4,
        color=C["up"],
        yerr=[d.est - d.lo, d.hi - d.est],
        capsize=2.5,
        error_kw=dict(ecolor=INK, lw=0.8),
    )
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(d.start)
    ax.set_xticklabels([f"{s:02d}–{(s + 8) % 24:02d}" for s in d.start])
    ax.set_xlabel("8-hour window (UTC)")
    ax.set_ylabel("Uptrend − downtrend $g_1$")
    save(fig, "figA2")


def main():
    for make in (fig1, fig2, fig3, fig4, figA1, figA2):
        make()
        print(f"{make.__name__} done", flush=True)


if __name__ == "__main__":
    main()
