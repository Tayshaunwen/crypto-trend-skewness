"""Tables of the paper and online appendix, and the numbers quoted in the text.

Every statistic has a single source file; where the same statistic appears twice, both places
use the same value (for example, the baseline g1 gap always comes from skew_by_state.csv,
1,999 replications). p-values are printed as "<0.001" or with three decimals; ***, **, * mark
significance at 1%, 5% and 10%.

Reads the result files of the other scripts from results/. Writes table_1.md ... table_8.md and
table_A1.md ... table_A12.md to results/tables/, and results/paper_numbers.json (the numbers
quoted in the text, keyed by name).
Supports Tables 1-8 and A1-A12.

Usage: python src/tables.py
"""

import json
import re

import numpy as np
import pandas as pd

import config

HS = ["1h", "4h", "15h", "24h", "7d"]
H3 = ["24h", "15h", "4h"]
HL = {"1h": "1 h", "4h": "4 h", "15h": "15 h", "24h": "24 h", "7d": "1 w"}
WORD = [
    "none",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "all fifteen",
]
VAR_TAILS = {0.01: "1% long", 0.05: "5% long", 0.95: "5% short", 0.99: "1% short"}
TABLE_NUMBER = {
    "T_sample": "1",
    "T_state": "2",
    "T_scalings": "3",
    "T_notis": "4",
    "T_robust": "5",
    "T_scores": "6",
    "T_var": "7",
    "T_payoff": "8",
    "A_coins": "A1",
    "A_rules": "A2",
    "A_phase": "A3",
    "A_coingaps": "A4",
    "A_joint15": "A5",
    "A_mc": "A6",
    "A_qr": "A7",
    "A_universe": "A8",
    "A_varfull": "A9",
    "A_payofffull": "A10",
    "A_funding": "A11",
    "A_tod": "A12",
}
NUM = {}  # numbers quoted in the text

# Shared inputs, filled by load_shared().
BP = BC = S = TL = M = None


def f(x, d=3):
    return f"{x:.{d}f}".replace("-", "−")


def f2(x):
    return f(x, 2)


def fs(x, d=3):
    return f"{x:+.{d}f}".replace("-", "−")


def star(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def pf(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def peq(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def cs(est, p, d=3):
    return f"{f(est, d)}{star(p)}"


def read(name, **kw):
    return pd.read_csv(config.result_path(name), **kw)


def table_path(key):
    return config.TABLE_DIR / f"table_{TABLE_NUMBER[key]}.md"


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows]) + "\n"


def write(key, title, body, note=""):
    """Write one table; [[T:key]] and [[A:key]] in the note become table numbers."""
    note = re.sub(r"\[\[[TA]:(\w+)\]\]", lambda m: TABLE_NUMBER[m.group(1)], note)
    text = f"**Table {TABLE_NUMBER[key]} {title}**\n\n{body}" + (f"\n*Note: {note}*\n" if note else "")
    table_path(key).write_text(text, encoding="utf-8")


def load_shared():
    global BP, BC, S, TL, M
    B = read("skew_by_state.csv")  # baseline, 1,999 replications
    BP = B[B.coin == "PANEL"].set_index("h")
    BC = B[B.coin != "PANEL"]
    S = read("scalings.csv").set_index(["h", "scaling", "measure"])  # 999 replications
    TL = read("tails.csv").set_index(["h", "scaling"])
    M = read("mechanism_checks.csv")


def mm(k, h):
    d = M[(M.check == k) & (M.h == h)]
    return None if d.empty else d.iloc[0]


def t_sample():
    D = read("describe_by_coin.csv")
    Sp = read("state_spells.csv")
    A = read("pooled_skewness.csv")
    A = A[A.coin == "PANEL"].set_index("h")
    rows = []
    for h in HS:
        d = D[D.h == h]
        rows.append(
            [
                HL[h],
                f"{int(d.n.sum()):,}",
                d.start.min(),
                f(d["mean"].mean()),
                f(d["sd"].mean()),
                f(A.at[h, "g1"]),
                f(d["kurt"].mean(), 2),
                f"{d['up'].mean():.0%} / {d['rg'].mean():.0%} / {d['dn'].mean():.0%}",
            ]
        )
        NUM[f"n_{h}"] = f"{int(d.n.sum()):,}"
    sp = Sp.groupby("state").agg(n=("n_spells", "sum"), days=("mean_days", "mean"))
    Tr = read("state_transitions.csv", index_col=0)
    NUM["stay_up"], NUM["stay_dn"] = f"{Tr.loc[1.0, '1.0']:.2f}", f"{Tr.loc[-1.0, '-1.0']:.2f}"
    NUM["spells_up"], NUM["spells_dn"] = str(int(sp.at["up", "n"])), str(int(sp.at["dn", "n"]))
    NUM["spell_days_up"], NUM["spell_days_dn"] = f"{sp.at['up', 'days']:.0f}", f"{sp.at['dn', 'days']:.0f}"
    write(
        "T_sample",
        "Main sample: standardized returns z and trend states",
        md(
            [
                "Horizon",
                "Windows",
                "First window",
                "Mean of z",
                "Std. dev. of z",
                "Skewness $g_1$ of z",
                "Kurtosis of z (raw)",
                "Up / range / down",
            ],
            rows,
        ),
        "z = r/√BV. Mean, standard deviation and kurtosis are averages across coins; skewness is computed "
        "on the pooled panel. "
        f"In the 24-hour panel there are {NUM['spells_up']} uptrend spells (mean length "
        f"{NUM['spell_days_up']} days) and {NUM['spells_dn']} downtrend spells "
        f"(mean {NUM['spell_days_dn']} days).",
    )


def t_state():
    rows = []
    for h in HS:
        p = BP.loc[h]
        c = BC[BC.h == h]
        rows.append(
            [
                HL[h],
                f"{f(p.up_g1)} [{f(p.up_lo, 2)}, {f(p.up_hi, 2)}]",
                f(p.rg_g1),
                f"{f(p.dn_g1)} [{f(p.dn_lo, 2)}, {f(p.dn_hi, 2)}]",
                f"{f(p['diff'])} [{f(p.diff_lo, 2)}, {f(p.diff_hi, 2)}]",
                pf(p.diff_p),
                f"{int((c.up_g1 > c.dn_g1).sum())}/{len(c)}",
            ]
        )
        NUM[f"up_{h}"], NUM[f"dn_{h}"] = f2(p.up_g1), f2(p.dn_g1)
        NUM[f"gap_{h}"], NUM[f"gap_p_{h}"] = f2(p["diff"]), peq(p.diff_p)
        NUM[f"coins_pos_{h}"] = str(int((c.up_g1 > c.dn_g1).sum()))
    write(
        "T_state",
        "Moment skewness of z by trend state",
        md(
            [
                "Horizon",
                "Uptrend [95% CI]",
                "Range",
                "Downtrend [95% CI]",
                "Up − down [95% CI]",
                "p",
                "Coins with up > down",
            ],
            rows,
        ),
        "z = r/√BV, ten-coin panel; states from the t-statistic of the past 45 daily returns with threshold "
        "±0.75. Intervals and p-values from a stationary bootstrap that resamples 45-day blocks of dates "
        "jointly across coins, 1,999 replications. Coin-level gaps with Holm-adjusted p-values are in "
        "Table [[A:A_coingaps]].",
    )


def t_scalings():
    rows = []
    lab = {
        "BV (baseline)": "Own-window BV (z)",
        "lagged BV": "Previous-window BV",
        "ex-ante": "Past-day RV",
        "unstandardized": "Constant",
    }
    for h in ["24h", "15h", "4h", "1h"]:
        for sc in ["BV (baseline)", "lagged BV", "ex-ante", "unstandardized"]:
            cells = [HL[h], lab[sc]]
            if sc == "BV (baseline)":
                cells.append(cs(BP.at[h, "diff"], BP.at[h, "diff_p"]))
            elif (h, sc) in TL.index:
                r = TL.loc[(h, sc)]
                cells.append(f"{cs(r.wins_gap, r.wins_p)}†")
            else:
                cells.append("–")
            for mn in ["bowley", "q10"]:
                if (h, sc, mn) in S.index:
                    r = S.loc[(h, sc, mn)]
                    c = cs(r.gap, r.p)
                    cells.append(f"{c} ‡" if (h, sc, mn) == ("24h", "ex-ante", "q10") else c)
                else:
                    cells.append("–")
            if (h, sc) in TL.index:
                cells.append(f"{TL.loc[(h, sc)].hill_upper:.1f} / {TL.loc[(h, sc)].hill_lower:.1f}")
            else:
                cells.append("–")
            rows.append(cells)
    write(
        "T_scalings",
        "Uptrend-minus-downtrend skewness gap under four scalings of returns",
        md(["Horizon", "Scaling", "Moment $g_1$", "Bowley", "Q10/90", "Hill index (upper / lower)"], rows),
        "own-window BV divides the return by the square root of its window's bipower variation "
        "(standardized returns z); previous-window BV by that of the previous window; past-day RV by the "
        "square root of the realized variance over the past 24 hours, rescaled to the window length; "
        "constant by the coin's full-sample standard deviation. "
        "The last three use no information from the window itself. ‡ primary test. Hill indices use the "
        "top 2% of absolute values, pooled across states. Because the tail index under the last three "
        "scalings is about 3, the moment column (†) reports the skewness of returns winsorized at the 0.5% "
        "and 99.5% quantiles. "
        "Stationary bootstrap with 45-day blocks, 999 replications (own-window BV moment column: 1,999, as "
        "in Table [[T:T_state]]); ***, **, * = 1%, 5%, 10%. At one hour only the quantile measure is "
        "computed for the alternative scalings and tail indices are not computed; the weekly horizon is "
        "omitted because of its small number of windows.",
    )
    r = S.loc[("24h", "ex-ante", "q10")]
    NUM["prim_gap"], NUM["prim_lo"], NUM["prim_hi"], NUM["prim_p"] = f(r.gap), f(r.lo), f(r.hi), peq(r.p)
    NUM["prim_up"], NUM["prim_dn"] = f(r.up), f(r.dn)
    NUM["prim_p_plain"] = pf(r.p)
    for h in ["24h", "15h"]:
        for sc, k in (
            ("ex-ante", "ex"),
            ("unstandardized", "raw"),
            ("lagged BV", "lag"),
            ("BV (baseline)", "bv"),
        ):
            NUM[f"q10_{k}_{h}"] = f2(S.loc[(h, sc, "q10")].gap)
    alt = TL.drop(index="BV (baseline)", level=1)[["hill_upper", "hill_lower"]]
    NUM["hill_alt_min"] = f"{min(alt.min()):.1f}"
    NUM["hill_alt_max"] = f"{max(alt.max()):.1f}"
    NUM["hill_bv_min"] = f"{min(TL.xs('BV (baseline)', level=1)[['hill_upper', 'hill_lower']].min()):.1f}"


def t_notis():
    Sm = read("smoothness.csv").set_index(["check", "h"])
    Tv = read("trend_vs_vol.csv")

    def sm(k, h, sig=True):
        r = Sm.loc[(k, h)]
        return cs(r.est, r.p) if sig and pd.notna(r.get("p")) else f(r.est)

    def me(k, h, sig=True):
        r = mm(k, h)
        return "–" if r is None else (cs(r.est, r.p) if sig and pd.notna(r.p) else f(r.est))

    def tv(check, h, vt=None):
        d = Tv[(Tv.h == h) & (Tv.check == check)]
        d = d if vt is None else d[d.vol_tercile == vt]
        r = d.iloc[0]
        return cs(r.g1, r.p)

    def pair(get, k_up, k_dn):
        return [f"{f(get(k_up, h))} / {f(get(k_dn, h))}" for h in H3]

    def sm_est(k, h):
        return Sm.loc[(k, h)].est

    def mm_est(k, h):
        return mm(k, h).est

    rows = [
        ["*States defined from other periods*", "", "", ""],
        ["States from the 45 days after the window"] + [me("expost_immediate", h) for h in H3],
        ["States from a 45-day span centered on the window"] + [me("expost_centered", h) for h in H3],
        ["States from 45 days starting 90 days after the window"] + [me("expost_plus90", h) for h in H3],
        ["*Path smoothness*", "", "", ""],
        ["Intra-window first-order autocorrelation, uptrend / downtrend"]
        + pair(sm_est, "ac1_mean_up", "ac1_mean_dn"),
        ["Gap within terciles of intra-window autocorrelation"]
        + [sm("gap_within_ac1_terciles", h) for h in H3],
        ["*Volatility*", "", "", ""],
        ["corr(r, log BV), uptrend / downtrend"] + pair(mm_est, "corr_r_logBV_up", "corr_r_logBV_dn"),
        ["Gap within terciles of the BV surprise"] + [me("gap_within_BVsurprise_terciles", h) for h in H3],
        ["High − low past-return tercile, low past volatility"]
        + [tv("ret_gap_within_vol", h, "low") for h in H3],
        ["High − low past-return tercile, middle past volatility"]
        + [tv("ret_gap_within_vol", h, "mid") for h in H3],
        ["High − low past-return tercile, high past volatility"]
        + [tv("ret_gap_within_vol", h, "high") for h in H3],
        ["*Jumps and events*", "", "", ""],
        ["Signed jump variation (RS⁺ − RS⁻)/RV, uptrend / downtrend"]
        + pair(mm_est, "signed_jump_mean_up", "signed_jump_mean_dn"),
        ["Gap excluding the top decile of windows by jump share"]
        + [me("gap_excluding_jump_windows", h) for h in H3],
        ["Gap excluding March 2020, May 2021, Terra–Luna and FTX weeks"]
        + [me("gap_excluding_event_weeks", h) for h in H3],
        ["*Drift*", "", "", ""],
        ["Gap after removing coin × state × month means of z"]
        + [tv("demean_coin_state_month", h) for h in H3],
    ]
    write(
        "T_notis",
        "Alternative explanations: uptrend-minus-downtrend gap in moment skewness of z",
        md(["", "24 h", "15 h", "4 h"], rows),
        "the baseline gaps are in Table [[T:T_state]]. The BV surprise is log BV minus its HAR forecast; RS⁺ "
        "and RS⁻ are realized semivariances of positive and negative intra-window returns. The "
        "return–volatility rows sort windows into terciles of the past 45-day cumulative return within "
        "terciles of the past 45-day volatility of daily returns. Jump share is (RV − BV)/RV. Stationary "
        "bootstrap with 45-day blocks, 999 replications; ***, **, * = 1%, 5%, 10%.",
    )
    for h in ["24h", "15h", "4h"]:
        NUM[f"xp_imm_{h}"] = f(mm("expost_immediate", h).est)
        NUM[f"xp_cen_{h}"] = f(mm("expost_centered", h).est)
        NUM[f"xp_90_{h}"] = f(mm("expost_plus90", h).est)
        NUM[f"ac_up_{h}"], NUM[f"ac_dn_{h}"] = f(sm_est("ac1_mean_up", h)), f(sm_est("ac1_mean_dn", h))
        NUM[f"ac_strat_{h}"] = f2(sm_est("gap_within_ac1_terciles", h))
        NUM[f"cbv_up_{h}"] = f2(mm("corr_r_logBV_up", h).est)
        NUM[f"cbv_dn_{h}"] = f2(mm("corr_r_logBV_dn", h).est)
        NUM[f"bvs_{h}"] = f2(mm("gap_within_BVsurprise_terciles", h).est)
        NUM[f"jump_ex_{h}"] = f2(mm("gap_excluding_jump_windows", h).est)
        NUM[f"event_ex_{h}"] = f2(mm("gap_excluding_event_weeks", h).est)
        NUM[f"dem_{h}"] = f2(Tv[(Tv.h == h) & (Tv.check == "demean_coin_state_month")].iloc[0].g1)
        d = Tv[(Tv.h == h) & (Tv.check == "ret_gap_within_vol")].set_index("vol_tercile")
        NUM[f"tv_low_{h}"], NUM[f"tv_mid_{h}"] = f2(d.at["low", "g1"]), f2(d.at["mid", "g1"])
        NUM[f"tv_high_{h}"] = f2(d.at["high", "g1"])
        NUM[f"tv_mid_p_{h}"] = peq(d.at["mid", "p"])


def t_robust():
    G2 = read("inference_checks.csv").set_index(["check", "h"])
    Y = read("year_clusters.csv").set_index(["check", "h"])
    I = read("year_clusters_all.csv").set_index("h")
    Q = read("quantile_regression.csv").set_index(["h", "stat"])
    G = read("robustness_subsamples.csv").set_index(["variant", "h"])
    U = read("point_in_time.csv")
    Ho = read("replication.csv").set_index("h")
    HJ = read("pooled_20coins.csv").set_index("h")
    HX = read("replication_primary.csv").set_index("h")
    Ph = read("start_hour.csv")
    u10 = U[U.topn == 10].set_index("h")

    def g2(k, h):
        r = G2.loc[(k, h)]
        return cs(r.est, r.p)

    def gr(v, h):
        r = G.loc[(v, h)]
        return cs(r["diff"], r.diff_p)

    def qr(s, h):
        r = Q.loc[(h, s)]
        return cs(r.est, r.p)

    def placebo(h):
        r = G2.loc[("shift_placebo_g1", h)]
        return f"{pf(r.p)} [{f(r.lo, 2)}, {f(r.hi, 2)}]"

    def jackknife(h):
        # In year_clusters.csv the jackknife's t and p columns hold its 2.5% and 97.5% percentiles.
        r = Y.loc[("grouped_jackknife_20pct", h)]
        return f"[{f(r.t)}, {f(r.p)}]"

    def im_years(h):
        return f"{int(I.at[h, 'n_pos'])}/{int(I.at[h, 'n_years'])} ({pf(I.at[h, 'p'])})"

    def im_halfyears(h):
        r = Y.loc[("IM_halfyear", h)]
        return f"{int(r.n_pos)}/{int(r.n_clusters)} ({pf(r.p)})"

    # Start hour: the 00:00 row is the baseline; the other five start hours come from start_hour.csv.
    ph = Ph[Ph.start != 0]
    phase_g1 = [BP.at["24h", "diff"]] + list(ph.g1_gap)
    phase_g1p = [BP.at["24h", "diff_p"]] + list(ph.g1_p)
    phase_q = [S.loc[("24h", "BV (baseline)", "q10")].gap] + list(ph.q10_gap)
    phase_qp = [S.loc[("24h", "BV (baseline)", "q10")].p] + list(ph.q10_p)
    phase_cell = (
        f"[{f(min(phase_g1), 2)}, {f(max(phase_g1), 2)}], {sum(p < 0.05 for p in phase_g1p)}/6 significant"
    )
    rows = [
        ["*Inference*", "", "", ""],
        ["Gap with 90-day bootstrap blocks"] + [g2("block90d_g1", h) for h in H3],
        ["Shifted-state placebo: p-value [95% range of placebo gaps]"] + [placebo(h) for h in H3],
        ["Calendar years with gap > 0, including 2026 (IM p)"] + [im_years(h) for h in H3],
        ["Half-years with gap > 0 (IM p)"] + [im_halfyears(h) for h in H3],
        ["Grouped jackknife over trend episodes, 95% range"] + [jackknife(h) for h in H3],
        ["Quantile regression, up − down $A_{10/90}$"] + [qr("QS10_diff", h) for h in H3],
        ["Quantile regression, up − down $A_{5/95}$"] + [qr("QS05_diff", h) for h in H3],
        ["*State definition and scaling*", "", "", ""],
        ["Adaptive (expanding-quantile) thresholds"] + [gr("adaptive", h) for h in H3],
        ["Standardized by realized instead of bipower variation"] + [gr("std_rv", h) for h in H3],
        ["24-hour window start shifted (six start hours): range of gaps", phase_cell, "–", "–"],
        ["*Subsamples*", "", "", ""],
        ["Excluding 2021"] + [gr("ex2021", h) for h in H3],
        ["From 2023"] + [gr("from2023", h) for h in H3],
        ["BTC and ETH only"] + [gr("majors", h) for h in H3],
        ["Eight altcoins"] + [gr("alts", h) for h in H3],
        ["*Other coins*", "", "", ""],
        ["Point-in-time top-10 universe, $g_1$"] + [cs(u10.at[h, "g1_gap"], u10.at[h, "g1_p"]) for h in H3],
        ["Point-in-time top-10 universe, Q10/90 of past-day-RV-scaled returns (primary statistic)"]
        + [cs(u10.at[h, "exq10_gap"], u10.at[h, "exq10_p"]) for h in H3],
        ["Replication sample (ten further coins), $g_1$"] + [cs(Ho.at[h, "gap"], Ho.at[h, "p"]) for h in H3],
        ["Replication sample, Q10/90 of z"] + [cs(Ho.at[h, "q10_gap"], Ho.at[h, "q10_p"]) for h in H3],
        ["Replication sample, Q10/90 of past-day-RV-scaled returns (primary statistic)"]
        + [cs(HX.at[h, "exq10_gap"], HX.at[h, "exq10_p"]) for h in H3],
        ["Main and replication samples pooled (20 coins), $g_1$"]
        + [cs(HJ.at[h, "comb_g1"], HJ.at[h, "comb_g1_p"]) for h in H3],
    ]
    write(
        "T_robust",
        "Robustness of the uptrend-minus-downtrend gap",
        md(["", "24 h", "15 h", "4 h"], rows),
        "unless stated otherwise, entries are gaps in the moment skewness of z with significance from a "
        "stationary bootstrap with a mean block length of 45 days and 999 replications; ***, **, * = 1%, 5%, "
        "10%. "
        "Placebo: share of 999 joint circular shifts (at least 90 days) of all coins' state series that "
        "produce a gap at least as large in absolute value. "
        "IM: Ibragimov and Müller (2010) t-test treating calendar years or half-years as clusters; the "
        "number of half-years with enough windows in both states is larger at four hours. "
        "Grouped jackknife: 200 draws, each dropping a random 20% of market trend episodes (maximal runs of "
        "the BTC 24-hour state). "
        "Quantile regressions report the difference in the unnormalized quantile asymmetry "
        "$A_\\tau = b_{1-\\tau} + b_\\tau - 2b_{0.5}$ (units of z), not the normalized quantile skewness of "
        "Table [[T:T_scalings]]; they include controls and coin fixed effects, with standard errors "
        "clustered by 45-day blocks (Table [[A:A_qr]]). "
        "Start-hour shift: 24-hour windows starting at 00:00, 04:00, …, 20:00 UTC (Table [[A:A_phase]]). "
        "Point-in-time universe: each month the ten USDT pairs with the highest trailing 30-day volume "
        "among 41 candidates, including later-delisted coins (Table [[A:A_universe]]).",
    )
    for h in H3:
        NUM[f"im_pos_{h}"], NUM[f"im_n_{h}"] = str(int(I.at[h, "n_pos"])), str(int(I.at[h, "n_years"]))
        NUM[f"im_p_{h}"] = peq(I.at[h, "p"])
        NUM[f"hy_p_{h}"] = peq(Y.loc[("IM_halfyear", h)].p)
        NUM[f"gj_lo_{h}"] = f2(Y.loc[("grouped_jackknife_20pct", h)].t)
        NUM[f"gj_hi_{h}"] = f2(Y.loc[("grouped_jackknife_20pct", h)].p)
        NUM[f"qr10_{h}"], NUM[f"qr05_{h}"] = f2(Q.loc[(h, "QS10_diff")].est), f2(Q.loc[(h, "QS05_diff")].est)
        NUM[f"qr10_p_{h}"] = peq(Q.loc[(h, "QS10_diff")].p)
        for v in ("from2023", "majors", "ex2021", "alts"):
            NUM[f"{v}_{h}"] = f2(G.loc[(v, h)]["diff"])
            NUM[f"{v}_p_{h}"] = peq(G.loc[(v, h)].diff_p)
        NUM[f"pit_g1_{h}"], NUM[f"pit_ex_{h}"] = f2(u10.at[h, "g1_gap"]), f2(u10.at[h, "exq10_gap"])
        NUM[f"pit_ex_p_{h}"] = peq(u10.at[h, "exq10_p"])
        NUM[f"rep_g1_{h}"], NUM[f"rep_q_{h}"] = f2(Ho.at[h, "gap"]), f2(Ho.at[h, "q10_gap"])
        NUM[f"rep_g1_p_{h}"], NUM[f"rep_q_p_{h}"] = peq(Ho.at[h, "p"]), peq(Ho.at[h, "q10_p"])
        NUM[f"rep_ex_{h}"], NUM[f"rep_ex_p_{h}"] = f(HX.at[h, "exq10_gap"]), peq(HX.at[h, "exq10_p"])
        NUM[f"pool_g1_{h}"], NUM[f"pool_g1_p_{h}"] = f2(HJ.at[h, "comb_g1"]), peq(HJ.at[h, "comb_g1_p"])
        NUM[f"pool_q_{h}"], NUM[f"pool_q_p_{h}"] = f2(HJ.at[h, "comb_q10"]), peq(HJ.at[h, "comb_q10_p"])
    episodes = read("episode_jackknife.csv").set_index("h").loc["24h", ["n_episodes", "n_up_ep", "n_dn_ep"]]
    NUM["n_episodes"], NUM["n_ep_up"], NUM["n_ep_dn"] = [str(int(x)) for x in episodes]
    NUM["state_corr"] = f2(HJ.at["24h", "state_corr"])
    NUM["pit_coins10"] = str(int(u10.coins.max()))
    NUM["pit_coins20"] = str(int(U[U.topn == 20].coins.max()))
    NUM["phase_q_pos"] = str(sum(x > 0 for x in phase_q))
    NUM["phase_q_sig"] = str(sum(p < 0.05 for p in phase_qp))
    NUM["phase_g_sig"] = str(sum(p < 0.05 for p in phase_g1p))
    NUM["phase_g_lo"], NUM["phase_g_hi"] = f2(min(phase_g1)), f2(max(phase_g1))
    NUM["rep_pos_24h"], NUM["rep_pos_1h"] = (
        str(int(Ho.at["24h", "coins_pos"])),
        str(int(Ho.at["1h", "coins_pos"])),
    )
    NUM["rep_g1_1h"], NUM["rep_q_1h"] = f2(Ho.at["1h", "gap"]), f2(Ho.at["1h", "q10_gap"])

    # Appendix: point-in-time universe, replication sample and pooled panel in full.
    rows = []
    for topn in (10, 20):
        for h in H3:
            r = U[(U.topn == topn) & (U.h == h)].iloc[0]
            rows.append(
                [
                    f"Point-in-time top {topn}",
                    HL[h],
                    f"{int(r.n):,}",
                    str(int(r.coins)),
                    cs(r.g1_gap, r.g1_p),
                    cs(r.q10_gap, r.q10_p),
                    cs(r.exq10_gap, r.exq10_p),
                ]
            )
    for h in ["24h", "15h", "4h", "1h"]:
        r = Ho.loc[h]
        primary = cs(HX.at[h, "exq10_gap"], HX.at[h, "exq10_p"]) if h in HX.index else "–"
        rows.append(
            [
                "Replication sample",
                HL[h],
                f"{int(r.n):,}",
                f"10 ({int(r.coins_pos)} with gap > 0)",
                cs(r.gap, r.p),
                cs(r.q10_gap, r.q10_p),
                primary,
            ]
        )
    for h in ["24h", "15h", "4h", "1h"]:
        r = HJ.loc[h]
        rows.append(
            [
                "Main + replication pooled",
                HL[h],
                f"{int(r.n):,}",
                "20",
                cs(r.comb_g1, r.comb_g1_p),
                cs(r.comb_q10, r.comb_q10_p),
                "–",
            ]
        )
    write(
        "A_universe",
        "Other coins: point-in-time universe, replication sample and pooled panel",
        md(
            [
                "Sample",
                "Horizon",
                "Windows",
                "Coins",
                "Moment $g_1$ of z",
                "Q10/90 of z",
                "Q10/90, past-day RV",
            ],
            rows,
        ),
        "Q10/90, past-day RV is the primary statistic. The 41 candidates are the 20 coins of the main and "
        "replication samples and 21 further USDT pairs that were among the largest by trading volume at "
        "some time since 2017 (LUNA, FTT, MATIC, UNI, FIL, ICP, NEAR, SHIB, XMR, NEO, IOTA, DASH, ZEC, ONT, "
        "QTUM, AAVE, APT, ARB, OP, SUI, WAVES), several of which were later delisted or collapsed. "
        "Each month the 10 (or 20) candidates with the highest trailing 30-day quote volume are admitted; "
        "only windows of admitted coins in that month are used. "
        "The replication coins (TRX, XLM, ETC, BCH, EOS, ATOM, XTZ, VET, ALGO, DOT) were analyzed before "
        "the point-in-time universe was built; the average correlation of their state indicators with "
        f"those of the main sample is {NUM['state_corr']}. Stationary bootstrap with 45-day blocks, 999 "
        "replications.",
    )
    rows = [
        ["00:00 (baseline)", cs(BP.at["24h", "diff"], BP.at["24h", "diff_p"]), cs(phase_q[0], phase_qp[0])]
    ]
    rows += [
        [f"{int(r.start):02d}:00", cs(r.g1_gap, r.g1_p), cs(r.q10_gap, r.q10_p)] for r in ph.itertuples()
    ]
    write(
        "A_phase",
        "24-hour windows with different start hours (UTC)",
        md(["Start", "Moment $g_1$ of z", "Q10/90 of z"], rows),
        "the 00:00 row repeats the baseline (Tables [[T:T_state]] and [[T:T_scalings]]); other rows use 999 "
        "bootstrap replications with 45-day blocks.",
    )


def t_scores():
    O = read("oos_scores.csv")
    PW = read("oos_power.csv").set_index("h")
    Sc = read("mloc_scores.csv").set_index(["h", "metric"])
    m1p = O[(O.window == "expanding") & (O.base == "M0s")]

    def o(h, mt):
        d = m1p[(m1p.h == h) & (m1p.metric == mt)].iloc[0]
        return f"{fs(d.rel * 100)}% ({fs(d.dm_t, 2)})"

    def s(h, mt):
        if (h, mt) not in Sc.index:
            return "–"
        d = Sc.loc[(h, mt)]
        return f"{fs(d.rel * 100)}% ({fs(d.dm_t, 2)})"

    rows = [
        ["*M1p − M0s (state-dependent asymmetry only)*", "", ""],
        ["Log score"] + [o(h, "LS") for h in ["24h", "15h"]],
        ["Tail-weighted CRPS"] + [o(h, "twCRPS") for h in ["24h", "15h"]],
        ["Approximate power of the log-score test"] + [f2(PW.at[h, "approx_power"]) for h in ["24h", "15h"]],
        ["*Mloc+λ − Mloc (asymmetry added to state-dependent location and scale)*", "", ""],
        ["Log score"] + [s(h, "LS") for h in ["24h", "15h"]],
        ["Tail-weighted CRPS"] + [s(h, "twCRPS") for h in ["24h", "15h"]],
    ]
    for t in (0.01, 0.05, 0.95, 0.99):
        rows.append([f"Quantile loss, {VAR_TAILS[t]} VaR"] + [s(h, f"QL{t}") for h in ["24h", "15h"]])
    write(
        "T_scores",
        "Out-of-sample density and quantile forecasts of HAR-scaled returns: relative loss differences",
        md(["", "24 h", "15 h"], rows),
        "ẑ = r/σ̂ with σ̂ a real-time HAR forecast of realized volatility; monthly re-estimation on an "
        "expanding window from January 2019. Entries are relative differences in mean loss (negative favors "
        "the richer model) with Diebold–Mariano t-statistics (Newey–West) in parentheses. "
        "Power: probability that the log-score test detects the gain implied by the estimated M1p asymmetry, "
        "at the 5% level.",
    )
    for h in ["24h", "15h"]:
        NUM[f"pow_{h}"] = f2(PW.at[h, "approx_power"])
        NUM[f"ls_m1p_t_{h}"] = fs(m1p[(m1p.h == h) & (m1p.metric == "LS")].iloc[0].dm_t, 2)
        for mt in ("LS", "twCRPS", "QL0.01", "QL0.05", "QL0.95", "QL0.99"):
            if (h, mt) in Sc.index:
                NUM[f"ml_{mt}_t_{h}"] = fs(Sc.loc[(h, mt)].dm_t, 2)
    ql = Sc[Sc.index.get_level_values(1).str.startswith("QL")]
    NUM["ql_maxabs_t"] = f2(ql.dm_t.abs().max())
    r = ql.dm_t.abs().idxmax()
    NUM["ql_maxabs_where"] = f"{HL[r[0]]}, {VAR_TAILS[float(r[1][2:])]}"
    O2 = O[(O.base == "M0s")]
    NUM["m1p_n_cmp"] = str(len(O2))
    NUM["m1p_n_sig"] = str(int((O2.dm_t.abs() > 1.96).sum()))


def t_var():
    Vm = read("var_backtests.csv")
    Mo = read("state_moments.csv").set_index(["h", "state"])

    def g(h, mn, t, st):
        return Vm[(Vm.h == h) & (Vm.model == mn) & (Vm.tau == t) & (Vm.state == st)].iloc[0]

    def cell(h, mn, t):
        return (
            f"{g(h, mn, t, 'up').hit * 100:.2f} / {g(h, mn, t, 'dn').hit * 100:.2f} "
            f"({f(g(h, mn, t, 'tests').dq_state_p, 2)})"
        )

    def rows_for(taus):
        return [
            [HL[h], VAR_TAILS[t]] + [cell(h, mn, t) for mn in ["M0s", "M1p", "Mloc", "Mloc+λ"]]
            for h in ["24h", "15h"]
            for t in taus
        ]

    hdr = ["Horizon", "VaR", "M0s", "M1p", "Mloc", "Mloc+λ"]
    note = (
        "out-of-sample VaR of HAR-scaled returns from skewed-t models: M0s unconditional; M1p "
        "state-dependent asymmetry; Mloc state-dependent location and scale; Mloc+λ state-dependent "
        "location, scale and asymmetry. "
        "Entries are breach rates (%) after uptrends / after downtrends and, in parentheses, the p-value of "
        "a Wald test that the uptrend and downtrend dummies are zero in a regression of breaches minus the "
        "nominal rate on a constant, the two dummies and the lagged breach, with standard errors clustered "
        "by 45-day blocks."
    )
    write(
        "T_var",
        "Value-at-Risk breach rates by trend state (5% level)",
        md(hdr, rows_for([0.05, 0.95])),
        note + " The 1% levels are in Table [[A:A_varfull]].",
    )
    write(
        "A_varfull",
        "Value-at-Risk breach rates by trend state, all four tails",
        md(hdr, rows_for([0.01, 0.05, 0.95, 0.99])),
        note,
    )
    for h in ["24h", "15h"]:
        for mn, k in (("M0s", "m0"), ("M1p", "m1"), ("Mloc", "ml"), ("Mloc+λ", "mll")):
            for t, tk in ((0.95, "s5"), (0.05, "l5"), (0.01, "l1"), (0.99, "s1")):
                NUM[f"v_{k}_{tk}_up_{h}"] = f"{g(h, mn, t, 'up').hit * 100:.2f}%"
                NUM[f"v_{k}_{tk}_dn_{h}"] = f"{g(h, mn, t, 'dn').hit * 100:.2f}%"
                p = g(h, mn, t, "tests").dq_state_p
                NUM[f"v_{k}_{tk}_dq_{h}"] = "p < 0.01" if p < 0.01 else f"p = {p:.2f}"
    NUM["mz_up"], NUM["mz_dn"] = (
        fs(Mo.loc[("24h", "up"), "mean_zhat"]),
        fs(Mo.loc[("24h", "dn"), "mean_zhat"]),
    )
    NUM["vz_up"], NUM["vz_dn"] = f2(Mo.loc[("24h", "up"), "var_zhat"]), f2(Mo.loc[("24h", "dn"), "var_zhat"])


def t_payoff():
    Pf = read("payoff_ratios.csv")

    def g(h, k, st):
        return Pf[(Pf.h == h) & (Pf.strike == k) & (Pf.state == st)].iloc[0]

    def ratio_ci(h, k, st):
        r = g(h, k, st)
        return f"{r.payoff_ratio:.2f} [{r.lo:.2f}, {r.hi:.2f}]"

    def rows_for(ks):
        rr = []
        for pre, lab in (("", "Raw ẑ"), ("resid_", "Location and scale removed")):
            for h in ["24h", "15h"]:
                for k in ks:
                    d = g(h, k, f"{pre}up_minus_dn")
                    rr.append(
                        [lab, HL[h], f"±{k:g}"]
                        + [ratio_ci(h, k, pre + st) for st in ("up", "rg", "dn")]
                        + [f"{d.payoff_ratio:.2f} ({peq(d.p)})"]
                    )
        return rr

    hdr = ["Returns", "Horizon", "Strikes", "Uptrend", "Range", "Downtrend", "Up − down"]
    note = (
        "for each window from January 2019, ẑ = r/σ̂ with σ̂ a real-time HAR forecast. Entries are "
        "E[(x − k)⁺] / E[(−k − x)⁺], the ratio of average realized payoffs of symmetric out-of-the-money "
        "calls and puts on x under the physical measure (not option prices), with strikes k in units of x "
        "and 95% intervals from a circular block bootstrap over dates. "
        "Raw ẑ: x = ẑ. Location and scale removed: x = (ẑ − μ̂_s)/σ̂_s, the residual of the state-dependent "
        "location–scale model Mloc, re-estimated monthly, so that only differences in shape remain."
    )
    write(
        "T_payoff",
        "Realized payoff ratios of symmetric out-of-the-money calls and puts (strikes ±1.28)",
        md(hdr, rows_for([1.28])),
        note + " Strikes of ±2 are in Table [[A:A_payofffull]].",
    )
    write("A_payofffull", "Realized payoff ratios with strikes of ±2", md(hdr, rows_for([2.0])), note)
    for h in ["24h", "15h"]:
        for pre, k in (("", "raw"), ("resid_", "res")):
            NUM[f"po_{k}_up_{h}"] = f2(g(h, 1.28, pre + "up").payoff_ratio)
            NUM[f"po_{k}_dn_{h}"] = f2(g(h, 1.28, pre + "dn").payoff_ratio)
            NUM[f"po_{k}_p_{h}"] = peq(g(h, 1.28, pre + "up_minus_dn").p)
            NUM[f"po_{k}_lo_up_{h}"] = f2(g(h, 1.28, pre + "up").lo)


def a_coins():
    D = read("describe_by_coin.csv")
    d = D[D.h == "24h"]
    rows = [
        [
            r.coin,
            r.start,
            f"{int(r.n):,}",
            f(r["mean"]),
            f(r["sd"]),
            f(r["skew"]),
            f(r["kurt"], 2),
            f"{r.up:.0%} / {r.rg:.0%} / {r.dn:.0%}",
        ]
        for _, r in d.sort_values("coin").iterrows()
    ]
    write(
        "A_coins",
        "Coin-level description of 24-hour standardized returns",
        md(
            [
                "Coin",
                "First window",
                "Windows",
                "Mean",
                "Std. dev.",
                "Skewness",
                "Kurtosis",
                "Up / range / down",
            ],
            rows,
        ),
    )


def a_rules():
    S2 = read("state_rule_sensitivity.csv")
    rules = (
        ("sgn30", "Sign of past 30-day return"),
        ("sgn60", "Sign of past 60-day return"),
        ("sgn90", "Sign of past 90-day return"),
        ("ma30", "Price above 30-day average"),
        ("ma60", "Price above 60-day average"),
    )
    rows = []
    for rg, lab in rules:
        cells = []
        for h in ["24h", "15h", "4h", "7d"]:
            r = S2[(S2.h == h) & (S2.regime == rg)].iloc[0]
            cells.append(cs(r["diff"], r.diff_p))
        rows.append([lab] + cells)
    write(
        "A_rules",
        "Gap in moment skewness of z under two-state sign and moving-average rules",
        md(["Rule", "24 h", "15 h", "4 h", "1 w"], rows),
        "the gap is up-state minus down-state skewness. Stationary bootstrap with 45-day blocks, 999 "
        "replications.",
    )


def a_coingaps():
    rows = []
    for c in sorted(BC.coin.unique()):
        cells = []
        for h in HS:
            r = BC[(BC.coin == c) & (BC.h == h)].iloc[0]
            cells.append(f"{f(r['diff'])} ({r['diff_p']:.3f}; {r['diff_p_holm']:.3f})")
        rows.append([c] + cells)
    write(
        "A_coingaps",
        "Coin-level gaps in moment skewness of z (raw p; Holm-adjusted p within horizon)",
        md(["Coin", "1 h", "4 h", "15 h", "24 h", "1 w"], rows),
        "stationary bootstrap with 45-day blocks, 1,999 replications.",
    )
    for h in ["24h", "15h"]:
        c = BC[BC.h == h]
        NUM[f"coins_holm_{h}"] = str(int((c.diff_p_holm < 0.05).sum()))


def a_joint15():
    if not config.result_path("joint_test.csv").exists():
        return
    J = read("joint_test.csv")
    lab = {
        ("bv", "g1"): "Moment $g_1$ of z",
        ("bv", "q10"): "Q10/90 of z",
        ("exante", "q10"): "Q10/90, past-day RV",
    }

    def row(h, sc, mn):
        return J[(J.h == h) & (J.scaling == sc) & (J.measure == mn)].iloc[0]

    rows = []
    for (sc, mn), nm in lab.items():
        for h in HS:
            r = row(h, sc, mn)
            c = f(r.gap)
            rows.append(
                [
                    nm if h == "1h" else "",
                    HL[h],
                    f"{c} ‡" if (h, sc, mn) == ("24h", "exante", "q10") else c,
                    f2(r.t),
                    pf(r.p_raw),
                    pf(r.p_rw),
                ]
            )
    write(
        "A_joint15",
        "Joint inference over fifteen statistics (five horizons × three measures)",
        md(["Measure", "Horizon", "Gap", "Studentized t", "Unadjusted p", "Romano–Wolf p"], rows),
        "the fifteen gaps are resampled jointly with the same 45-day blocks of dates (stationary bootstrap, "
        "1,999 replications). Both p-values are computed from the same studentized bootstrap distribution, so "
        "they can differ from the percentile p-values reported elsewhere. For the weekly moment gap (p = 0.063 "
        "in Table [[T:T_state]]) and the 24-hour quantile gap of z (p below 0.01 in Table [[T:T_scalings]]) the "
        "difference crosses a conventional threshold; the Romano–Wolf step-down p-value controls the "
        "family-wise error rate. ‡ primary test.",
    )
    pr = row("24h", "exante", "q10")
    NUM["j15_n05"] = WORD[int((J.p_rw < 0.05).sum())]
    NUM["j15_n10"] = WORD[int((J.p_rw < 0.10).sum())]
    NUM["j15_prim_raw"], NUM["j15_prim_rw"] = pf(pr.p_raw), pf(pr.p_rw)
    for (sc, mn), k in ((("bv", "g1"), "g1"), (("bv", "q10"), "q"), (("exante", "q10"), "ex")):
        for h in HS:
            NUM[f"j15_{k}_rw_{h}"] = peq(row(h, sc, mn).p_rw)
    NUM["j15_15h_max"] = peq(J[J.h == "15h"].p_rw.max())
    NUM["j15_sig_list"] = "; ".join(
        f"{HL[r.h]} {lab[(r.scaling, r.measure)].replace('$g_1$', 'g1')}"
        for r in J[J.p_rw < 0.05].itertuples()
    )


def a_mc():
    desc = {
        "nodrift": "Symmetric returns, no drift",
        "drift": "State-dependent drift calibrated to the data",
        "asymvol": "Volatility responds to within-window return, same in all states",
        "statevol": "Volatility response whose sign follows the state",
    }
    rows = []
    mc = {}
    for sc in ("nodrift", "drift", "asymvol", "statevol"):
        d = read(f"mc_{sc}.csv")
        mc[sc] = d
        rows.append(
            [
                desc[sc],
                f"{len(d):,}",
                fs(d.gap.mean()),
                f(d.gap.std()),
                pct(np.mean(d.p < 0.05)),
                pct(np.mean(d.p < 0.01)),
            ]
        )
    # Size-adjusted 5% critical value: 5th percentile of the simulated p-values without drift.
    crit = float(np.percentile(mc["nodrift"].p, 5))
    write(
        "A_mc",
        "Monte Carlo size of the bootstrap test for the 24-hour gap in moment skewness of z",
        md(
            [
                "Design",
                "Replications",
                "Mean simulated gap",
                "Std. dev. of gap",
                "Rejection at 5%",
                "Rejection at 1%",
            ],
            rows,
        ),
        "ten coins, 3,200 days of 288 five-minute returns with a common factor, Student-t(6) innovations, "
        "symmetric jumps and stochastic daily volatility responding symmetrically to past absolute returns; "
        "trend states, BV and z are computed from the simulated paths as in the data. Each replication uses "
        "199 bootstrap draws with 45-day blocks. The volatility-feedback designs use a response coefficient "
        "that reproduces the observed sign pattern of corr(r, log BV). "
        f"The 5th percentile of simulated p-values without drift, the size-adjusted 5% critical value, "
        f"is {f(crit)}.",
    )
    NUM["mc_crit"] = f(crit)
    NUM["mc_nsizeadj"] = WORD[int((BP.diff_p < crit).sum())]
    for sc in mc:
        NUM[f"mc_rej_{sc}"] = pct(np.mean(mc[sc].p < 0.05))
        NUM[f"mc_gap_{sc}"] = f(mc[sc].gap.mean())
        NUM[f"mc_n_{sc}"] = f"{len(mc[sc]):,}"


def a_qr():
    Q = read("quantile_regression.csv").set_index(["h", "stat"])
    Qn = read("quantile_regression_noctrl.csv").set_index(["h", "stat"])
    rows = [
        [HL[h]]
        + [cs(Q.loc[(h, s)].est, Q.loc[(h, s)].p) for s in ("QS10_up", "QS10_dn", "QS10_diff", "QS05_diff")]
        + [cs(Qn.loc[(h, "QS10_diff")].est, Qn.loc[(h, "QS10_diff")].p)]
        for h in H3
    ]
    n = {h: f"{int(Q.loc[(h, 'QS10_up')].n):,}" for h in H3}
    write(
        "A_qr",
        "Panel quantile regressions: state effects on quantile asymmetry "
        "$A_\\tau = b_{1-\\tau} + b_\\tau - 2b_{0.5}$ (units of z)",
        md(
            [
                "Horizon",
                "$A^{up}$, 10/90",
                "$A^{dn}$, 10/90",
                "Up − down, 10/90",
                "Up − down, 5/95",
                "Up − down, 10/90, no controls",
            ],
            rows,
        ),
        "$Q_\\tau(z_{it}) = a_\\tau + b^{up}_\\tau 1^{up}_{it} + b^{dn}_\\tau 1^{dn}_{it} + \\gamma_\\tau' C_{it} "
        "+ \\alpha_{i,\\tau}$ with the range state omitted. Controls C: lagged log BV, log realized variance "
        "over the past 24 hours and 7 days, relative volume, past 24-hour return, lagged z and a weekend "
        "dummy; coin fixed effects. "
        "The covariance across the three quantile regressions is estimated jointly with cluster-robust "
        "sandwich formulas (Parente & Santos Silva, 2016) over clusters of 45 consecutive days; a pairs "
        "cluster bootstrap gives standard errors between 0.97 and 1.18 times the analytical ones. "
        f"N = {n['24h']} (24 h), {n['15h']} (15 h), {n['4h']} (4 h). The 1-hour and 1-week horizons are "
        "omitted for computational reasons and because of too few observations per state, respectively.",
    )


def a_funding():
    def cell(k, h):
        r = mm(k, h)
        return cs(r.est, r.p)

    rows = [
        ["Gap, low pre-window funding rate"] + [cell("gap_low_funding", h) for h in H3],
        ["Gap, high pre-window funding rate"] + [cell("gap_high_funding", h) for h in H3],
        ["Uptrend skewness, low − high funding"]
        + [cell("uptrend_skew_low_minus_high_funding", h) for h in H3],
    ]
    write(
        "A_funding",
        "Perpetual-futures funding rates and the gap in moment skewness of z",
        md(["", "24 h", "15 h", "4 h"], rows),
        "windows are split at the median of the coin's average funding rate over the seven days before the "
        "window; sample from 2020. Stationary bootstrap with 45-day blocks, 999 replications.",
    )
    for h in H3:
        r = mm("uptrend_skew_low_minus_high_funding", h)
        NUM[f"fund_{h}"], NUM[f"fund_p_{h}"] = fs(r.est, 2), peq(r.p)


def a_tod():
    tod = read("time_of_day.csv").set_index(["check", "h"])
    ck = M[M.check == "clock_gap"].copy()
    ck["s"] = ck.h.str.extract(r"@(\d+)").astype(int)
    ck = ck.sort_values("s")
    dd = mm("difference_in_gaps", M[M.check == "difference_in_gaps"].h.iloc[0])
    rows = [
        [
            "08:00–17:00 (9 h)",
            f(tod.loc[("C_pooled_g1", "9h"), "est"]),
            f(tod.loc[("C_g1_up", "9h"), "est"]),
            f(tod.loc[("C_g1_dn", "9h"), "est"]),
            cs(tod.loc[("C_gap", "9h"), "est"], tod.loc[("C_gap", "9h"), "p"]),
        ]
    ]
    rows += [
        [f"{s:02d}:00–{(s + 8) % 24:02d}:00 (8 h)", f(g), "–", "–", cs(e, pv)]
        for s, g, e, pv in zip(ck.s, ck.g1_all, ck.est, ck.p)
    ]
    Ck = read("nested_daytime.csv")
    Ck = Ck[Ck.measure == "q10"].set_index("window")
    rows += [
        [
            f"{w}, Q10/90 skewness",
            "–",
            f(Ck.at[k, "up"]),
            f(Ck.at[k, "dn"]),
            cs(Ck.at[k, "gap"], Ck.at[k, "p"]),
        ]
        for k, w in (("8h_08_16", "08:00–16:00"), ("9h_08_17", "08:00–17:00"))
    ]
    write(
        "A_tod",
        "Time of day (exploratory): skewness of z in intraday windows (UTC)",
        md(["Window", "All states, $g_1$", "Uptrend", "Downtrend", "Up − down"], rows),
        "moment skewness unless stated. The 15-hour (17:00–08:00) and 24-hour windows are in Table "
        "[[T:T_state]]. The difference between the 17:00–08:00 and 08:00–17:00 gaps in moment skewness is "
        f"{f(dd.est)} ({peq(dd.p)}), computed on common dates. "
        "The 08:00–16:00 and 08:00–17:00 windows have very different moment-skewness gaps but almost "
        "identical quantile gaps, because a few windows with large standardized returns enter one but not "
        "the other. Stationary bootstrap with 45-day blocks, 999 replications.",
    )
    b1, b5 = mm("gap_bv_1min_sampling", "1h"), mm("gap_bv_5min_sampling", "1h")
    if b1 is not None and b5 is not None:
        NUM["bv1h_1m"], NUM["bv1h_5m"] = f2(b1.est), f2(b5.est)
        NUM["bv1h_5m_p"] = peq(b5.p)


def figure_numbers():
    """Numbers quoted with Figures 2-4: year 2026 gap, last rolling gap, SNR deciles."""
    Y = read("gap_by_year.csv")
    Y = Y[Y.year != "IM_test"]
    for h in ["24h", "15h"]:
        d = Y[Y.h == h]
        NUM[f"y2026_{h}"] = f(float(d[d.year.astype(str) == "2026"].gap.iloc[0]), 3)
    Ro = read("rolling_gap.csv")
    Ro["end"] = pd.to_datetime(Ro.end)
    NUM["roll_last_q_24h"] = f2(Ro[Ro.h == "24h"].gap_q10.iloc[-1])
    NUM["roll_last_g_24h"] = f2(Ro[Ro.h == "24h"].gap.iloc[-1])
    NUM["roll_end"] = Ro.end.max().strftime("%B %Y")
    D = read("snr_deciles.csv").set_index(["h", "decile"])
    for h in ["24h", "15h"]:
        NUM[f"dec10_g1_{h}"], NUM[f"dec10_q_{h}"] = f2(D.loc[(h, 10)].g1), f2(D.loc[(h, 10)].q10)
        mid = [D.loc[(h, k)].g1 for k in range(1, 9)]
        NUM[f"dec_mid_g1_{h}"] = f"{min(mid):.2f}–{max(mid):.2f}".replace("-0", "−0")


def main():
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    load_shared()
    for make in (
        t_sample,
        t_state,
        t_scalings,
        t_notis,
        t_robust,
        t_scores,
        t_var,
        t_payoff,
        a_coins,
        a_rules,
        a_coingaps,
        a_joint15,
        a_mc,
        a_qr,
        a_funding,
        a_tod,
        figure_numbers,
    ):
        make()
    config.result_path("paper_numbers.json").write_text(
        json.dumps(NUM, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"{len(list(config.TABLE_DIR.glob('table_*.md')))} tables, {len(NUM)} numbers")


if __name__ == "__main__":
    main()
