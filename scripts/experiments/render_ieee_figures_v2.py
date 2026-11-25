"""Render publication-grade IEEE Transactions (TKDE/TEVC) double-column vector figures on lab.

Enforces strict IEEE graphic standards:
- Exact IEEEtran textwidth (7.16 in) and columnwidth (3.48 in)
- STIX serif typography matching Times/Computer Modern journal body text
- Colorblind-safe AND grayscale-distinguishable curves (distinct line styles + sparse markers)
- Zero raw code identifiers or underscores in legends/axes (SP-PD, SP-PD-Num, piBO, etc.)
- Direct callout annotations on critical scientific phenomena
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter

EXP_ROOT = Path("/home/sunxie/paper-experiments-20261124")
OLD_CLIN = Path("/home/sunxie/clinical-prospective-20261123.cjUGun3X")
NEW_CLIN = Path("/home/sunxie/clinical-extension-20261124")
OUT_DIR = Path("/home/sunxie/paper-archive-20261124/ieee_draft/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EARLIER_SIX = ("brf", "cad", "gbsg", "thor", "thyroid", "voice")
NEW_SIX = ("whas", "colontrial", "actg", "ercp", "opttrial", "gargle")
ALL_12 = EARLIER_SIX + NEW_SIX

# Strict IEEE Transactions rcParams (STIX serif + inward ticks + subtle dashed grid)
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.4,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "axes.linewidth": 0.75,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#CCCCCC",
    "grid.alpha": 0.45,
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# High-contrast colorblind-safe & grayscale-separated palette
PAL = {
    "sppd":   "#005AB5",  # deep IEEE blue
    "static": "#008B6A",  # dark teal
    "pibo":   "#DC3220",  # vivid vermilion
    "mel":    "#984EA3",  # deep violet
    "lasso":  "#D97706",  # warm amber
    "lsp":    "#6B4C9A",  # muted indigo
    "raw":    "#555555",  # dark slate grey
    "bpso":   "#3898EC",  # sky blue
    "ga":     "#8C564B",  # dark brown
    "num":    "#222222",  # near black
    "wrong":  "#B2182B",  # crimson
}


def load_clin_per_source():
    p1 = pd.read_csv(EXP_ROOT / "results/clinical_non_llm_per_source_n3.tsv", sep="\t")
    p2 = pd.read_csv(EXP_ROOT / "results/clinical_llm_per_source_n3.tsv", sep="\t")
    tbl = {}
    for _, r in p1.iterrows():
        tbl.setdefault(r["method"], {})[r["task"]] = {k: float(r[k]) for k in ("b2_16", "b2", "b16", "b32", "b64", "b128")}
    for _, r in p2.iterrows():
        tbl.setdefault(r["method"], {})[r["task"]] = {k: float(r[k]) for k in ("b2_16", "b2", "b16", "b32", "b128")}
    return tbl


def fig1_clinical_budget_curves():
    c_df = pd.read_csv(EXP_ROOT / "results/clinical_non_llm_curves.tsv", sep="\t")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55))
    mark_idx = [0, 1, 3, 7, 15, 31, 63, 127]  # B = 1,2,4,8,16,32,64,128

    series = [
        ("gemini_static",     "LLM-Select (Rank, Static)",   PAL["static"], ":",  1.35, None, 0),
        ("gemini_paired_ext", "SP-PD (Proposed)",            PAL["sppd"],   "-",  2.0,  "o",  3.6),
        ("pibo_correct",      r"$\pi$BO (Same Prior)",       PAL["pibo"],   "-",  1.6,  "s",  3.4),
        ("mel",               "MEL",                         PAL["mel"],    "-.", 1.4,  "D",  3.1),
        ("gemini_raw_ext",    "SP-PD (argmax)",              PAL["raw"],    "--", 1.25, "v",  3.1),
        ("bpso",              "BPSO",                        PAL["bpso"],   "--", 1.15, "^",  3.0),
        ("ga",                "GA",                          PAL["ga"],     ":",  1.25, "x",  3.4),
    ]

    for ax, (panel, title, ylim) in zip(
        axes,
        [
            ("all_twelve", r"(a) All 12 Clinical Cohorts ($n{=}3, K{=}4$)", (52.5, 61.35)),
            ("new_six",    r"(b) Zero-History New-6 Cohorts ($n{=}3, K{=}4$)", (50.8, 56.55)),
        ],
    ):
        sub = c_df[(c_df["panel"] == panel) & (c_df["n"] == 3)]
        for mk, lbl, col, ls, lw, mkr, ms in series:
            d = sub[sub["method"] == mk].sort_values("budget")
            kw = dict(label=lbl, color=col, linestyle=ls, linewidth=lw)
            if mkr:
                kw.update(marker=mkr, markersize=ms, markevery=mark_idx, markeredgecolor="white", markeredgewidth=0.4)
            ax.plot(d["budget"], d["ba"] * 100.0, **kw)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
        ax.get_xaxis().set_major_formatter(FixedFormatter(["1", "2", "4", "8", "16", "32", "64", "128"]))
        ax.axvspan(2, 16, color=PAL["sppd"], alpha=0.055, zorder=0)
        ax.set_ylim(*ylim)
        ax.set_xlabel(r"Evaluation Budget $B$ ($\log_2$ scale)")
        ax.set_title(title, pad=5)

    # Direct scientific callouts
    axes[0].annotate(
        r"$\pi$BO erosion ($-2.72$ pp)",
        xy=(128, 57.40), xytext=(22, 54.0),
        arrowprops=dict(arrowstyle="->", color=PAL["pibo"], lw=0.8),
        fontsize=7.0, color=PAL["pibo"], fontweight="bold",
    )
    axes[0].annotate(
        r"$+1.80$ pp (12/12)",
        xy=(128, 59.20), xytext=(22, 59.68),
        arrowprops=dict(arrowstyle="->", color=PAL["sppd"], lw=0.8),
        fontsize=6.9, color=PAL["sppd"], fontweight="bold",
    )
    axes[0].set_ylabel("Outer Balanced Accuracy (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=4, frameon=False, columnspacing=1.2, handlelength=2.2)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(OUT_DIR / "fig1_clinical_budget_curves.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig1_clinical_budget_curves.png", bbox_inches="tight")
    plt.close(fig)


def fig2_pareto_and_mixture(tbl):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55))

    # Panel (a): Aligned vs Misaligned Prior Trade-off (All-12, B=2..16)
    ax = axes[0]
    pts = [
        ("gemini_static",            "wrong_static",            "LLM-Select (Rank)",   PAL["static"], "s", 36, (5, 2)),
        ("llm_select_score_correct", "llm_select_score_wrong",  "LLM-Select (Score)",  "#1B7837",     "D", 36, (5, -2)),
        ("gemini_paired_ext",        "wrong_paired_ext",        "SP-PD (Proposed)",    PAL["sppd"],   "*", 105, (-44, -13)),
        ("gemini_tied_ext",          "wrong_tied_ext",          "SP-PD (tie)",         "#3898EC",     "^", 36, (-52, -3)),
        ("gemini_raw_ext",           "wrong_raw_ext",           "SP-PD (argmax)",      PAL["raw"],    "v", 36, (-63, 3)),
        ("pibo_correct",             "pibo_wrong",              r"$\pi$BO ($B_{2\mathrm{-}16}$)", PAL["pibo"], "o", 36, (-14, 6)),
        ("llm_lasso_correct",        "llm_lasso_wrong",         "LLM-Lasso",           PAL["lasso"],  "P", 42, (-12, 6)),
        ("lsp_adapted_correct",      "lsp_adapted_wrong",       "LSP-Adapted",         PAL["lsp"],    "X", 42, (-8, -11)),
        ("numeric_ext",              "numeric_ext",             "Numeric Only",        PAL["num"],    "o", 32, (-58, -4)),
        ("mel",                      "mel",                     "MEL",                 PAL["mel"],    "d", 36, (5, 1)),
    ]
    for kc, kw, lbl, col, mk, sz, ofs in pts:
        x = np.mean([tbl[kw][t]["b2_16"] for t in ALL_12]) * 100.0
        y = np.mean([tbl[kc][t]["b2_16"] for t in ALL_12]) * 100.0
        ax.scatter(x, y, color=col, marker=mk, s=sz, zorder=5, edgecolors="white", linewidths=0.45)
        ax.annotate(lbl, (x, y), xytext=ofs, textcoords="offset points", fontsize=6.9, color=col,
                    fontweight="bold" if mk == "*" else "normal")

    seq_x = [np.mean([tbl[k][t]["b2_16"] for t in ALL_12]) * 100 for k in ("wrong_raw_ext", "wrong_tied_ext", "wrong_paired_ext", "wrong_static")]
    seq_y = [np.mean([tbl[k][t]["b2_16"] for t in ALL_12]) * 100 for k in ("gemini_raw_ext", "gemini_tied_ext", "gemini_paired_ext", "gemini_static")]
    ax.plot(seq_x, seq_y, color=PAL["sppd"], linestyle="--", linewidth=1.0, alpha=0.65, zorder=3)
    ax.set_xlim(54.5, 58.1)
    ax.set_ylim(57.05, 61.25)
    ax.set_xlabel("Misaligned Prior Outer BA (%)")
    ax.set_ylabel("Aligned Prior Outer BA (%)")
    ax.set_title(r"(a) Aligned vs. Misaligned Trade-off ($B \in [2,16]$)", pad=5)

    # Panel (b): Expected BA under Prior Alignment Probability q in [0, 1]
    ax = axes[1]
    qs = np.linspace(0.0, 1.0, 101)
    mix_specs = [
        ("gemini_paired_ext", "wrong_paired_ext", "b2_16", r"SP-PD ($B \in [2,16]$)",   PAL["sppd"],   "-",  2.0),
        ("gemini_static",     "wrong_static",     "b2_16", "LLM-Select (Rank, Static)", PAL["static"], ":",  1.45),
        ("llm_lasso_correct", "llm_lasso_wrong",  "b2_16", "LLM-Lasso",                 PAL["lasso"],  "--", 1.35),
        ("pibo_correct",      "pibo_wrong",       "b128",  r"$\pi$BO ($B{=}128$)",      PAL["pibo"],   "-.", 1.45),
        ("numeric_ext",       "numeric_ext",      "b2_16", "Numeric Search Only",       PAL["num"],    ":",  1.15),
        ("ga",                "ga",               "b128",  r"GA ($B{=}128$)",           PAL["ga"],     ":",  1.15),
    ]
    for kc, kw, band, lbl, col, ls, lw in mix_specs:
        yc = np.mean([tbl[kc][t][band] for t in ALL_12]) * 100.0
        yw = np.mean([tbl[kw][t][band] for t in ALL_12]) * 100.0
        ax.plot(qs, qs * yc + (1.0 - qs) * yw, label=lbl, color=col, linestyle=ls, linewidth=lw)
    ax.set_xlabel(r"Prior Alignment Probability $q$")
    ax.set_ylabel("Expected Outer BA (%)")
    ax.set_title(r"(b) Expected BA Under Alignment Mixture $q$", pad=5)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#CCCCCC", fontsize=6.8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_pareto_and_alignment_mixture.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig2_pareto_and_alignment_mixture.png", bbox_inches="tight")
    plt.close(fig)


def fig3_cost_and_stability():
    cost_df = pd.read_csv(EXP_ROOT / "results/llm_cost_ledger.tsv", sep="\t")
    cost_c = cost_df[cost_df["kind"] == "correct"].groupby("method", as_index=False).agg(
        batched_calls=("offline_calls", "sum"),
        native_calls=("native_calls", "sum"),
        total_tokens=("total_tokens", "sum"),
    )
    stab_df = pd.read_csv(EXP_ROOT / "results/llm_stability_new6.tsv", sep="\t")

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.45))

    # Panel (a): LLM API call & token efficiency
    ax = axes[0]
    labels = ["SP-PD\n(1/task)", "LLM-Select\n(Score, Batch)", "LLM-Lasso\n(Batch)", "LLM-Select\n(Score, Native)"]
    c_sp = int(cost_c[cost_c["method"] == "ours_top8_rank"]["batched_calls"].iloc[0])
    c_sc = int(cost_c[cost_c["method"] == "llm_select_score"]["batched_calls"].iloc[0])
    c_la = int(cost_c[cost_c["method"] == "llm_lasso"]["batched_calls"].iloc[0])
    c_na = int(cost_c[cost_c["method"] == "llm_select_score"]["native_calls"].iloc[0])
    t_sp = int(cost_c[cost_c["method"] == "ours_top8_rank"]["total_tokens"].iloc[0]) / 1000.0
    t_sc = int(cost_c[cost_c["method"] == "llm_select_score"]["total_tokens"].iloc[0]) / 1000.0
    t_la = int(cost_c[cost_c["method"] == "llm_lasso"]["total_tokens"].iloc[0]) / 1000.0

    vals = [c_sp, c_sc, c_la, c_na]
    toks = [f"{t_sp:.1f}k tok", f"{t_sc:.1f}k tok", f"{t_la:.1f}k tok", "per-feature"]
    cols = [PAL["sppd"], PAL["static"], PAL["lasso"], "#999999"]
    hatches = ["", "//", "\\\\", ".."]
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, width=0.52, color=cols, edgecolor="#222222", linewidth=0.6)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.3)
    ax.set_ylim(5, 3600)
    ax.set_ylabel("Total LLM Calls (12 Cohorts, log)")
    ax.set_title("(a) Offline LLM API Call & Token Footprint", pad=5)
    for b, v, tk in zip(bars, vals, toks):
        ax.text(b.get_x() + b.get_width() / 2.0, v * 1.18, f"{v}\n({tk})", ha="center", va="bottom", fontsize=6.8)

    # Panel (b): 5-Run LLM sampling stability on New-6
    ax = axes[1]
    sub = stab_df[(stab_df["kind"] == "correct") & (stab_df["task"] != "NEW6_MEAN")]
    mean_row = stab_df[(stab_df["kind"] == "correct") & (stab_df["task"] == "NEW6_MEAN")].iloc[0]
    x_s = np.arange(len(sub) + 1)
    lbls = list(sub["task"]) + ["Mean"]
    mp = list(sub["paired_b2_16_mean"] * 100) + [float(mean_row["paired_b2_16_mean"]) * 100]
    sp = list(sub["paired_b2_16_sd"] * 100) + [float(mean_row["paired_b2_16_sd"]) * 100]
    ms = list(sub["static_ba_mean"] * 100) + [float(mean_row["static_ba_mean"]) * 100]
    ss = list(sub["static_ba_sd"] * 100) + [float(mean_row["static_ba_sd"]) * 100]
    w = 0.36
    ax.bar(x_s - w / 2, mp, w, yerr=sp, capsize=2.2, color=PAL["sppd"], edgecolor="#222222", linewidth=0.5,
           error_kw=dict(lw=0.8), label=r"SP-PD (Mean SD$=0.18\%$)")
    ax.bar(x_s + w / 2, ms, w, yerr=ss, capsize=2.2, color=PAL["static"], alpha=0.85, hatch="//", edgecolor="#222222", linewidth=0.5,
           error_kw=dict(lw=0.8), label=r"Static Top-4 (Mean SD$=0.35\%$)")
    ax.set_xticks(x_s)
    ax.set_xticklabels(lbls, rotation=20, ha="right", fontsize=7.4)
    ax.set_ylim(47, 68)
    ax.set_ylabel("Outer BA (%) Across 5 LLM Runs")
    ax.set_title("(b) Multi-Run LLM Repeatability (New-6)", pad=5)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="#CCCCCC", fontsize=6.8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_llm_cost_and_stability.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig3_llm_cost_and_stability.png", bbox_inches="tight")
    plt.close(fig)


def fig4_cohort_and_scaling(tbl):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7))

    # Panel (a): Per-cohort gains
    ax = axes[0]
    items = [
        (t, (tbl["gemini_paired_ext"][t]["b2_16"] - tbl["numeric_ext"][t]["b2_16"]) * 100.0,
            (tbl["gemini_paired_ext"][t]["b2_16"] - tbl["ga"][t]["b128"]) * 100.0)
        for t in ALL_12
    ]
    items.sort(key=lambda z: z[1])
    yp = np.arange(len(items))
    ax.axvline(0.0, color="#222222", linewidth=0.8)
    ax.barh(yp + 0.19, [z[1] for z in items], height=0.35, color=PAL["sppd"], edgecolor="#222222", linewidth=0.4,
            label=r"SP-PD ($B_{2\mathrm{-}16}$) $-$ Numeric ($B_{2\mathrm{-}16}$)")
    ax.barh(yp - 0.19, [z[2] for z in items], height=0.35, color=PAL["lasso"], alpha=0.85, hatch="//", edgecolor="#222222", linewidth=0.4,
            label=r"SP-PD ($B_{2\mathrm{-}16}$) $-$ GA ($B_{128}$)")
    ax.set_yticks(yp)
    ax.set_yticklabels([f"{z[0]}*" if z[0] in NEW_SIX else z[0] for z in items], fontsize=7.4)
    ax.set_xlabel("Outer BA Gain (percentage points)")
    ax.set_title("(a) Per-Cohort Gains (* = Zero-History New-6)", pad=5)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#CCCCCC", fontsize=6.6)

    # Panel (b): Sample-size scaling n in {2,3,5,10,20}
    ax = axes[1]
    m_both = pd.concat([
        pd.read_csv(OLD_CLIN / "results/means.tsv", sep="\t"),
        pd.read_csv(NEW_CLIN / "results/means.tsv", sep="\t"),
    ], ignore_index=True)
    m_sub = m_both[(m_both["k"] == 4) & (m_both["band"] == "main")].groupby(["n", "arm"], as_index=False)["ba"].mean()
    specs = [
        ("gemini_static", "LLM-Select (Rank, Static)", PAL["static"], ":",  "s", 1.4),
        ("gemini",        "SP-PD (Proposed)",          PAL["sppd"],   "-",  "o", 1.9),
        ("gemini_raw",    "SP-PD (argmax)",            PAL["raw"],    "--", "v", 1.3),
        ("numeric",       "Numeric Search Only",       PAL["num"],    "-.", "^", 1.3),
        ("wrong",         "SP-PD (Misaligned Prior)",  PAL["wrong"],  "-",  "d", 1.5),
        ("wrong_static",  "Static (Misaligned Prior)", PAL["pibo"],   ":",  "x", 1.3),
    ]
    for arm, lbl, col, ls, mkr, lw in specs:
        d = m_sub[m_sub["arm"] == arm].sort_values("n")
        ax.plot(d["n"], d["ba"] * 100.0, label=lbl, color=col, linestyle=ls, marker=mkr,
                markersize=3.8, markeredgecolor="white", markeredgewidth=0.35, linewidth=lw)
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 3, 5, 10, 20])
    ax.get_xaxis().set_major_formatter(FixedFormatter(["2", "3", "5", "10", "20"]))
    ax.set_xlabel(r"Labeled Training Patients per Class ($n$)")
    ax.set_ylabel("Outer BA (%) Across All 12 Cohorts")
    ax.set_title(r"(b) Sample-Size Scaling ($K{=}4, B \in [2,16]$)", pad=5)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#CCCCCC", fontsize=6.5)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_cohort_effects_and_scaling.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig4_cohort_effects_and_scaling.png", bbox_inches="tight")
    plt.close(fig)


def fig5_numerical_b1000():
    nc = pd.read_csv(EXP_ROOT / "results/numerical_b1000_curves.tsv", sep="\t")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55))
    mark_b = [8, 16, 32, 64, 128, 256, 500, 1000]

    for ax, (suite, title) in zip(axes, [
        ("prospective_10", r"(a) Zero-History 10 Sources ($K \leq 8$)"),
        ("author_13",      r"(b) Author 13 High-Dim Sources ($K \leq 8$)"),
    ]):
        sub = nc[(nc["suite"] == suite) & (nc["budget"] >= 8)]
        dtu = sub[sub["method"] == "tree_union"].sort_values("budget")
        ax.plot(dtu["budget"], dtu["ceil_precision"] * 100.0, color=PAL["sppd"], linestyle=":",
                linewidth=1.25, alpha=0.85, label="SP-PD-Num Pool Ceiling")

        specs = [
            ("tree_union", "SP-PD-Num (Proposed)", PAL["sppd"],   "-",  2.0, "o"),
            ("mel",        "MEL (IEEE TKDE)",      PAL["mel"],    "-",  1.5, "s"),
            ("iid",        "Strong-Init IID",      PAL["static"], "--", 1.25, "D"),
            ("bpso",       "BPSO",                 PAL["bpso"],   "-.", 1.25, "^"),
            ("ga",         "GA",                   PAL["ga"],     ":",  1.3, "x"),
        ]
        for mk, lbl, col, ls, lw, mkr in specs:
            d = sub[sub["method"] == mk].sort_values("budget")
            idx = [int(np.argmin(np.abs(d["budget"].values - mb))) for mb in mark_b]
            ax.plot(d["budget"], d["precision"] * 100.0, label=lbl, color=col, linestyle=ls, linewidth=lw,
                    marker=mkr, markersize=3.3, markevery=idx, markeredgecolor="white", markeredgewidth=0.35)
        ax.set_xscale("log")
        ax.set_xticks([8, 16, 32, 64, 128, 256, 500, 1000])
        ax.get_xaxis().set_major_formatter(FixedFormatter(["8", "16", "32", "64", "128", "256", "500", "1k"]))
        ax.set_xlabel("Evaluation Budget $B$ (log scale)")
        ax.set_title(title, pad=5)

    # Annotate pool-vs-delivery scissors gap on panel (a)
    axes[0].annotate(
        "", xy=(1000, 72.36), xytext=(1000, 81.73),
        arrowprops=dict(arrowstyle="<->", color="#333333", lw=0.8),
    )
    axes[0].text(740, 77.0, "9.4 pp\ngap", fontsize=6.8, ha="right", va="center", color="#333333")
    axes[0].set_ylabel("Outer Macro-Precision (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=False, columnspacing=1.4, handlelength=2.2)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(OUT_DIR / "fig5_numerical_b1000_scissors.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig5_numerical_b1000_scissors.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    tbl = load_clin_per_source()
    fig1_clinical_budget_curves()
    fig2_pareto_and_mixture(tbl)
    fig3_cost_and_stability()
    fig4_cohort_and_scaling(tbl)
    fig5_numerical_b1000()
    print("ALL 5 STRICT IEEE FIGURES RENDERED IN", OUT_DIR)
