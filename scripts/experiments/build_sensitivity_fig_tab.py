"""Generate IEEE Fig. 6 (Gate threshold tau sweep + Cardinality cap K=4 vs K=8) and Table VII."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OLD_CLIN = Path("/home/sunxie/clinical-prospective-20261123.cjUGun3X")
NEW_CLIN = Path("/home/sunxie/clinical-extension-20261124")
FIG_DIR = Path("/home/sunxie/paper-archive-20261124/ieee_draft/figures")
TAB_DIR = Path("/home/sunxie/paper-archive-20261124/ieee_draft/tables")

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.1,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
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

PAL = {
    "sppd": "#005AB5",
    "static": "#008B6A",
    "wrong": "#DC3220",
    "raw": "#666666",
    "num": "#222222",
}


def main():
    tp = pd.read_csv(NEW_CLIN / "results/threshold-path.tsv", sep="\t")
    # Average Earlier-6 and New-6 for All-12, plus show New-6 separately
    e6 = tp[tp["panel"] == "earlier_six"].reset_index(drop=True)
    n6 = tp[tp["panel"] == "new_six"].reset_index(drop=True)
    all_g = (e6["gemini"].values + n6["gemini"].values) / 2.0
    all_w = (e6["wrong"].values + n6["wrong"].values) / 2.0
    # Monotonic x-axis positions for discrete threshold steps (with inf as last tick)
    tau_lbls = ["0", "0.42", "0.54", "0.79", "1.0*", "1.46", "1.58", "2.0", "2.24", "3.16", "5.0", r"$\infty$"]
    x_idx = np.arange(len(tau_lbls))

    # K=4 vs K=8 data
    m_old = pd.read_csv(OLD_CLIN / "results/means.tsv", sep="\t")
    m_new = pd.read_csv(NEW_CLIN / "results/means.tsv", sep="\t")

    def get_k(df, k, arm):
        return float(df[(df["n"] == 3) & (df["k"] == k) & (df["band"] == "main") & (df["arm"] == arm)]["ba"].iloc[0]) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))

    # Panel (a): Threshold tau sensitivity on All-12 & New-6
    ax = axes[0]
    ax.axvline(4, color="#005AB5", linestyle="--", linewidth=0.9, alpha=0.75, zorder=1)
    ax.plot(x_idx, all_g, color=PAL["sppd"], marker="o", markersize=3.4, linewidth=1.8, label="All-12 Aligned")
    ax.plot(x_idx, all_w, color=PAL["wrong"], marker="s", markersize=3.2, linewidth=1.6, linestyle="-.", label="All-12 Misaligned")
    ax.plot(x_idx, n6["gemini"].values, color=PAL["static"], marker="^", markersize=3.2, linewidth=1.4, linestyle="--", label="New-6 Aligned")
    ax.plot(x_idx, n6["wrong"].values, color="#D97706", marker="d", markersize=3.0, linewidth=1.3, linestyle=":", label="New-6 Misaligned")
    ax.annotate(r"Default $\tau{=}1.0$" + "\n(1-SE elbow)", xy=(4, all_g[4]), xytext=(4.6, 60.45),
                arrowprops=dict(arrowstyle="->", color=PAL["sppd"], lw=0.75), fontsize=6.7, color=PAL["sppd"], fontweight="bold")
    ax.set_xticks(x_idx[::2].tolist() + [4, 11])
    tick_pos = [0, 2, 4, 6, 8, 10, 11]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([tau_lbls[i] for i in tick_pos], fontsize=7.2)
    ax.set_ylim(48.8, 62.2)
    ax.set_xlabel(r"Paired Gate Threshold $\tau$ ($D^2 > \tau^2 m$)")
    ax.set_ylabel("Outer Balanced Accuracy (%)")
    ax.set_title(r"(a) Sensitivity to Delivery Gate Threshold $\tau$", pad=5)
    ax.legend(loc="lower left", ncol=2, frameon=True, framealpha=0.95, edgecolor="#CCCCCC", fontsize=6.4, columnspacing=0.8)

    # Panel (b): Cardinality cap K=4 vs K=8 (All-12 & New-6)
    ax = axes[1]
    groups = [
        ("All-12\n($K{=}4$)", [(get_k(m_old, 4, a) + get_k(m_new, 4, a)) / 2 for a in ("gemini", "gemini_static", "gemini_raw", "numeric")]),
        ("All-12\n($K{=}8$)", [(get_k(m_old, 8, a) + get_k(m_new, 8, a)) / 2 for a in ("gemini", "gemini_static", "gemini_raw", "numeric")]),
        ("New-6\n($K{=}4$)",  [get_k(m_new, 4, a) for a in ("gemini", "gemini_static", "gemini_raw", "numeric")]),
        ("New-6\n($K{=}8$)",  [get_k(m_new, 8, a) for a in ("gemini", "gemini_static", "gemini_raw", "numeric")]),
    ]
    gx = np.arange(len(groups))
    bw = 0.19
    bar_specs = [
        ("SP-PD (Proposed)", PAL["sppd"], ""),
        ("LLM-Select (Static)", PAL["static"], "//"),
        ("SP-PD (argmax)", PAL["raw"], "\\\\"),
        ("SP-PD-Num", "#BBBBBB", ".."),
    ]
    for bi, (lbl, col, ht) in enumerate(bar_specs):
        vals = [g[1][bi] for g in groups]
        bars = ax.bar(gx + (bi - 1.5) * bw, vals, width=bw, color=col, hatch=ht,
                      edgecolor="#222222", linewidth=0.5, label=lbl)
        if bi == 0:
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2 - 0.03, v + 0.25, f"{v:.1f}", ha="center", va="bottom",
                        fontsize=5.9, color=PAL["sppd"], fontweight="bold")
    ax.set_xticks(gx)
    ax.set_xticklabels([g[0] for g in groups], fontsize=7.3)
    ax.set_ylim(52.0, 62.5)
    ax.set_ylabel("Outer Balanced Accuracy (%)")
    ax.set_title(r"(b) Sensitivity to Hard Feature Cap $K \in \{4,8\}$", pad=5)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="#CCCCCC", fontsize=6.4)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_parameter_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig6_parameter_sensitivity.png", bbox_inches="tight")
    plt.close(fig)

    # Generate compact single-column Table VII
    key_idx = [0, 2, 4, 6, 8, 11]  # tau = 0, 0.54, 1.0, 1.58, 2.24, inf
    t7 = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\begin{threeparttable}",
        r"\caption{Sensitivity of \textbf{SP-PD} to Gate Threshold $\tau$ and Hard Feature Cap $K$ ($n{=}3, B \in [2,16]$, Outer BA \%).}",
        r"\label{tab:sensitivity}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.6pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Setting & All-12 Alig. & All-12 Misal. & New-6 Alig. & New-6 Misal. \\",
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\textit{Gate Threshold $\tau$ ($D^2 > \tau^2 m$, $K{=}4$)}} \\",
    ]
    for i in key_idx:
        tstr = r"\textbf{$\tau{=}1.0$ (Default)}" if i == 4 else (r"$\tau{=}\infty$ (Static)" if i == 11 else f"$\\tau{{=}}{tau_lbls[i]}$")
        if i == 4:
            t7.append(f"{tstr} & \\textbf{{{all_g[i]:.2f}}} & \\textbf{{{all_w[i]:.2f}}} & \\textbf{{{n6['gemini'].iloc[i]:.2f}}} & \\textbf{{{n6['wrong'].iloc[i]:.2f}}} \\\\")
        else:
            t7.append(f"{tstr} & {all_g[i]:.2f} & {all_w[i]:.2f} & {n6['gemini'].iloc[i]:.2f} & {n6['wrong'].iloc[i]:.2f} \\\\")
    t7 += [
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\textit{Hard Feature Cardinality Cap $K$ ($\tau{=}1.0$)}} \\",
    ]
    for k in (4, 8):
        a12_c = (get_k(m_old, k, "gemini") + get_k(m_new, k, "gemini")) / 2
        a12_w = (get_k(m_old, k, "wrong") + get_k(m_new, k, "wrong")) / 2
        n6_c = get_k(m_new, k, "gemini")
        n6_w = get_k(m_new, k, "wrong")
        n6_st = get_k(m_new, k, "gemini_static")
        tag = f"$K{{=}}{k}$ (Primary)" if k == 4 else f"$K{{=}}{k}$ (Static: {n6_st:.2f}\\tnote{{a}})"
        t7.append(f"{tag} & {a12_c:.2f} & {a12_w:.2f} & {n6_c:.2f} & {n6_w:.2f} \\\\")
    t7 += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\scriptsize",
        r"\item[a] At $K{=}8$ on Zero-History New-6, \textbf{SP-PD} (55.34\%) directly exceeds static LLM-Select (55.23\%) while also beating SP-PD-Num (54.00\%, +1.34 pp).",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    (TAB_DIR / "tab7_sensitivity.tex").write_text("\n".join(t7) + "\n")
    print("FIG 6 AND TABLE VII GENERATED.")


if __name__ == "__main__":
    main()
