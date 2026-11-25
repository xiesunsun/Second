import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

base = Path("/home/sunxie/paper-experiments-20261124/results")
out_dir = Path("/home/sunxie/paper-experiments-20261124/tevc_figures")
out_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.0,
    "axes.titlesize": 8.2,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

wc = pd.read_csv(base / "numerical_wide_curves.tsv", sep="\t")
tc = pd.read_csv(base / "numerical_b1000_curves.tsv", sep="\t")
eda_tc = pd.read_csv(base / "numerical_b1000_eda_curves.tsv", sep="\t")
tc_comb = pd.concat([
    tc[tc["suite"].isin(["author_13", "prospective_10"])][["suite", "method", "budget", "precision", "ba"]],
    eda_tc[["suite", "method", "budget", "precision", "ba"]],
], ignore_index=True)

t_a13 = tc_comb[tc_comb["suite"] == "author_13"].set_index(["method", "budget"])[["precision", "ba"]]
t_p10 = tc_comb[tc_comb["suite"] == "prospective_10"].set_index(["method", "budget"])[["precision", "ba"]]
t_23 = ((13.0 * t_a13 + 10.0 * t_p10) / 23.0).reset_index()

styles = {
    "SP-PD":  dict(color="#B2182B", ls="-",  lw=1.7, marker="o", ms=3.8, zorder=10),
    "MEL":    dict(color="#2166AC", ls="--", lw=1.25, marker="s", ms=3.2, zorder=8),
    "EDA":    dict(color="#1B7837", ls="-.", lw=1.15, marker="^", ms=3.2, zorder=6),
    "BPSO":   dict(color="#E66101", ls=":",  lw=1.25, marker="D", ms=3.0, zorder=5),
    "GA":     dict(color="#762A83", ls=(0, (3, 1, 1, 1)), lw=1.1, marker="v", ms=3.0, zorder=4),
    "mRMR":   dict(color="#4D4D4D", ls=(0, (5, 2)), lw=1.0, marker=None, ms=0, zorder=3),
    "RF-Imp": dict(color="#878787", ls=(0, (1, 1.5)), lw=1.0, marker=None, ms=0, zorder=2),
}

mark_budgets = [8, 16, 32, 64, 128, 256, 500, 1000]
b_vals = np.arange(8, 1001)

fig, axes = plt.subplots(1, 3, figsize=(7.10, 2.38), dpi=300)
plt.subplots_adjust(left=0.064, right=0.992, bottom=0.18, top=0.77, wspace=0.29)

def plot_panel(ax, df_map, y_scale=100.0, y_label="", title="", ylim=None):
    for label, series in df_map.items():
        st = styles[label]
        y = series.loc[b_vals].to_numpy() * y_scale
        ax.plot(b_vals, y, color=st["color"], ls=st["ls"], lw=st["lw"], label=label, zorder=st["zorder"])
        if st["marker"] is not None:
            ym = series.loc[mark_budgets].to_numpy() * y_scale
            ax.plot(mark_budgets, ym, color=st["color"], ls="none", marker=st["marker"],
                    ms=st["ms"], markerfacecolor="white" if label != "SP-PD" else st["color"],
                    markeredgewidth=0.9, zorder=st["zorder"] + 1)
    ax.set_xscale("log")
    ax.set_xlim(7.2, 1120)
    ax.set_xticks([8, 32, 128, 500, 1000])
    ax.set_xticklabels(["8", "32", "128", "500", "1000"])
    ax.minorticks_off()
    ax.set_xlabel(r"Evaluation Budget $B$")
    ax.set_ylabel(y_label)
    ax.set_title(title, pad=5)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.45, color="#CCCCCC", alpha=0.75)

# Panel (a): Relaxed Limit Macro-Precision
w_prec = {
    "SP-PD":  wc[wc["method"] == "geometry"].set_index("budget")["precision"],
    "MEL":    wc[wc["method"] == "mel"].set_index("budget")["precision"],
    "EDA":    wc[wc["method"] == "eda"].set_index("budget")["precision"],
    "BPSO":   wc[wc["method"] == "bpso"].set_index("budget")["precision"],
    "GA":     wc[wc["method"] == "ga"].set_index("budget")["precision"],
    "mRMR":   wc[wc["method"] == "mrmr_static"].set_index("budget")["precision"],
    "RF-Imp": wc[wc["method"] == "rf_imp_static"].set_index("budget")["precision"],
}
plot_panel(axes[0], w_prec, y_scale=100.0, y_label="Macro-Precision (%)",
           title=r"(a) $K = \lfloor 0.15p \rfloor$: Precision", ylim=(75.5, 89.2))

# Panel (b): Relaxed Limit Selected Features
w_feat = {
    "SP-PD":  wc[wc["method"] == "geometry"].set_index("budget")["width"],
    "MEL":    wc[wc["method"] == "mel"].set_index("budget")["width"],
    "EDA":    wc[wc["method"] == "eda"].set_index("budget")["width"],
    "BPSO":   wc[wc["method"] == "bpso"].set_index("budget")["width"],
    "GA":     wc[wc["method"] == "ga"].set_index("budget")["width"],
    "mRMR":   wc[wc["method"] == "mrmr_static"].set_index("budget")["width"],
}
plot_panel(axes[1], w_feat, y_scale=1.0, y_label="Selected Features",
           title=r"(b) $K = \lfloor 0.15p \rfloor$: Subset Size", ylim=(0, 560))

# Panel (c): Compact Limit (K=8, 23 sources) Balanced Accuracy (BA)
ex_t = pd.read_csv(base / "numerical_b1000_extra_per_source.tsv", sep="\t")
rf_ba_val = ex_t[ex_t["method"] == "rf_imp_static"]["b1000_ba"].mean()
t_ba = {
    "SP-PD":  t_23[t_23["method"] == "tree_union"].set_index("budget")["ba"],
    "MEL":    t_23[t_23["method"] == "mel"].set_index("budget")["ba"],
    "EDA":    t_23[t_23["method"] == "eda"].set_index("budget")["ba"],
    "BPSO":   t_23[t_23["method"] == "bpso"].set_index("budget")["ba"],
    "GA":     t_23[t_23["method"] == "ga"].set_index("budget")["ba"],
    "mRMR":   t_23[t_23["method"] == "mrmr_static"].set_index("budget")["ba"],
    "RF-Imp": pd.Series(rf_ba_val, index=np.arange(1, 1001)),
}
plot_panel(axes[2], t_ba, y_scale=100.0, y_label="Balanced Accuracy (%)",
           title=r"(c) $K = 8$: Balanced Accuracy", ylim=(50.0, 74.0))

handles, labels = [], []
for label in ["SP-PD", "MEL", "EDA", "BPSO", "GA", "mRMR", "RF-Imp"]:
    st = styles[label]
    h = plt.Line2D([0], [0], color=st["color"], ls=st["ls"], lw=st["lw"],
                   marker=st["marker"], ms=st["ms"] if st["marker"] else 0,
                   markerfacecolor="white" if (st["marker"] and label != "SP-PD") else st["color"],
                   markeredgewidth=0.9)
    handles.append(h)
    labels.append(label)

fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.525, 0.99),
           ncol=7, frameon=False, handlelength=2.1, columnspacing=1.15, handletextpad=0.45)

pdf_path = out_dir / "fig1_highdim_convergence.pdf"
png_path = out_dir / "fig1_highdim_convergence.png"
fig.savefig(pdf_path, format="pdf")
fig.savefig(png_path, format="png", dpi=220)
print("SAVED:", pdf_path, png_path)
