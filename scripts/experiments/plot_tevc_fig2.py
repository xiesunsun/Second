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
    "legend.fontsize": 7.2,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

df_nc = pd.read_csv(base / "clinical_12_per_dataset_nc.tsv", sep="\t")
means = df_nc.groupby(["n", "method"])["b128"].mean().unstack() * 100.0
ns = [3, 10, 20, 30]

styles = {
    "SP-PD":              dict(color="#B2182B", ls="-",  lw=1.7,  marker="o", ms=4.2, zorder=10),
    "LLM-Select (Rank)":  dict(color="#2166AC", ls="--", lw=1.2,  marker="s", ms=3.5, zorder=8),
    "LLM-Select (Score)": dict(color="#4393C3", ls=":",  lw=1.25, marker="D", ms=3.3, zorder=7),
    "LLM-Lasso":          dict(color="#1B7837", ls="-.", lw=1.15, marker="^", ms=3.6, zorder=6),
    "LSP":                dict(color="#762A83", ls=(0, (3, 1, 1, 1)), lw=1.1, marker="v", ms=3.5, zorder=5),
    r"$\pi$BO":           dict(color="#E66101", ls=(0, (5, 2)), lw=1.1, marker="P", ms=3.6, zorder=5),
    "MEL":                dict(color="#4D4D4D", ls="-",  lw=1.0,  marker="X", ms=3.3, zorder=4),
    "L1-LR":              dict(color="#878787", ls="--", lw=0.95, marker="h", ms=3.2, zorder=3),
}

aln_keys = {
    "SP-PD": "gemini_paired_tau15",
    "LLM-Select (Rank)": "gemini_static",
    "LLM-Select (Score)": "llm_select_score_correct",
    "LLM-Lasso": "llm_lasso_correct",
    "LSP": "lsp_adapted_correct",
    r"$\pi$BO": "pibo_correct",
    "MEL": "mel",
    "L1-LR": "l1_lr_static",
}
mis_keys = {
    "SP-PD": "wrong_paired_tau15",
    "LLM-Select (Rank)": "wrong_static",
    "LLM-Select (Score)": "llm_select_score_wrong",
    "LLM-Lasso": "llm_lasso_wrong",
    "LSP": "lsp_adapted_wrong",
    r"$\pi$BO": "pibo_wrong",
    "MEL": "mel",
}

fig, axes = plt.subplots(1, 3, figsize=(7.10, 2.48), dpi=300)
plt.subplots_adjust(left=0.064, right=0.990, bottom=0.17, top=0.73, wspace=0.29)

# Panel (a): Aligned scaling across n
ax = axes[0]
for label, col in aln_keys.items():
    st = styles[label]
    y = [means.loc[n, col] for n in ns]
    ax.plot(ns, y, color=st["color"], ls=st["ls"], lw=st["lw"], marker=st["marker"], ms=st["ms"],
            markerfacecolor=st["color"] if label == "SP-PD" else "white",
            markeredgewidth=0.9, zorder=st["zorder"])
ax.set_xticks(ns)
ax.set_xlim(1.5, 31.5)
ax.set_ylim(56.2, 68.5)
ax.set_xlabel(r"Training Instances per Class $n$")
ax.set_ylabel("Balanced Accuracy (%)")
ax.set_title("(a) Aligned Prior", pad=5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, linestyle="--", linewidth=0.45, color="#CCCCCC", alpha=0.75)

# Panel (b): Misaligned recovery across n
ax = axes[1]
for label, col in mis_keys.items():
    st = styles[label]
    y = [means.loc[n, col] for n in ns]
    ax.plot(ns, y, color=st["color"], ls=st["ls"], lw=st["lw"], marker=st["marker"], ms=st["ms"],
            markerfacecolor=st["color"] if label == "SP-PD" else "white",
            markeredgewidth=0.9, zorder=st["zorder"])
ax.set_xticks(ns)
ax.set_xlim(1.5, 31.5)
ax.set_ylim(53.8, 66.8)
ax.set_xlabel(r"Training Instances per Class $n$")
ax.set_ylabel("Balanced Accuracy (%)")
ax.set_title("(b) Misaligned Prior", pad=5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, linestyle="--", linewidth=0.45, color="#CCCCCC", alpha=0.75)

# Panel (c): Aligned vs Misaligned scatter at n=30 (all 14 methods!)
ax = axes[2]
# Diagonal y = x reference line (prior-independent baselines lie on this line)
diag = np.linspace(58.5, 67.8, 50)
ax.plot(diag, diag, color="#BDBDBD", ls=":", lw=0.8, zorder=1)

# Plot all 8 non-LLM baselines in clean muted gray dots (with MEL highlighted)
non_llm = [
    ("GA", "ga"), ("BPSO", "bpso"), ("EDA", "eda"), ("MEL", "mel"),
    ("mRMR", "mrmr_static"), ("ReliefF", "relieff_static"),
    ("L1-LR", "l1_lr_static"), ("RF-Imp", "rf_imp_static"),
]
for nm, col in non_llm:
    v = means.loc[30, col]
    if nm == "MEL":
        st = styles["MEL"]
        ax.scatter([v], [v], color=st["color"], marker=st["marker"], s=24, facecolors="white",
                   edgecolors=st["color"], linewidths=0.9, zorder=5)
    else:
        ax.scatter([v], [v], color="#969696", marker="o", s=14, alpha=0.8, zorder=3)

# Plot 6 prior/LLM methods with clean, non-overlapping direct annotations
prior_pts = [
    ("SP-PD", "gemini_paired_tau15", "wrong_paired_tau15", (-18, 7)),
    ("LLM-Select (Rank)", "gemini_static", "wrong_static", (5, -3)),
    ("LLM-Select (Score)", "llm_select_score_correct", "llm_select_score_wrong", (5, -2)),
    ("LLM-Lasso", "llm_lasso_correct", "llm_lasso_wrong", (-46, 5)),
    ("LSP", "lsp_adapted_correct", "lsp_adapted_wrong", (-23, -10)),
    (r"$\pi$BO", "pibo_correct", "pibo_wrong", (-24, -4)),
]
short_lbl = {
    "SP-PD": "SP-PD",
    "LLM-Select (Rank)": "Select (Rank)",
    "LLM-Select (Score)": "Select (Score)",
    "LLM-Lasso": "LLM-Lasso",
    "LSP": "LSP",
    r"$\pi$BO": r"$\pi$BO",
}
for lbl, a_col, m_col, (dx, dy) in prior_pts:
    st = styles[lbl]
    x_v, y_v = means.loc[30, m_col], means.loc[30, a_col]
    ax.scatter([x_v], [y_v], color=st["color"], marker=st["marker"],
               s=36 if lbl == "SP-PD" else 24,
               facecolors=st["color"] if lbl == "SP-PD" else "white",
               edgecolors=st["color"], linewidths=1.0, zorder=st["zorder"])
    ax.annotate(short_lbl[lbl], (x_v, y_v), xytext=(dx, dy), textcoords="offset points",
                fontsize=6.8, fontweight="bold" if lbl == "SP-PD" else "normal",
                color=st["color"])

ax.annotate("Non-LLM baselines", (65.05, 65.12), xytext=(-68, -2), textcoords="offset points",
            fontsize=6.5, color="#636363",
            arrowprops=dict(arrowstyle="->", color="#969696", lw=0.6))

ax.set_xlim(58.5, 66.4)
ax.set_ylim(64.5, 68.3)
ax.set_xlabel("Misaligned BA (%)")
ax.set_ylabel("Aligned BA (%)")
ax.set_title(r"(c) Trade-off at $n = 30$", pad=5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, linestyle="--", linewidth=0.45, color="#CCCCCC", alpha=0.75)

# Clean 2-row shared legend above panels
handles, labels = [], []
for label in ["SP-PD", "LLM-Select (Rank)", "LLM-Select (Score)", "LLM-Lasso", "LSP", r"$\pi$BO", "MEL", "L1-LR"]:
    st = styles[label]
    h = plt.Line2D([0], [0], color=st["color"], ls=st["ls"], lw=st["lw"],
                   marker=st["marker"], ms=st["ms"],
                   markerfacecolor=st["color"] if label == "SP-PD" else "white",
                   markeredgewidth=0.9)
    handles.append(h)
    labels.append(label)

fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.525, 0.995),
           ncol=4, frameon=False, handlelength=2.0, columnspacing=1.25, handletextpad=0.45)

pdf_path = out_dir / "fig2_clinical_scaling.pdf"
png_path = out_dir / "fig2_clinical_scaling.png"
fig.savefig(pdf_path, format="pdf")
fig.savefig(png_path, format="png", dpi=220)
print("SAVED:", pdf_path, png_path)
