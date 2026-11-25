"""Publication-grade IEEE TEVC Fig. 2 (strict Rule 8 compliance: STIX serif, hollow baseline markers,
solid red SP-PD, B&W distinguishable, crisp 2-contour Pareto plot with zero label/line overlap)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/sunxie/paper-experiments-20261124")
base = pd.read_csv(ROOT / "results/clinical_12_per_dataset_nc.tsv", sep="\t")
eb = pd.read_csv(ROOT / "results/clinical_schemeA_candidates_nc.tsv", sep="\t")

RULE = "eb_k4_0.50"
NS = [3, 10, 20, 30]

def get_curve(method, prior=None):
    if method == "SP-PD":
        sub = eb[(eb["rule"] == RULE) & (eb["prior"] == prior)]
        return [float(sub[sub["n"] == n]["ba"].mean()) * 100.0 for n in NS]
    sub = base[base["method"] == method]
    return [float(sub[sub["n"] == n]["b128"].mean()) * 100.0 for n in NS]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 8.0,
    "axes.labelsize": 8.2,
    "axes.titlesize": 8.6,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.40), gridspec_kw={"width_ratios": [1.0, 1.0, 1.18]})
plt.subplots_adjust(left=0.068, right=0.985, top=0.80, bottom=0.19, wspace=0.28)

# Rule 8 #3: SP-PD solid crimson circle; all comparison baselines hollow (white facecolor) with distinct linestyles
styles = {
    "SP-PD":              dict(color="#C62828", marker="o", mfc="#C62828", mec="#C62828", mew=0.9, lw=1.9, ms=4.8, ls="-",  zorder=8),
    "LLM-Select (Score)": dict(color="#7B1FA2", marker="s", mfc="white",   mec="#7B1FA2", mew=1.0, lw=1.3, ms=4.3, ls="--", zorder=5),
    "LLM-Lasso":          dict(color="#1565C0", marker="^", mfc="white",   mec="#1565C0", mew=1.0, lw=1.3, ms=4.5, ls="-.", zorder=6),
    "LSP":                dict(color="#00838F", marker="D", mfc="white",   mec="#00838F", mew=1.0, lw=1.2, ms=4.0, ls=":",  zorder=5),
    "MEL":                dict(color="#2E7D32", marker="v", mfc="white",   mec="#2E7D32", mew=1.0, lw=1.25, ms=4.4, ls="-", zorder=4),
    "mRMR":               dict(color="#E65100", marker="P", mfc="white",   mec="#E65100", mew=0.9, lw=1.15, ms=4.4, ls="--", zorder=3),
}

# Panel (a): Aligned Prior
ax = axes[0]
aln_map = [
    ("SP-PD",              get_curve("SP-PD", "correct")),
    ("LLM-Select (Score)", get_curve("llm_select_score_correct")),
    ("LLM-Lasso",          get_curve("llm_lasso_correct")),
    ("LSP",                get_curve("lsp_adapted_correct")),
    ("MEL",                get_curve("mel")),
    ("mRMR",               get_curve("mrmr_static")),
]
handles, labels = [], []
for name, vals in aln_map:
    ln, = ax.plot(NS, vals, label=name, **styles[name])
    handles.append(ln); labels.append(name)
ax.set_title("(a) Aligned Prior", pad=5)
ax.set_xlabel("Instances per Class ($n$)")
ax.set_ylabel("Balanced Accuracy (%)")
ax.set_xticks(NS)
ax.set_xlim(1.5, 31.8)
ax.set_ylim(54.5, 68.8)
ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.50)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Panel (b): Misaligned Prior
ax = axes[1]
mis_map = [
    ("SP-PD",              get_curve("SP-PD", "wrong")),
    ("LLM-Select (Score)", get_curve("llm_select_score_wrong")),
    ("LLM-Lasso",          get_curve("llm_lasso_wrong")),
    ("LSP",                get_curve("lsp_adapted_wrong")),
    ("MEL",                get_curve("mel")),
    ("mRMR",               get_curve("mrmr_static")),
]
for name, vals in mis_map:
    ax.plot(NS, vals, **styles[name])
ax.set_title("(b) Misaligned Prior", pad=5)
ax.set_xlabel("Instances per Class ($n$)")
ax.set_xticks(NS)
ax.set_xlim(1.5, 32.5)
ax.set_ylim(54.5, 68.8)
ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.50)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.annotate("", xy=(30.0, 65.45), xytext=(30.0, 59.62),
            arrowprops=dict(arrowstyle="<->", color="#C62828", lw=0.85))
ax.text(29.0, 62.3, "+6.63 pp", fontsize=6.8, color="#C62828", fontweight="bold", ha="right", va="center")

# Panel (c): Aligned vs. Misaligned Trade-off at n=30 (clean 2-contour layout, zero clutter)
ax = axes[2]
ax.set_title("(c) Robustness Trade-off ($n = 30$)", pad=5)
ax.set_xlabel("Aligned BA (%)")
ax.set_ylabel("Misaligned BA (%)")
ax.set_xlim(64.5, 68.3)
ax.set_ylim(58.4, 67.3)

xs_diag = np.linspace(64.5, 68.3, 60)
ax.plot(xs_diag, xs_diag, color="#9E9E9E", ls="--", lw=0.75, zorder=1)
ax.text(64.58, 63.65, "$y = x$", fontsize=6.4, color="#757575", rotation=25)

# Only TWO clean iso-mean contours: SP-PD frontier (66.25%) and reference (65.00%)
for c_iso, col_i, lw_i, lbl_pos in [
    (66.25, "#C62828", 0.85, (67.35, 65.45)),
    (65.00, "#BDBDBD", 0.60, None),
]:
    ys_iso = 2.0 * c_iso - xs_diag
    mask = (ys_iso >= 58.4) & (ys_iso <= 67.3)
    if np.any(mask):
        ax.plot(xs_diag[mask], ys_iso[mask], color=col_i, ls=":", lw=lw_i, alpha=0.80, zorder=1)
        if lbl_pos is not None:
            ax.text(lbl_pos[0], lbl_pos[1], "Mean 66.25%", fontsize=5.9, color="#C62828", rotation=-22, ha="left", va="center")

# Subtle hollow gray dots for remaining prior-independent baselines on y=x
for m_bg in ("ga", "bpso", "eda", "relieff_static", "l1_lr_static", "rf_imp_static"):
    v = get_curve(m_bg)[-1]
    ax.scatter([v], [v], facecolor="white", edgecolor="#9E9E9E", s=16, marker="o", linewidth=0.75, zorder=2)

# Highlighted points: hollow baselines, solid SP-PD; every label in clean open white space
pts = [
    ("SP-PD",              get_curve("SP-PD", "correct")[-1],       get_curve("SP-PD", "wrong")[-1],       "#C62828", "o", True,  48, (66.84, 66.62), False),
    ("MEL",                get_curve("mel")[-1],                    get_curve("mel")[-1],                  "#2E7D32", "v", False, 30, (64.75, 66.65), True),
    ("mRMR",               get_curve("mrmr_static")[-1],            get_curve("mrmr_static")[-1],          "#E65100", "P", False, 30, (64.62, 62.55), True),
    ("LLM-Lasso",          get_curve("llm_lasso_correct")[-1],      get_curve("llm_lasso_wrong")[-1],      "#1565C0", "^", False, 32, (66.78, 64.25), True),
    ("LSP",                get_curve("lsp_adapted_correct")[-1],    get_curve("lsp_adapted_wrong")[-1],    "#00838F", "D", False, 26, (65.38, 63.35), True),
    ("$\\pi$BO",           get_curve("pibo_correct")[-1],           get_curve("pibo_wrong")[-1],           "#6D4C41", "X", False, 28, (66.55, 62.85), True),
    ("LLM-Select\n(Rank)", get_curve("gemini_static")[-1],          get_curve("wrong_static")[-1],         "#8E24AA", "p", False, 28, (65.18, 59.35), False),
    ("LLM-Select\n(Score)",get_curve("llm_select_score_correct")[-1],get_curve("llm_select_score_wrong")[-1],"#7B1FA2","s", False, 30, (66.78, 60.62), False),
]

for name, x_v, y_v, col, mk, filled, sz, (tx, ty), use_arrow in pts:
    fc = col if filled else "white"
    ec = "black" if filled else col
    lw_e = 0.75 if filled else 1.0
    ax.scatter([x_v], [y_v], facecolor=fc, edgecolor=ec, marker=mk, s=sz, linewidth=lw_e, zorder=8 if filled else 5)
    fw = "bold" if filled else "normal"
    fs = 6.5 if "\n" not in name else 6.1
    if use_arrow:
        ax.annotate(name, xy=(x_v, y_v), xytext=(tx, ty), fontsize=fs, color=col, fontweight=fw,
                    va="center", arrowprops=dict(arrowstyle="-", color=col, lw=0.55, alpha=0.75, shrinkA=1, shrinkB=3.5))
    else:
        ax.text(tx, ty, name, fontsize=fs, color=col, fontweight=fw, va="center")

ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.45)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.995),
           ncol=6, frameon=False, handlelength=1.9, columnspacing=1.05, handletextpad=0.4)

out_pdf = ROOT / "fig2_clinical_scaling_v3.pdf"
fig.savefig(out_pdf, format="pdf")
print("Saved", out_pdf)
