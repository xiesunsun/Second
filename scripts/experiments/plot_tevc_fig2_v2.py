"""Generate publication-grade IEEE TEVC Fig. 2 (3-panel vector PDF, STIX serif, zero label overlap)."""
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

fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.38), gridspec_kw={"width_ratios": [1.0, 1.0, 1.15]})
plt.subplots_adjust(left=0.068, right=0.985, top=0.80, bottom=0.19, wspace=0.28)

styles = {
    "SP-PD":              dict(color="#C62828", marker="o", lw=1.9, ms=4.8, ls="-",  zorder=6),
    "LLM-Select (Score)": dict(color="#7B1FA2", marker="s", lw=1.35, ms=4.0, ls="--", zorder=4),
    "LLM-Lasso":          dict(color="#1565C0", marker="^", lw=1.35, ms=4.2, ls="-.", zorder=5),
    "LSP":                dict(color="#00838F", marker="D", lw=1.25, ms=3.7, ls=":",  zorder=4),
    "MEL":                dict(color="#2E7D32", marker="v", lw=1.25, ms=4.0, ls="-",  zorder=3),
    "mRMR":               dict(color="#EF6C00", marker="P", lw=1.15, ms=3.9, ls="--", zorder=2),
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
ax.set_ylim(54.0, 68.8)
ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.55)
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
ax.set_ylim(54.0, 68.8)
ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.55)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
# Clean vertical double-arrow showing +6.63 pp recovery over LLM-Select (Score) at n=30
ax.annotate("", xy=(30.0, 65.55), xytext=(30.0, 59.52),
            arrowprops=dict(arrowstyle="<->", color="#C62828", lw=0.85))
ax.text(29.1, 62.2, "+6.63 pp", fontsize=6.7, color="#C62828", fontweight="bold", ha="right", va="center")

# Panel (c): Aligned vs. Misaligned Pareto Trade-off at n=30
ax = axes[2]
ax.set_title("(c) Robustness Trade-off ($n = 30$)", pad=5)
ax.set_xlabel("Aligned BA (%)")
ax.set_ylabel("Misaligned BA (%)")
ax.set_xlim(64.5, 68.3)
ax.set_ylim(58.3, 67.2)

xs_diag = np.linspace(64.5, 68.3, 60)
ax.plot(xs_diag, xs_diag, color="#9E9E9E", ls="--", lw=0.75, zorder=1)
ax.text(64.58, 63.72, "$y = x$", fontsize=6.4, color="#757575", rotation=26)

for c_iso in (63.5, 65.0, 65.8, 66.25):
    ys_iso = 2.0 * c_iso - xs_diag
    mask = (ys_iso >= 58.3) & (ys_iso <= 67.2)
    if np.any(mask):
        lw_i = 0.85 if abs(c_iso - 66.25) < 1e-6 else 0.55
        col_i = "#C62828" if abs(c_iso - 66.25) < 1e-6 else "#BDBDBD"
        ax.plot(xs_diag[mask], ys_iso[mask], color=col_i, ls=":", lw=lw_i, alpha=0.75, zorder=1)

# Subtle background non-LLM dots on y=x
for m_bg in ("ga", "bpso", "eda", "relieff_static", "l1_lr_static", "rf_imp_static"):
    v = get_curve(m_bg)[-1]
    ax.scatter([v], [v], color="#9E9E9E", s=13, marker="o", alpha=0.60, zorder=2)

# Highlighted points with crisp non-overlapping label positions (and subtle leader arrows where helpful)
pts = [
    ("SP-PD",              get_curve("SP-PD", "correct")[-1],       get_curve("SP-PD", "wrong")[-1],       "#C62828", "o", 46, (66.88, 66.48), False),
    ("MEL",                get_curve("mel")[-1],                    get_curve("mel")[-1],                  "#2E7D32", "v", 28, (64.72, 66.45), True),
    ("mRMR",               get_curve("mrmr_static")[-1],            get_curve("mrmr_static")[-1],          "#EF6C00", "P", 26, (64.62, 62.65), True),
    ("LLM-Lasso",          get_curve("llm_lasso_correct")[-1],      get_curve("llm_lasso_wrong")[-1],      "#1565C0", "^", 30, (66.82, 64.92), True),
    ("LSP",                get_curve("lsp_adapted_correct")[-1],    get_curve("lsp_adapted_wrong")[-1],    "#00838F", "D", 24, (65.48, 63.65), True),
    ("$\\pi$BO",           get_curve("pibo_correct")[-1],           get_curve("pibo_wrong")[-1],           "#6D4C41", "X", 26, (66.75, 63.75), True),
    ("LLM-Select\n(Rank)", get_curve("gemini_static")[-1],          get_curve("wrong_static")[-1],         "#8E24AA", "p", 26, (65.18, 59.35), False),
    ("LLM-Select\n(Score)",get_curve("llm_select_score_correct")[-1],get_curve("llm_select_score_wrong")[-1],"#7B1FA2","s", 28, (66.78, 60.68), False),
]

for name, x_v, y_v, col, mk, sz, (tx, ty), use_arrow in pts:
    edge = "black" if name == "SP-PD" else "white"
    lw_e = 0.7 if name == "SP-PD" else 0.4
    ax.scatter([x_v], [y_v], color=col, marker=mk, s=sz, edgecolor=edge, linewidth=lw_e, zorder=6 if name=="SP-PD" else 4)
    fw = "bold" if name == "SP-PD" else "normal"
    fs = 6.5 if "\n" not in name else 6.1
    if use_arrow:
        ax.annotate(name, xy=(x_v, y_v), xytext=(tx, ty), fontsize=fs, color=col, fontweight=fw,
                    va="center", arrowprops=dict(arrowstyle="-", color=col, lw=0.55, alpha=0.75, shrinkA=1, shrinkB=3))
    else:
        ax.text(tx, ty, name, fontsize=fs, color=col, fontweight=fw, va="center")

ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.45)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.995),
           ncol=6, frameon=False, handlelength=1.8, columnspacing=1.1, handletextpad=0.4)

out_pdf = ROOT / "fig2_clinical_scaling_v2.pdf"
fig.savefig(out_pdf, format="pdf")
print("Saved", out_pdf)
