"""Publication-grade 2x2 IEEE TEVC Fig. 2 (symmetric with Fig. 3, STIX serif, hollow baselines, B&W readable):
  (a) Evaluation-Budget Acceleration (B in {4, 8, 16, 32, 64, 128}, n = 30)
  (b) Aligned Prior Sample-Size Scaling (n in {3, 10, 20, 30}, B = 128)
  (c) Misaligned Prior Sample-Size Scaling (n in {3, 10, 20, 30}, B = 128)
  (d) Aligned vs. Misaligned Robustness Trade-off (n = 30, B = 128)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/sunxie/paper-experiments-20261124")
base = pd.read_csv(ROOT / "results/clinical_12_per_dataset_nc.tsv", sep="\t")
eb = pd.read_csv(ROOT / "results/clinical_schemeA_candidates_nc.tsv", sep="\t")
b_sp = pd.read_csv(ROOT / "results/clinical_sppd_budget_curves.tsv", sep="\t")
b_bl = pd.read_csv(ROOT / "results/clinical_non_llm_curves.tsv", sep="\t")

RULE = "eb_k4_0.50"
NS = [3, 10, 20, 30]
BS = [4, 8, 16, 32, 64, 128]

def get_n_curve(method, prior=None):
    if method == "SP-PD":
        sub = eb[(eb["rule"] == RULE) & (eb["prior"] == prior)]
        return [float(sub[sub["n"] == n]["ba"].mean()) * 100.0 for n in NS]
    sub = base[base["method"] == method]
    return [float(sub[sub["n"] == n]["b128"].mean()) * 100.0 for n in NS]

def get_b_curve(method):
    if method.startswith("sppd_"):
        sub = b_sp[(b_sp["n"] == 30) & (b_sp["method"] == method)]
        return [float(sub[sub["budget"] == b]["ba"].mean()) * 100.0 for b in BS]
    sub = b_bl[(b_bl["panel"] == "all_twelve") & (b_bl["n"] == 30) & (b_bl["method"] == method)]
    return [float(sub[sub["budget"] == b]["ba"].values[0]) * 100.0 for b in BS]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 8.0,
    "axes.labelsize": 8.2,
    "axes.titlesize": 8.6,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 6.6,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.46))
plt.subplots_adjust(left=0.078, right=0.985, top=0.935, bottom=0.105, wspace=0.25, hspace=0.40)

# Panel (a): Evaluation-Budget Acceleration (B = 4..128, n = 30)
ax = axes[0, 0]
c_sp_a = get_b_curve("sppd_correct")
c_sp_m = get_b_curve("sppd_wrong")
c_sp_0 = get_b_curve("sppd_wo_prior")
c_mel  = get_b_curve("mel")
c_ga   = get_b_curve("ga")
c_pm   = get_b_curve("pibo_wrong")

ax.axhline(c_sp_a[0], color="#C62828", ls=":", lw=0.85, alpha=0.65, zorder=1)
ax.plot(BS, c_sp_a, color="#C62828", marker="o", mfc="#C62828", mec="#C62828", mew=0.9, ls="-",  lw=1.9, ms=4.7, label="SP-PD (Aln.)", zorder=8)
ax.plot(BS, c_sp_m, color="#D84315", marker="^", mfc="#D84315", mec="#D84315", mew=0.9, ls="--", lw=1.55, ms=4.5, label="SP-PD (Mis.)", zorder=7)
ax.plot(BS, c_sp_0, color="#455A64", marker="D", mfc="white",   mec="#455A64", mew=0.95, ls=":", lw=1.25, ms=4.0, label="SP-PD (w/o Prior)", zorder=5)
ax.plot(BS, c_mel,  color="#2E7D32", marker="v", mfc="white",   mec="#2E7D32", mew=1.0, ls="-",  lw=1.25, ms=4.4, label="MEL", zorder=4)
ax.plot(BS, c_ga,   color="#7B1FA2", marker="s", mfc="white",   mec="#7B1FA2", mew=1.0, ls="-.", lw=1.2,  ms=4.1, label="GA", zorder=3)
ax.plot(BS, c_pm,   color="#6D4C41", marker="X", mfc="white",   mec="#6D4C41", mew=0.95, ls="--", lw=1.15, ms=4.5, label="$\\pi$BO (Mis.)", zorder=3)

ax.annotate("$B{=}4$ beats all EAs\nat $B{=}128$ (32$\\times$)", xy=(4, c_sp_a[0]), xytext=(10.5, 67.25),
            fontsize=6.2, color="#C62828", fontweight="bold", ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color="#C62828", lw=0.75, connectionstyle="arc3,rad=0.14"))

ax.set_xscale("log", base=2)
ax.set_xticks(BS)
ax.set_xticklabels([str(b) for b in BS])
ax.set_xlim(3.3, 155)
ax.set_ylim(61.4, 67.95)
ax.set_title("(a) Budget Acceleration ($n = 30$)", pad=5)
ax.set_xlabel("Evaluation Budget ($B$, log$_2$ scale)")
ax.set_ylabel("Balanced Accuracy (%)")
ax.legend(loc="lower right", ncol=2, frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CFD8DC", columnspacing=0.65, handlelength=1.45, fontsize=5.9)
ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.50)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

styles_n = {
    "SP-PD":              dict(color="#C62828", marker="o", mfc="#C62828", mec="#C62828", mew=0.9, lw=1.9, ms=4.7, ls="-",  zorder=8),
    "LLM-Select (Score)": dict(color="#7B1FA2", marker="s", mfc="white",   mec="#7B1FA2", mew=1.0, lw=1.3, ms=4.2, ls="--", zorder=5),
    "LLM-Lasso":          dict(color="#1565C0", marker="^", mfc="white",   mec="#1565C0", mew=1.0, lw=1.3, ms=4.4, ls="-.", zorder=6),
    "LSP":                dict(color="#00838F", marker="D", mfc="white",   mec="#00838F", mew=1.0, lw=1.2, ms=3.9, ls=":",  zorder=5),
    "MEL":                dict(color="#2E7D32", marker="v", mfc="white",   mec="#2E7D32", mew=1.0, lw=1.25, ms=4.3, ls="-", zorder=4),
    "mRMR":               dict(color="#E65100", marker="P", mfc="white",   mec="#E65100", mew=0.9, lw=1.15, ms=4.3, ls="--", zorder=3),
}

# Panel (b): Aligned Prior Sample-Size Scaling
ax = axes[0, 1]
for name, vals in [
    ("SP-PD",              get_n_curve("SP-PD", "correct")),
    ("LLM-Select (Score)", get_n_curve("llm_select_score_correct")),
    ("LLM-Lasso",          get_n_curve("llm_lasso_correct")),
    ("LSP",                get_n_curve("lsp_adapted_correct")),
    ("MEL",                get_n_curve("mel")),
    ("mRMR",               get_n_curve("mrmr_static")),
]:
    ax.plot(NS, vals, label=name, **styles_n[name])
ax.set_title("(b) Aligned Prior Scaling ($B = 128$)", pad=5)
ax.set_xlabel("Instances per Class ($n$)")
ax.set_ylabel("Balanced Accuracy (%)")
ax.set_xticks(NS)
ax.set_xlim(1.5, 31.8)
ax.set_ylim(54.5, 68.8)
ax.legend(loc="lower right", ncol=2, frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CFD8DC", columnspacing=0.65, handlelength=1.45, fontsize=5.9)
ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.50)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Panel (c): Misaligned Prior Sample-Size Scaling
ax = axes[1, 0]
for name, vals in [
    ("SP-PD",              get_n_curve("SP-PD", "wrong")),
    ("LLM-Select (Score)", get_n_curve("llm_select_score_wrong")),
    ("LLM-Lasso",          get_n_curve("llm_lasso_wrong")),
    ("LSP",                get_n_curve("lsp_adapted_wrong")),
    ("MEL",                get_n_curve("mel")),
    ("mRMR",               get_n_curve("mrmr_static")),
]:
    ax.plot(NS, vals, label=name, **styles_n[name])
ax.annotate("", xy=(30.0, 65.45), xytext=(30.0, 59.62),
            arrowprops=dict(arrowstyle="<->", color="#C62828", lw=0.9))
ax.text(29.0, 62.35, "+6.63 pp", fontsize=6.8, color="#C62828", fontweight="bold", ha="right", va="center")
ax.set_title("(c) Misaligned Prior Scaling ($B = 128$)", pad=5)
ax.set_xlabel("Instances per Class ($n$)")
ax.set_ylabel("Balanced Accuracy (%)")
ax.set_xticks(NS)
ax.set_xlim(1.5, 32.5)
ax.set_ylim(54.5, 68.8)
ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.50)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Panel (d): Robustness Trade-off (n = 30, B = 128)
ax = axes[1, 1]
ax.set_title("(d) Robustness Trade-off ($n = 30$, $B = 128$)", pad=5)
ax.set_xlabel("Aligned BA (%)")
ax.set_ylabel("Misaligned BA (%)")
ax.set_xlim(64.5, 68.3)
ax.set_ylim(58.4, 67.3)

xs_diag = np.linspace(64.5, 68.3, 60)
ax.plot(xs_diag, xs_diag, color="#9E9E9E", ls="--", lw=0.75, zorder=1)
ax.text(64.58, 63.65, "$y = x$", fontsize=6.4, color="#757575", rotation=22)

for c_iso, col_i, lw_i, lbl_pos in [
    (66.25, "#C62828", 0.85, (67.32, 65.45)),
    (65.00, "#BDBDBD", 0.60, None),
]:
    ys_iso = 2.0 * c_iso - xs_diag
    mask = (ys_iso >= 58.4) & (ys_iso <= 67.3)
    if np.any(mask):
        ax.plot(xs_diag[mask], ys_iso[mask], color=col_i, ls=":", lw=lw_i, alpha=0.80, zorder=1)
        if lbl_pos is not None:
            ax.text(lbl_pos[0], lbl_pos[1], "Mean 66.25%", fontsize=6.0, color="#C62828", rotation=-20, ha="left", va="center")

for m_bg in ("ga", "bpso", "eda", "relieff_static", "l1_lr_static", "rf_imp_static"):
    v = get_n_curve(m_bg)[-1]
    ax.scatter([v], [v], facecolor="white", edgecolor="#9E9E9E", s=16, marker="o", linewidth=0.75, zorder=2)

pts = [
    ("SP-PD",              get_n_curve("SP-PD", "correct")[-1],       get_n_curve("SP-PD", "wrong")[-1],       "#C62828", "o", True,  48, (66.84, 66.62), False),
    ("MEL",                get_n_curve("mel")[-1],                    get_n_curve("mel")[-1],                  "#2E7D32", "v", False, 30, (64.75, 66.65), True),
    ("mRMR",               get_n_curve("mrmr_static")[-1],            get_n_curve("mrmr_static")[-1],          "#E65100", "P", False, 30, (64.62, 62.55), True),
    ("LLM-Lasso",          get_n_curve("llm_lasso_correct")[-1],      get_n_curve("llm_lasso_wrong")[-1],      "#1565C0", "^", False, 32, (66.78, 64.25), True),
    ("LSP",                get_n_curve("lsp_adapted_correct")[-1],    get_n_curve("lsp_adapted_wrong")[-1],    "#00838F", "D", False, 26, (65.38, 63.35), True),
    ("$\\pi$BO",           get_n_curve("pibo_correct")[-1],           get_n_curve("pibo_wrong")[-1],           "#6D4C41", "X", False, 28, (66.55, 62.85), True),
    ("LLM-Select\n(Rank)", get_n_curve("gemini_static")[-1],          get_n_curve("wrong_static")[-1],         "#8E24AA", "p", False, 28, (65.18, 59.35), False),
    ("LLM-Select\n(Score)",get_n_curve("llm_select_score_correct")[-1],get_n_curve("llm_select_score_wrong")[-1],"#7B1FA2","s", False, 30, (66.78, 60.62), False),
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

out_pdf = ROOT / "fig2_clinical_scaling_v4.pdf"
fig.savefig(out_pdf, format="pdf")
print("Saved", out_pdf)
