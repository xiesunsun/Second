"""Final publication-grade 2x2 IEEE TEVC Fig. 3 (100% numerically reconciled with Tables VI & VII)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/sunxie/paper-experiments-20261124")
abl_nc = pd.read_csv(ROOT / "results/sec4d_ablation_nc.tsv", sep="\t")
abl_dn = pd.read_csv(ROOT / "results/sec4d_ablation_down.tsv", sep="\t")
sens_l = pd.read_csv(ROOT / "results/sec4d_sens_lambda.tsv", sep="\t")
sens_r = pd.read_csv(ROOT / "results/sec4d_sens_R.tsv", sep="\t")
cost = pd.read_csv(ROOT / "results/llm_cost_ledger.tsv", sep="\t")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 8.0,
    "axes.labelsize": 8.2,
    "axes.titlesize": 8.6,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.44))
plt.subplots_adjust(left=0.118, right=0.985, top=0.935, bottom=0.105, wspace=0.25, hspace=0.40)

# ==============================================================================
# Panel (a): Diverging Component Ablation Impact (exact 2-dec match to Table VI)
# ==============================================================================
ax = axes[0, 0]
nc30 = abl_nc[abl_nc["n"] == 30].groupby(["variant", "prior"], as_index=False)["ba"].mean()
nc_m = (nc30.groupby("variant")["ba"].mean() * 100.0).round(2)
dn_m = (abl_dn.groupby("variant")[["lr", "rf"]].mean() * 100.0).round(2)

full_nc, full_lr, full_rf = nc_m["full_sppd"], dn_m.loc["full_sppd", "lr"], dn_m.loc["full_sppd", "rf"]
v_order = [
    ("wo_search", "w/o\nSearch*", True),
    ("wo_prior",  "w/o\nPrior*", True),
    ("wo_reg",    "w/o Prior\nReg.*", True),
    ("wo_rep",    "w/o Rep.\nPairing", False),
]
y_pos = np.arange(len(v_order))
h = 0.23
d_nc = [round(nc_m[k] - full_nc, 2) for k, _, _ in v_order]
d_lr = [round(dn_m.loc[k, "lr"] - full_lr, 2) for k, _, _ in v_order]
d_rf = [round(dn_m.loc[k, "rf"] - full_rf, 2) for k, _, _ in v_order]

ax.barh(y_pos + h, d_nc, height=h, color="#C62828", edgecolor="#C62828", linewidth=0.5, label="NC Mean", zorder=4)
ax.barh(y_pos,     d_lr, height=h, color="white",   edgecolor="#1565C0", linewidth=1.0, hatch="///", label="LR Mean", zorder=4)
ax.barh(y_pos - h, d_rf, height=h, color="#E0F2F1", edgecolor="#00695C", linewidth=0.9, label="RF Mean", zorder=4)

ax.axvline(0.0, color="#212121", lw=0.9, ls="-", zorder=5)
ax.set_yticks(y_pos)
ax.set_yticklabels([lbl for _, lbl, _ in v_order], fontsize=7.1, linespacing=0.95)
ax.invert_yaxis()
ax.set_xlim(-3.82, 0.22)
ax.set_xlabel("$\\Delta$ Mean BA vs. SP-PD (pp)")
ax.set_title("(a) Component Ablation Impact ($n = 30$)", pad=5)

for i, (vn, vl, vr) in enumerate(zip(d_nc, d_lr, d_rf)):
    worst = min(vn, vl, vr)
    if i == 0:
        ax.text(vn - 0.08, i + h, f"{vn:.2f}", fontsize=6.2, color="#C62828", ha="right", va="center", fontweight="bold")
        ax.text(vl - 0.08, i,     f"{vl:.2f}", fontsize=6.2, color="#1565C0", ha="right", va="center")
        ax.text(vr - 0.08, i - h, f"{vr:.2f}", fontsize=6.2, color="#00695C", ha="right", va="center")
    else:
        ax.text(worst - 0.09, i, f"NC {vn:.2f} / LR {vl:.2f} / RF {vr:.2f}", fontsize=6.0, color="#37474F", ha="right", va="center")

ax.text(-3.72, 3.42, "* $p < 0.05$ on all 3 classifiers", fontsize=6.1, color="#424242", style="italic")
ax.legend(loc="lower left", bbox_to_anchor=(0.01, 0.14), frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CFD8DC", handlelength=1.4, fontsize=6.4)
ax.grid(True, axis="x", linestyle=":", linewidth=0.5, alpha=0.55)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# ==============================================================================
# Panel (b): Prior-Regularization Sensitivity Wedge (lambda in [0, 1.5], R=10)
# ==============================================================================
ax = axes[0, 1]
pl = sens_l.groupby(["lam", "prior"], as_index=False)["nc"].mean().pivot(index="lam", columns="prior", values="nc") * 100.0
lams = pl.index.to_numpy(dtype=float)
y_aln = pl["correct"].to_numpy()
y_mis = pl["wrong"].to_numpy()
y_mean = 0.5 * (y_aln + y_mis)

ax.fill_between(lams, y_mis, y_aln, color="#90CAF9", alpha=0.22, zorder=1, label="Prior-shift gap")
ax.axvspan(0.25, 0.75, color="#FFF8E1", alpha=0.85, zorder=0)
ax.axvline(0.50, color="#C62828", ls=":", lw=0.85, alpha=0.8, zorder=2)

ax.plot(lams, y_aln, color="#1565C0", marker="^", mfc="white", mec="#1565C0", mew=1.0, ls="--", lw=1.3, ms=4.4, label="Aligned BA", zorder=4)
ax.plot(lams, y_mis, color="#00695C", marker="D", mfc="white", mec="#00695C", mew=1.0, ls="-.", lw=1.3, ms=4.0, label="Misaligned BA", zorder=4)
ax.plot(lams, y_mean, color="#C62828", marker="o", mfc="#C62828", mec="#C62828", mew=0.9, ls="-", lw=1.9, ms=4.8, label="Mean BA", zorder=6)

ax.annotate("Peak $\\lambda = 0.5$ (66.25%)", xy=(0.50, 66.28), xytext=(0.04, 67.18),
            fontsize=6.4, color="#C62828", fontweight="bold", ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color="#C62828", lw=0.8, connectionstyle="arc3,rad=-0.12"))

ax.set_title("(b) Regularization Trade-off Wedge ($R = 10$)", pad=5)
ax.set_xlabel("Prior-Regularization Weight ($\\lambda$)")
ax.set_ylabel("Balanced Accuracy (%)")
ax.set_xticks(lams)
ax.set_xlim(-0.06, 1.56)
ax.set_ylim(63.75, 67.55)
ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CFD8DC", ncol=2, columnspacing=0.7, handlelength=1.5, fontsize=6.1)
ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.50)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# ==============================================================================
# Panel (c): Repeated-Pairing Accuracy-Runtime Knee (R = 1 -> 20, lambda=0.5)
# ==============================================================================
ax = axes[1, 0]
pr_nc = sens_r.groupby(["R", "prior"], as_index=False)["nc"].mean().pivot(index="R", columns="prior", values="nc").mean(axis=1) * 100.0
pr_lr = sens_r.groupby(["R", "prior"], as_index=False)["lr"].mean().pivot(index="R", columns="prior", values="lr").mean(axis=1) * 100.0
pr_s = sens_r.groupby("R")["deliv_ms"].mean() / 1000.0

Rs = pr_nc.index.to_numpy(dtype=int)
xs_t = pr_s.to_numpy()
ys_nc = pr_nc.to_numpy()
ys_lr = pr_lr.to_numpy()

ax.axvline(xs_t[3], color="#C62828", ls=":", lw=0.85, alpha=0.75, zorder=1)
ax.plot(xs_t, ys_nc, color="#C62828", marker="o", mfc="#C62828", mec="#C62828", lw=1.8, ms=4.8, ls="-", label="NC Mean BA", zorder=6)
ax.plot(xs_t, ys_lr, color="#1565C0", marker="s", mfc="white",   mec="#1565C0", mew=1.0, lw=1.35, ms=4.4, ls="--", label="LR Mean BA", zorder=5)

ax.scatter([xs_t[3]], [ys_nc[3]], s=110, facecolors="none", edgecolors="#C62828", linewidths=1.1, zorder=7)
ax.scatter([xs_t[3]], [ys_lr[3]], s=95,  facecolors="none", edgecolors="#1565C0", linewidths=1.0, zorder=6)

offsets = {1: (0.04, -0.075), 3: (0.04, 0.055), 5: (0.04, -0.075), 10: (0.06, 0.045), 20: (-0.06, 0.06)}
for rv, xt, yn in zip(Rs, xs_t, ys_nc):
    dx, dy = offsets[rv]
    lbl = f"$R={rv}$ (default)" if rv == 10 else f"$R={rv}$"
    ha = "right" if rv == 20 else "left"
    fw = "bold" if rv == 10 else "normal"
    col = "#C62828" if rv == 10 else "#37474F"
    ax.text(xt + dx, yn + dy, lbl, fontsize=6.4, color=col, fontweight=fw, ha=ha, va="center")

ax.text(0.06, 65.86, "Search CPU: 0.35 s", fontsize=6.2, color="#455A64", style="italic")
ax.set_title("(c) Repeated-Pairing Accuracy-Time Knee ($\\lambda = 0.5$)", pad=5)
ax.set_xlabel("Delivery CPU Time per Run (s)")
ax.set_ylabel("Mean Balanced Accuracy (%)")
ax.set_xlim(0.0, 2.15)
ax.set_ylim(65.58, 66.36)
ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CFD8DC", handlelength=1.6, fontsize=6.6)
ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.50)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# ==============================================================================
# Panel (d): O(1) vs. O(p) LLM Peak Output Token Scaling (exact match Table VII)
# ==============================================================================
ax = axes[1, 1]
c_piv = cost.groupby(["task", "p", "method"], as_index=False)["output_tokens"].max().pivot(index=["task", "p"], columns="method", values="output_tokens").reset_index().sort_values("p")
ps = c_piv["p"].to_numpy(dtype=float)
tok_sp = c_piv["ours_top8_rank"].to_numpy(dtype=float)
tok_ls = c_piv["llm_lasso"].to_numpy(dtype=float)
tok_sc = c_piv["llm_select_score"].to_numpy(dtype=float)

ax.fill_between(ps, tok_sp, tok_sc, color="#CE93D8", alpha=0.16, zorder=1)
ax.loglog(ps, tok_sc, color="#7B1FA2", marker="s", mfc="white", mec="#7B1FA2", mew=1.0, ls="--", lw=1.35, ms=4.2, label="LLM-Select (Score) / LSP, $O(p)$", zorder=4)
ax.loglog(ps, tok_ls, color="#1565C0", marker="^", mfc="white", mec="#1565C0", mew=1.0, ls="-.", lw=1.35, ms=4.4, label="LLM-Lasso, $O(p)$", zorder=5)
ax.loglog(ps, tok_sp, color="#C62828", marker="o", mfc="#C62828", mec="#C62828", mew=0.9, ls="-", lw=1.9, ms=4.6, label="SP-PD (1 call), $O(1)$", zorder=7)

ax.annotate("", xy=(753, 8900), xytext=(753, 86),
            arrowprops=dict(arrowstyle="<->", color="#C62828", lw=0.95))
ax.text(620, 800, "153$\\times$ fewer\n(68 vs. 10,387)", fontsize=6.4, color="#C62828", fontweight="bold", ha="right", va="center")
ax.text(753, 28, "voice ($p{=}753$)", fontsize=6.0, color="#424242", ha="right", va="center")

ax.set_title("(d) LLM Peak Output Token Scaling ($p = 8 \\dots 753$)", pad=5)
ax.set_xlabel("Number of Features ($p$, log scale)")
ax.set_ylabel("Peak Output Tokens (log)")
ax.set_xlim(6.5, 1100)
ax.set_ylim(20, 22000)
ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CFD8DC", handlelength=1.6, fontsize=6.2)
ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.55)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

out_pdf = ROOT / "fig3_ablation_sensitivity_cost.pdf"
fig.savefig(out_pdf, format="pdf")
print("Saved", out_pdf)
