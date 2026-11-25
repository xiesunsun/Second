"""Production evaluation of Scheme A (strict 1-call O(1) Rank prior + Empirical-Bayes Repeated-Pairing Delivery)
across all 12 clinical datasets x 20 repeats x n in {3, 10, 20, 30} for NC and Downstream LR/RF (n=30).
Compares eb_any (lam=0.50, 0.60) vs eb_k4 (full K=4 shortlist, lam=0.50, 0.60),
computes exact Wilcoxon signed-rank p-values against all 14 baselines, and saves frozen TSVs.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
ROOT = Path("/home/sunxie/paper-experiments-20261124")
sys.path.insert(0, str(ROOT))
import run_clinical_non_llm as clin
from test_eb_delivery import repeated_lopo


def eval_job(job):
    t, r = job
    q_outer, v_outer, _ = clin.load_role(t, "outer")
    orders, p = clin.load_gemini_orders(t)
    out_nc, out_dn = [], []

    for n in clin.NS:
        x, y, ids = clin.get_donors(t, r, n)
        rs = clin.seed("clinical-EA-20261123", t, r, n)
        d_outer = clin.margins(x, y, q_outer)
        d_outer_t = np.ascontiguousarray(d_outer.T)
        outer_cache = {}
        def score_ba(mask):
            mask = tuple(mask)
            if mask not in outer_cache:
                pred = d_outer_t[list(mask)].sum(0) > 0
                outer_cache[mask] = float(clin.measures(pred, v_outer)[0])
            return outer_cache[mask]
        down = clin.Downstream(x, y, q_outer, v_outer, clin.seed("clinical-readout-20261123", t, r, n)) if n == 30 else None

        for prior_kind in ("correct", "wrong"):
            order = orders[prior_kind]
            bk, st, ret, _ = clin.run_frozen_ext(x, y, clin.CAP, 128, order, rs)
            path = st["path"]
            a_mask = tuple(bk.masks[ret["anchor"]])
            eval_ids = list(dict.fromkeys(path))

            y_sign = np.where(y == 1, 1.0, -1.0)
            col_m = bk.margin * y_sign[:, None]
            acc_sc = {j: bk.scores[j]["fold"] for j in eval_ids}
            cont_sc = {}
            for j in eval_ids:
                sv = col_m[:, list(bk.masks[j])].sum(axis=1)
                cont_sc[j] = float(np.tanh(sv / (0.5 * (np.std(sv) + 1e-6))).mean())

            topK = set(int(f) for f in order[:clin.CAP])
            top2K = set(int(f) for f in order[:2 * clin.CAP])
            def phi_fullK(m):
                return sum(1.0 if f in topK else (0.5 if f in top2K else 0.0) for f in m) / float(clin.CAP)

            top_pure_any = sorted(eval_ids, key=lambda j: (-(acc_sc[j] + 0.15 * cont_sc[j]), -len(bk.masks[j])))[:8]
            top_tilt_any = sorted(eval_ids, key=lambda j: (-(acc_sc[j] + 0.10 * cont_sc[j] + 0.25 * phi_fullK(bk.masks[j])), -len(bk.masks[j])))[:6]
            sl_any = list(dict.fromkeys([a_mask] + [tuple(bk.masks[j]) for j in top_pure_any + top_tilt_any]))

            full_ids = [j for j in eval_ids if len(bk.masks[j]) == clin.CAP]
            top_pure_k4 = sorted(full_ids, key=lambda j: -(acc_sc[j] + 0.15 * cont_sc[j]))[:10]
            top_tilt_k4 = sorted(full_ids, key=lambda j: -(acc_sc[j] + 0.10 * cont_sc[j] + 0.25 * phi_fullK(bk.masks[j])))[:6]
            sl_k4 = list(dict.fromkeys([a_mask] + [tuple(bk.masks[j]) for j in top_pure_k4 + top_tilt_k4]))

            sl_union = list(dict.fromkeys(sl_any + sl_k4))
            rep_preds, rep_smarg = repeated_lopo(x, y, sl_union, rs, n_reps=10)
            rep_acc = {m: float(rep_preds[m].mean()) for m in sl_union}
            rep_sm = {m: float(np.tanh(rep_smarg[m] / (0.5 * (np.std(rep_smarg[m]) + 1e-6))).mean()) for m in sl_union}

            dim_factor = np.sqrt(np.log(max(2.0, p / 2.0))) / np.sqrt(2.0 * n)
            cands = {}
            for lam in (0.50, 0.60):
                cands[f"eb_any_{lam:.2f}"] = max(sl_any, key=lambda m: (rep_acc[m] + 0.03 * rep_sm[m] + lam * dim_factor * phi_fullK(m), len(m)))
                cands[f"eb_k4_{lam:.2f}"] = max(sl_k4, key=lambda m: (rep_acc[m] + 0.03 * rep_sm[m] + lam * dim_factor * phi_fullK(m), phi_fullK(m)))

            for rname, m in cands.items():
                out_nc.append((t, r, n, prior_kind, rname, score_ba(m), len(m)))
            if down is not None:
                for rname, m in cands.items():
                    res_d = down.scores(m)
                    out_dn.append((t, r, prior_kind, rname, float(res_d["lr"][0]), float(res_d["rf"][0]), len(m)))
    return out_nc, out_dn


def wilcoxon_p(a, b):
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if np.all(np.abs(diff) < 1e-12):
        return 1.0
    try:
        return float(wilcoxon(diff, alternative="two-sided", zero_method="wilcox").pvalue)
    except Exception:
        return 1.0


if __name__ == "__main__":
    jobs = [(t, r) for t in clin.ALL_TASKS for r in range(clin.REPEATS)]
    with ProcessPoolExecutor(max_workers=20) as ex:
        results = list(ex.map(eval_job, jobs))
    nc_flat = [x for a, _ in results for x in a]
    dn_flat = [x for _, b in results for x in b]

    df_nc = pd.DataFrame(nc_flat, columns=["t", "r", "n", "prior", "rule", "ba", "sz"])
    df_dn = pd.DataFrame(dn_flat, columns=["t", "r", "prior", "rule", "lr", "rf", "sz"])

    agg_nc = df_nc.groupby(["t", "n", "prior", "rule"], as_index=False)[["ba", "sz"]].mean()
    agg_dn = df_dn.groupby(["t", "prior", "rule"], as_index=False)[["lr", "rf", "sz"]].mean()

    agg_nc.to_csv(ROOT / "results/clinical_schemeA_candidates_nc.tsv", sep="\t", index=False)
    agg_dn.to_csv(ROOT / "results/clinical_schemeA_candidates_down.tsv", sep="\t", index=False)

    piv = agg_nc.groupby(["prior", "rule", "n"], as_index=False)["ba"].mean().pivot(index="rule", columns=["prior", "n"], values="ba") * 100.0
    piv["Aln_avg"] = piv["correct"].mean(axis=1)
    piv["Mis_avg"] = piv["wrong"].mean(axis=1)
    piv["All_avg"] = 0.5 * (piv["Aln_avg"] + piv["Mis_avg"])
    piv["Mean_n30"] = 0.5 * (piv[("correct", 30)] + piv[("wrong", 30)])
    print("=== SCHEME A CANDIDATES (1-CALL RANK PRIOR): NC BA (%) ===")
    print(piv.round(2).to_string())

    ad = agg_dn.groupby(["prior", "rule"], as_index=False)[["lr", "rf", "sz"]].mean()
    pd_lr = ad.pivot(index="rule", columns="prior", values="lr") * 100.0
    pd_rf = ad.pivot(index="rule", columns="prior", values="rf") * 100.0
    comb_d = pd.DataFrame({
        "LR_Aln": pd_lr["correct"], "LR_Mis": pd_lr["wrong"], "LR_Mean": 0.5*(pd_lr["correct"]+pd_lr["wrong"]),
        "RF_Aln": pd_rf["correct"], "RF_Mis": pd_rf["wrong"], "RF_Mean": 0.5*(pd_rf["correct"]+pd_rf["wrong"]),
    })
    print("\n=== SCHEME A CANDIDATES: DOWNSTREAM LR & RF AT n=30 ===")
    print(comb_d.round(2).to_string())

    base_nc = pd.read_csv(ROOT / "results/clinical_12_per_dataset_nc.tsv", sep="\t")
    base_dn = pd.read_csv(ROOT / "results/clinical_12_per_dataset_down.tsv", sep="\t")
    tasks = sorted(clin.ALL_TASKS)

    non_llm = ["ga", "bpso", "eda", "mel", "mrmr_static", "relieff_static", "l1_lr_static", "rf_imp_static"]
    prior_m = [
        ("pibo", "pibo_correct", "pibo_wrong"),
        ("llm_lasso", "llm_lasso_correct", "llm_lasso_wrong"),
        ("lsp", "lsp_adapted_correct", "lsp_adapted_wrong"),
        ("llm_rank", "gemini_static", "wrong_static"),
        ("llm_score", "llm_select_score_correct", "llm_select_score_wrong"),
    ]

    for rname in ("eb_any_0.50", "eb_k4_0.50"):
        print(f"\n=== WILCOXON SIGNIFICANCE SUMMARY FOR {rname} (p < 0.05 marked *) ===")
        sub_c = agg_nc[(agg_nc["rule"] == rname) & (agg_nc["prior"] == "correct")]
        sub_w = agg_nc[(agg_nc["rule"] == rname) & (agg_nc["prior"] == "wrong")]
        rows = []
        for m in non_llm:
            rec = {"method": m}
            for n in clin.NS:
                va_c = [sub_c[(sub_c["t"] == t) & (sub_c["n"] == n)]["ba"].values[0] for t in tasks]
                vb = [base_nc[(base_nc["t"] == t) & (base_nc["n"] == n) & (base_nc["method"] == m)]["b128"].values[0] for t in tasks]
                pc = wilcoxon_p(va_c, vb)
                star = "*" if (np.mean(va_c) > np.mean(vb) and pc < 0.05) else ""
                rec[f"A_n{n}"] = f"{np.mean(vb)*100:.2f}{star}"
            va_m30 = [0.5*(sub_c[(sub_c["t"] == t) & (sub_c["n"] == 30)]["ba"].values[0] + sub_w[(sub_w["t"] == t) & (sub_w["n"] == 30)]["ba"].values[0]) for t in tasks]
            vb30 = [base_nc[(base_nc["t"] == t) & (base_nc["n"] == 30) & (base_nc["method"] == m)]["b128"].values[0] for t in tasks]
            pm = wilcoxon_p(va_m30, vb30)
            star_m = "*" if (np.mean(va_m30) > np.mean(vb30) and pm < 0.05) else ""
            rec["Mean30"] = f"{np.mean(vb30)*100:.2f}{star_m} (p={pm:.3f})"
            rows.append(rec)
        for label, mc, mw in prior_m:
            rec = {"method": label}
            for n in clin.NS:
                va_c = [sub_c[(sub_c["t"] == t) & (sub_c["n"] == n)]["ba"].values[0] for t in tasks]
                va_w = [sub_w[(sub_w["t"] == t) & (sub_w["n"] == n)]["ba"].values[0] for t in tasks]
                vbc = [base_nc[(base_nc["t"] == t) & (base_nc["n"] == n) & (base_nc["method"] == mc)]["b128"].values[0] for t in tasks]
                vbw = [base_nc[(base_nc["t"] == t) & (base_nc["n"] == n) & (base_nc["method"] == mw)]["b128"].values[0] for t in tasks]
                pc = wilcoxon_p(va_c, vbc)
                pw = wilcoxon_p(va_w, vbw)
                sc = "*" if (np.mean(va_c) > np.mean(vbc) and pc < 0.05) else ""
                sw = "*" if (np.mean(va_w) > np.mean(vbw) and pw < 0.05) else ""
                rec[f"A_n{n}"] = f"{np.mean(vbc)*100:.2f}{sc}"
                rec[f"M_n{n}"] = f"{np.mean(vbw)*100:.2f}{sw}"
            va_m30 = [0.5*(sub_c[(sub_c["t"] == t) & (sub_c["n"] == 30)]["ba"].values[0] + sub_w[(sub_w["t"] == t) & (sub_w["n"] == 30)]["ba"].values[0]) for t in tasks]
            vbm30 = [0.5*(base_nc[(base_nc["t"] == t) & (base_nc["n"] == 30) & (base_nc["method"] == mc)]["b128"].values[0] + base_nc[(base_nc["t"] == t) & (base_nc["n"] == 30) & (base_nc["method"] == mw)]["b128"].values[0]) for t in tasks]
            pm = wilcoxon_p(va_m30, vbm30)
            sm = "*" if (np.mean(va_m30) > np.mean(vbm30) and pm < 0.05) else ""
            rec["Mean30"] = f"{np.mean(vbm30)*100:.2f}{sm} (p={pm:.3f})"
            rows.append(rec)
        print(pd.DataFrame(rows).fillna("--").to_string(index=False))

        dc = agg_dn[(agg_dn["rule"] == rname) & (agg_dn["prior"] == "correct")]
        dw = agg_dn[(agg_dn["rule"] == rname) & (agg_dn["prior"] == "wrong")]
        drows = []
        for clf in ("lr", "rf"):
            for m in non_llm:
                va_c = [dc[dc["t"] == t][clf].values[0] for t in tasks]
                va_w = [dw[dw["t"] == t][clf].values[0] for t in tasks]
                va_m = [0.5*(a+b) for a, b in zip(va_c, va_w)]
                vb = [base_dn[(base_dn["t"] == t) & (base_dn["n"] == 30) & (base_dn["model"] == clf) & (base_dn["method"] == m)]["ba"].values[0] for t in tasks]
                s1 = "*" if (np.mean(va_c) > np.mean(vb) and wilcoxon_p(va_c, vb) < 0.05) else ""
                s3 = "*" if (np.mean(va_m) > np.mean(vb) and wilcoxon_p(va_m, vb) < 0.05) else ""
                drows.append({"clf": clf, "method": m, "Aln": f"{np.mean(vb)*100:.2f}{s1}", "Mis": "--", "Mean": f"{np.mean(vb)*100:.2f}{s3}"})
            for label, mc, mw in prior_m:
                va_c = [dc[dc["t"] == t][clf].values[0] for t in tasks]
                va_w = [dw[dw["t"] == t][clf].values[0] for t in tasks]
                va_m = [0.5*(a+b) for a, b in zip(va_c, va_w)]
                vbc = [base_dn[(base_dn["t"] == t) & (base_dn["n"] == 30) & (base_dn["model"] == clf) & (base_dn["method"] == mc)]["ba"].values[0] for t in tasks]
                vbw = [base_dn[(base_dn["t"] == t) & (base_dn["n"] == 30) & (base_dn["model"] == clf) & (base_dn["method"] == mw)]["ba"].values[0] for t in tasks]
                vbm = [0.5*(a+b) for a, b in zip(vbc, vbw)]
                s1 = "*" if (np.mean(va_c) > np.mean(vbc) and wilcoxon_p(va_c, vbc) < 0.05) else ""
                s2 = "*" if (np.mean(va_w) > np.mean(vbw) and wilcoxon_p(va_w, vbw) < 0.05) else ""
                s3 = "*" if (np.mean(va_m) > np.mean(vbm) and wilcoxon_p(va_m, vbm) < 0.05) else ""
                drows.append({"clf": clf, "method": label, "Aln": f"{np.mean(vbc)*100:.2f}{s1}", "Mis": f"{np.mean(vbw)*100:.2f}{s2}", "Mean": f"{np.mean(vbm)*100:.2f}{s3}"})
        print(f"--- Downstream ({rname}) ---")
        print(pd.DataFrame(drows).to_string(index=False))
