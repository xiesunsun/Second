"""Measure exact clinical evaluation-budget acceleration curve (B in {4, 8, 16, 32, 64, 128})
at n in {10, 30} across all 12 clinical datasets x 20 repeats for:
  - SP-PD (Aligned, R=10, lam=0.5, full-K)
  - SP-PD (Misaligned, R=10, lam=0.5, full-K)
  - SP-PD (w/o Prior, R=10, lam=0.0, full-K)
  - Also pull exact B in {4, 8, 16, 32, 64, 128} curves for GA, BPSO, EDA, MEL, piBO from clinical_non_llm_curves.tsv!
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path("/home/sunxie/paper-experiments-20261124")
sys.path.insert(0, str(ROOT))
import run_clinical_non_llm as clin
from run_sec4d_complete import repeated_lopo_per_rep, build_k4_shortlist, select_eb_k4

BUDGETS = (4, 8, 16, 32, 64, 128)


def eval_budget_job(job):
    t, r = job
    q_outer, v_outer, _ = clin.load_role(t, "outer")
    orders, p = clin.load_gemini_orders(t)
    rows = []

    for n in (10, 30):
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

        # 1) w/o Prior (order=None)
        bk_np, st_np, _, _ = clin.run_frozen_ext(x, y, clin.CAP, 128, None, rs)
        path_np = st_np["path"]
        # Collect union of shortlists across all budgets so repeated_lopo runs ONCE per (t, r, n, arm)!
        sls_np = {b: build_k4_shortlist(bk_np, path_np[:b], None, None) for b in BUDGETS}
        u_np = list(dict.fromkeys(m for sl in sls_np.values() for m in sl))
        pr_np, sm_np, _ = repeated_lopo_per_rep(x, y, u_np, rs, max_reps=10)
        for b in BUDGETS:
            m = select_eb_k4(sls_np[b], pr_np, sm_np, 10, 0.0, p, n, None)
            rows.append((t, r, n, b, "sppd_wo_prior", score_ba(m)))

        # 2) SP-PD Aligned & Misaligned
        for pk in ("correct", "wrong"):
            order = orders[pk]
            bk, st, ret, _ = clin.run_frozen_ext(x, y, clin.CAP, 128, order, rs)
            path = st["path"]
            a_mask = tuple(bk.masks[ret["anchor"]])
            topK = set(int(f) for f in order[:clin.CAP])
            top2K = set(int(f) for f in order[:2 * clin.CAP])
            def phi_fullK(m):
                return sum(1.0 if f in topK else (0.5 if f in top2K else 0.0) for f in m) / float(clin.CAP)
            sls = {b: build_k4_shortlist(bk, path[:b], a_mask, phi_fullK) for b in BUDGETS}
            u_sl = list(dict.fromkeys(m for sl in sls.values() for m in sl))
            pr, sm, _ = repeated_lopo_per_rep(x, y, u_sl, rs, max_reps=10)
            for b in BUDGETS:
                m = select_eb_k4(sls[b], pr, sm, 10, 0.50, p, n, phi_fullK)
                rows.append((t, r, n, b, f"sppd_{pk}", score_ba(m)))
    return rows


if __name__ == "__main__":
    jobs = [(t, r) for t in clin.ALL_TASKS for r in range(clin.REPEATS)]
    with ProcessPoolExecutor(max_workers=20) as ex:
        res = list(ex.map(eval_budget_job, jobs))
    df = pd.DataFrame([x for sub in res for x in sub], columns=["t", "r", "n", "budget", "method", "ba"])
    agg = df.groupby(["t", "n", "budget", "method"], as_index=False)["ba"].mean()
    agg.to_csv(ROOT / "results/clinical_sppd_budget_curves.tsv", sep="\t", index=False)

    # Also read baseline curves from clinical_non_llm_curves.tsv (all_twelve cohort!)
    cbase = pd.read_csv(ROOT / "results/clinical_non_llm_curves.tsv", sep="\t")
    print("Baseline curve cols:", list(cbase.columns))
    print("Baseline panels:", cbase["panel"].unique() if "panel" in cbase.columns else "none")
    cb_sub = cbase[(cbase["panel"] == "all_twelve") & (cbase["n"].isin([10, 30])) & (cbase["budget"].isin(BUDGETS))]
    for n in (10, 30):
        print(f"\n=== CLINICAL BUDGET ACCELERATION AT n = {n} (12 DATASETS, NC BA %) ===")
        p_sp = agg[agg["n"] == n].groupby(["method", "budget"], as_index=False)["ba"].mean().pivot(index="method", columns="budget", values="ba") * 100.0
        p_bl = cb_sub[cb_sub["n"] == n].pivot(index="method", columns="budget", values="ba") * 100.0
        keep_bl = [m for m in ("mel", "ga", "bpso", "eda", "pibo_correct", "pibo_wrong", "mrmr_static", "gemini_static") if m in p_bl.index]
        comb = pd.concat([p_sp, p_bl.loc[keep_bl]])
        print(comb.round(2).to_string())
