"""Quick verification: compare sc_eb_fullK (allowing len<=K vs enforcing full size K=4)
at lam in {0.55, 0.65, 0.75, 0.85} on NC (n in {3,10,20,30}) + Downstream LR/RF (n=30).
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
import run_clinical_llm_baselines as llm_mod
from test_eb_delivery import repeated_lopo


def eval_one(job):
    t, r = job
    q_outer, v_outer, _ = clin.load_role(t, "outer")
    sc_cards, _ = llm_mod.load_llm_cards(t)
    p = len(sc_cards["correct"])
    orders = {pk: np.lexsort((np.arange(p), -sc_cards[pk])) for pk in ("correct", "wrong")}
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

            # Full-K shortlist (size == CAP) + anchor
            full_ids = [j for j in eval_ids if len(bk.masks[j]) == clin.CAP]
            top_pure = sorted(full_ids, key=lambda j: -(acc_sc[j] + 0.15 * cont_sc[j]))[:10]
            top_tilt = sorted(full_ids, key=lambda j: -(acc_sc[j] + 0.10 * cont_sc[j] + 0.25 * phi_fullK(bk.masks[j])))[:6]
            shortlist = list(dict.fromkeys([a_mask] + [tuple(bk.masks[j]) for j in top_pure + top_tilt]))

            rep_preds, rep_smarg = repeated_lopo(x, y, shortlist, rs, n_reps=10)
            rep_acc = {m: float(rep_preds[m].mean()) for m in shortlist}
            rep_sm = {m: float(np.tanh(rep_smarg[m] / (0.5 * (np.std(rep_smarg[m]) + 1e-6))).mean()) for m in shortlist}

            dim_factor = np.sqrt(np.log(max(2.0, p / 2.0))) / np.sqrt(2.0 * n)
            cands = {}
            for lam in (0.55, 0.65, 0.75, 0.85, 1.00):
                cands[f"ebK4_l{lam:.2f}"] = max(shortlist, key=lambda m: (rep_acc[m] + 0.03 * rep_sm[m] + lam * dim_factor * phi_fullK(m), phi_fullK(m)))

            for rname, m in cands.items():
                out_nc.append((t, r, n, prior_kind, rname, score_ba(m)))
            if down is not None:
                for rname in ("ebK4_l0.55", "ebK4_l0.65", "ebK4_l0.75", "ebK4_l0.85", "ebK4_l1.00"):
                    res_d = down.scores(cands[rname])
                    out_dn.append((t, r, prior_kind, rname, float(res_d["lr"][0]), float(res_d["rf"][0])))
    return out_nc, out_dn


if __name__ == "__main__":
    jobs = [(t, r) for t in clin.ALL_TASKS for r in range(clin.REPEATS)]
    with ProcessPoolExecutor(max_workers=20) as ex:
        results = list(ex.map(eval_one, jobs))
    nc_flat = [x for a, _ in results for x in a]
    dn_flat = [x for _, b in results for x in b]
    df = pd.DataFrame(nc_flat, columns=["t", "r", "n", "prior", "rule", "ba"])
    agg = df.groupby(["t", "n", "prior", "rule"], as_index=False)["ba"].mean()
    piv = agg.groupby(["prior", "rule", "n"], as_index=False)["ba"].mean().pivot(index="rule", columns=["prior", "n"], values="ba") * 100.0
    piv["Aln_avg"] = piv["correct"].mean(axis=1)
    piv["Mis_avg"] = piv["wrong"].mean(axis=1)
    piv["All_avg"] = 0.5 * (piv["Aln_avg"] + piv["Mis_avg"])
    piv["Mean_n30"] = 0.5 * (piv[("correct", 30)] + piv[("wrong", 30)])
    print("=== FULL-K (|S|=4) SCORE-PRIOR EB DELIVERY: NC BA (%) ===")
    print(piv.sort_values("All_avg", ascending=False).round(2).to_string())

    dfd = pd.DataFrame(dn_flat, columns=["t", "r", "prior", "rule", "lr", "rf"])
    ad = dfd.groupby(["t", "prior", "rule"], as_index=False)[["lr", "rf"]].mean().groupby(["prior", "rule"], as_index=False)[["lr", "rf"]].mean()
    pd_lr = ad.pivot(index="rule", columns="prior", values="lr") * 100.0
    pd_rf = ad.pivot(index="rule", columns="prior", values="rf") * 100.0
    comb_d = pd.DataFrame({
        "LR_Aln": pd_lr["correct"], "LR_Mis": pd_lr["wrong"], "LR_Mean": 0.5*(pd_lr["correct"]+pd_lr["wrong"]),
        "RF_Aln": pd_rf["correct"], "RF_Mis": pd_rf["wrong"], "RF_Mean": 0.5*(pd_rf["correct"]+pd_rf["wrong"]),
    })
    print("\n=== FULL-K (|S|=4) SCORE-PRIOR EB DELIVERY: DOWNSTREAM LR & RF AT n=30 ===")
    print(comb_d.round(2).to_string())
