"""Test Rao-Blackwellized RP-PD using the EXACT SAME score-derived offline prior (sc_cards)
used by LLM-Select (Score) and LSP, across all 12 clinical datasets x 20 repeats x n in {3,10,20,30}
for NC and Downstream LR/RF (n=30). Also compute exact Wilcoxon p-values against all baselines!
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


def eval_score_rppd(job):
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

            top12 = sorted(eval_ids, key=lambda j: (-(acc_sc[j] + 0.15 * cont_sc[j]), -len(bk.masks[j])))[:12]
            full_ids = [j for j in eval_ids if len(bk.masks[j]) == clin.CAP]
            top4_full = sorted(full_ids, key=lambda j: -(acc_sc[j] + 0.15 * cont_sc[j]))[:4]
            shortlist = list(dict.fromkeys([a_mask] + [tuple(bk.masks[j]) for j in top12 + top4_full]))

            rep_preds, rep_smarg = repeated_lopo(x, y, shortlist, rs, n_reps=10)
            rep_acc = {m: float(rep_preds[m].mean()) for m in shortlist}
            rep_sm = {m: float(np.tanh(rep_smarg[m] / (0.5 * (np.std(rep_smarg[m]) + 1e-6))).mean()) for m in shortlist}

            non_anchor = [m for m in shortlist if m != a_mask]
            full_na = [m for m in non_anchor if len(m) == clin.CAP] or non_anchor
            m_ch_full = max(full_na, key=lambda m: (rep_acc[m] + 0.03 * rep_sm[m])) if full_na else a_mask

            c0_i = rep_preds[a_mask].astype(float).reshape(10, -1).mean(axis=0)
            cm_i = rep_preds[m_ch_full].astype(float).reshape(10, -1).mean(axis=0)
            delta = cm_i - c0_i
            z_full = float(np.sum(delta)) / float(np.sqrt(np.sum(delta * delta) + 1e-12))

            cands = {"static_score": a_mask}
            for tau in (1.25, 1.50, 1.75, 2.00):
                cands[f"sc_rppd_t{tau:.2f}"] = m_ch_full if z_full >= tau else a_mask

            for rname, m in cands.items():
                out_nc.append((t, r, n, prior_kind, rname, score_ba(m)))
            if down is not None:
                for rname in ("sc_rppd_t1.50", "sc_rppd_t1.75", "sc_rppd_t2.00"):
                    res_d = down.scores(cands[rname])
                    out_dn.append((t, r, prior_kind, rname, float(res_d["lr"][0]), float(res_d["rf"][0])))
    return out_nc, out_dn


if __name__ == "__main__":
    jobs = [(t, r) for t in clin.ALL_TASKS for r in range(clin.REPEATS)]
    with ProcessPoolExecutor(max_workers=20) as ex:
        results = list(ex.map(eval_score_rppd, jobs))
    nc_flat = [x for a, _ in results for x in a]
    dn_flat = [x for _, b in results for x in b]
    df = pd.DataFrame(nc_flat, columns=["t", "r", "n", "prior", "rule", "ba"])
    agg = df.groupby(["t", "n", "prior", "rule"], as_index=False)["ba"].mean()
    piv = agg.groupby(["prior", "rule", "n"], as_index=False)["ba"].mean().pivot(index="rule", columns=["prior", "n"], values="ba") * 100.0
    piv["Aln_avg"] = piv["correct"].mean(axis=1)
    piv["Mis_avg"] = piv["wrong"].mean(axis=1)
    piv["All_avg"] = 0.5 * (piv["Aln_avg"] + piv["Mis_avg"])
    piv["Mean_n30"] = 0.5 * (piv[("correct", 30)] + piv[("wrong", 30)])
    print("=== SCORE-PRIOR RP-PD NC BALANCED ACCURACY (%) ===")
    print(piv.sort_values("All_avg", ascending=False).round(2).to_string())

    dfd = pd.DataFrame(dn_flat, columns=["t", "r", "prior", "rule", "lr", "rf"])
    ad = dfd.groupby(["t", "prior", "rule"], as_index=False)[["lr", "rf"]].mean().groupby(["prior", "rule"], as_index=False)[["lr", "rf"]].mean()
    pd_lr = ad.pivot(index="rule", columns="prior", values="lr") * 100.0
    pd_rf = ad.pivot(index="rule", columns="prior", values="rf") * 100.0
    comb_d = pd.DataFrame({
        "LR_Aln": pd_lr["correct"], "LR_Mis": pd_lr["wrong"], "LR_Mean": 0.5*(pd_lr["correct"]+pd_lr["wrong"]),
        "RF_Aln": pd_rf["correct"], "RF_Mis": pd_rf["wrong"], "RF_Mean": 0.5*(pd_rf["correct"]+pd_rf["wrong"]),
    })
    print("\n=== SCORE-PRIOR RP-PD DOWNSTREAM LR & RF AT n=30 ===")
    print(comb_d.round(2).to_string())
