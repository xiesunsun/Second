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


def eval_eb_high(job):
    t, r = job
    q_outer, v_outer, _ = clin.load_role(t, "outer")
    orders, p = clin.load_gemini_orders(t)
    sc_cards, _ = llm_mod.load_llm_cards(t)
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
            anchor_id = ret["anchor"]
            a_mask = tuple(bk.masks[anchor_id])
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
            def phi(m):
                return sum(1.0 if f in topK else (0.5 if f in top2K else 0.0) for f in m) / max(1.0, float(len(m)))
            def phi_fullK(m):
                # Reward having full K prior features rather than 1-feature subset of prior!
                return sum(1.0 if f in topK else (0.5 if f in top2K else 0.0) for f in m) / float(clin.CAP)

            top_pure = sorted(eval_ids, key=lambda j: (-(acc_sc[j] + 0.15 * cont_sc[j]), len(bk.masks[j])))[:8]
            top_tilt = sorted(eval_ids, key=lambda j: (-(acc_sc[j] + 0.10 * cont_sc[j] + 0.25 * phi(bk.masks[j])), len(bk.masks[j])))[:6]
            shortlist = list(dict.fromkeys([a_mask] + [tuple(bk.masks[j]) for j in top_pure + top_tilt]))

            rep_preds, rep_smarg = repeated_lopo(x, y, shortlist, rs, n_reps=10)
            rep_acc = {m: float(rep_preds[m].mean()) for m in shortlist}
            rep_sm = {m: float(np.tanh(rep_smarg[m] / (0.5 * (np.std(rep_smarg[m]) + 1e-6))).mean()) for m in shortlist}

            dim_factor = np.sqrt(np.log(max(2.0, p / 2.0))) / np.sqrt(2.0 * n)
            cands = {}
            cands["cur_tau15"] = tuple(bk.masks[ret["paired"][-1]])
            cands["static_rank"] = a_mask
            for lam in (0.35, 0.50, 0.65, 0.85, 1.10):
                cands[f"eb_dim_{lam:.2f}"] = max(shortlist, key=lambda m: (rep_acc[m] + 0.03 * rep_sm[m] + lam * dim_factor * phi(m), -len(m)))
                cands[f"eb_fullK_{lam:.2f}"] = max(shortlist, key=lambda m: (rep_acc[m] + 0.03 * rep_sm[m] + lam * dim_factor * phi_fullK(m), len(m)))

            for rname, m in cands.items():
                out_nc.append((t, r, n, prior_kind, rname, score_ba(m), len(m)))
            if down is not None:
                for rname in ("cur_tau15", "eb_dim_0.35", "eb_dim_0.50", "eb_dim_0.65", "eb_fullK_0.35", "eb_fullK_0.50", "eb_fullK_0.65"):
                    res_d = down.scores(cands[rname])
                    out_dn.append((t, r, prior_kind, rname, float(res_d["lr"][0]), float(res_d["rf"][0]), len(cands[rname])))
    return out_nc, out_dn


if __name__ == "__main__":
    jobs = [(t, r) for t in clin.ALL_TASKS for r in range(clin.REPEATS)]
    with ProcessPoolExecutor(max_workers=20) as ex:
        results = list(ex.map(eval_eb_high, jobs))
    nc_flat = [x for a, _ in results for x in a]
    dn_flat = [x for _, b in results for x in b]
    df = pd.DataFrame(nc_flat, columns=["t", "r", "n", "prior", "rule", "ba", "sz"])
    agg = df.groupby(["t", "n", "prior", "rule"], as_index=False)[["ba", "sz"]].mean()
    piv = agg.groupby(["prior", "rule", "n"], as_index=False)["ba"].mean().pivot(index="rule", columns=["prior", "n"], values="ba") * 100.0
    szp = agg[agg["n"] == 30].groupby(["prior", "rule"], as_index=False)["sz"].mean().pivot(index="rule", columns="prior", values="sz")
    piv["Aln_avg"] = piv["correct"].mean(axis=1)
    piv["Mis_avg"] = piv["wrong"].mean(axis=1)
    piv["All_avg"] = 0.5 * (piv["Aln_avg"] + piv["Mis_avg"])
    piv["Mean_n30"] = 0.5 * (piv[("correct", 30)] + piv[("wrong", 30)])
    piv["sz_aln"] = szp["correct"]
    piv["sz_mis"] = szp["wrong"]
    print("=== NC BALANCED ACCURACY (%) & MEAN SUBSET SIZE AT n=30 ===")
    print(piv.sort_values("All_avg", ascending=False).round(2).to_string())

    dfd = pd.DataFrame(dn_flat, columns=["t", "r", "prior", "rule", "lr", "rf", "sz"])
    ad = dfd.groupby(["t", "prior", "rule"], as_index=False)[["lr", "rf", "sz"]].mean().groupby(["prior", "rule"], as_index=False)[["lr", "rf", "sz"]].mean()
    pd_lr = ad.pivot(index="rule", columns="prior", values="lr") * 100.0
    pd_rf = ad.pivot(index="rule", columns="prior", values="rf") * 100.0
    pd_sz = ad.pivot(index="rule", columns="prior", values="sz")
    comb_d = pd.DataFrame({
        "LR_Aln": pd_lr["correct"], "LR_Mis": pd_lr["wrong"], "LR_Mean": 0.5*(pd_lr["correct"]+pd_lr["wrong"]),
        "RF_Aln": pd_rf["correct"], "RF_Mis": pd_rf["wrong"], "RF_Mean": 0.5*(pd_rf["correct"]+pd_rf["wrong"]),
        "sz_Aln": pd_sz["correct"], "sz_Mis": pd_sz["wrong"],
    })
    print("\n=== DOWNSTREAM LR & RF AT n=30 ===")
    print(comb_d.round(2).to_string())
