"""Test Rao-Blackwellized Repeated-Pairing Paired Delivery (RP-PD) on all 12 clinical datasets x 20 repeats.
Zero extra search budget B (same B=128 search path), zero test leakage, exact generalization of McNemar z!
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
from test_eb_delivery import repeated_lopo


def eval_rppd(job):
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

            # Shortlist: anchor + top-12 unique searched candidates by smooth single-pairing score
            top12 = sorted(eval_ids, key=lambda j: (-(acc_sc[j] + 0.15 * cont_sc[j]), -len(bk.masks[j])))[:12]
            # Also include top-4 full-K (size==CAP) candidates so downstream LR/RF never get starved
            full_ids = [j for j in eval_ids if len(bk.masks[j]) == clin.CAP]
            top4_full = sorted(full_ids, key=lambda j: -(acc_sc[j] + 0.15 * cont_sc[j]))[:4]
            shortlist = list(dict.fromkeys([a_mask] + [tuple(bk.masks[j]) for j in top12 + top4_full]))

            # R=10 repeated stratified LOPO on the SAME (x, y) training donors
            rep_preds, rep_smarg = repeated_lopo(x, y, shortlist, rs, n_reps=10)
            # rep_preds[m] has shape (R, 2n) binary correctness! Wait: check shape!
            rep_acc = {m: float(rep_preds[m].mean()) for m in shortlist}
            rep_sm = {m: float(np.tanh(rep_smarg[m] / (0.5 * (np.std(rep_smarg[m]) + 1e-6))).mean()) for m in shortlist}

            # Best empirical challenger (variant A: any size; variant B: slight size-K preference)
            non_anchor = [m for m in shortlist if m != a_mask]
            if not non_anchor:
                m_ch_any = a_mask
                m_ch_full = a_mask
            else:
                m_ch_any = max(non_anchor, key=lambda m: (rep_acc[m] + 0.03 * rep_sm[m], len(m)))
                full_na = [m for m in non_anchor if len(m) == clin.CAP] or non_anchor
                m_ch_full = max(full_na, key=lambda m: (rep_acc[m] + 0.03 * rep_sm[m]))

            # Compute Rao-Blackwellized per-instance correctness diff delta_i = mean_r c_{r,i}(m*) - mean_r c_{r,i}(m0)
            # rep_preds[m] is shape (R, 2n) or (2n,)? Let us handle both cleanly!
            c0 = rep_preds[a_mask]
            if c0.ndim == 2:
                c0_i = c0.astype(float).mean(axis=0)
            else:
                c0_i = c0.astype(float).reshape(10, -1).mean(axis=0)

            def rppd_z(m_ch):
                cm = rep_preds[m_ch]
                cm_i = cm.astype(float).mean(axis=0) if cm.ndim == 2 else cm.astype(float).reshape(10, -1).mean(axis=0)
                delta = cm_i - c0_i
                denom = float(np.sqrt(np.sum(delta * delta) + 1e-12))
                return float(np.sum(delta)) / denom

            z_any = rppd_z(m_ch_any)
            z_full = rppd_z(m_ch_full)

            # Also paired t-stat (exact one-sample paired t on delta_i!)
            def rppd_t(m_ch):
                cm = rep_preds[m_ch]
                cm_i = cm.astype(float).mean(axis=0) if cm.ndim == 2 else cm.astype(float).reshape(10, -1).mean(axis=0)
                delta = cm_i - c0_i
                sd = float(np.std(delta, ddof=1))
                if sd < 1e-12:
                    return 99.0 if float(np.mean(delta)) > 1e-12 else 0.0
                return float(np.mean(delta)) / (sd / np.sqrt(len(delta)))

            t_any = rppd_t(m_ch_any)
            t_full = rppd_t(m_ch_full)

            cands = {}
            cands["static_rank"] = a_mask
            cands["ch_any"] = m_ch_any
            cands["ch_full"] = m_ch_full
            for tau in (1.0, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50):
                cands[f"z_any_t{tau:.2f}"] = m_ch_any if z_any >= tau else a_mask
                cands[f"z_full_t{tau:.2f}"] = m_ch_full if z_full >= tau else a_mask
                cands[f"t_full_t{tau:.2f}"] = m_ch_full if t_full >= tau else a_mask

            # Dimension-adaptive Bonferroni/EVT gate: tau(p) = c * sqrt(ln(max(e, p)))
            for c_evt in (0.6, 0.8, 1.0, 1.2):
                thr = c_evt * np.sqrt(np.log(max(np.e, float(p))))
                cands[f"zevt_full_c{c_evt:.1f}"] = m_ch_full if z_full >= thr else a_mask

            for rname, m in cands.items():
                out_nc.append((t, r, n, prior_kind, rname, score_ba(m), len(m)))
            if down is not None:
                for rname in ("z_any_t1.50", "z_any_t1.75", "z_any_t2.00", "z_full_t1.50", "z_full_t1.75", "z_full_t2.00", "z_full_t2.25", "zevt_full_c0.8", "zevt_full_c1.0"):
                    res_d = down.scores(cands[rname])
                    out_dn.append((t, r, prior_kind, rname, float(res_d["lr"][0]), float(res_d["rf"][0]), len(cands[rname])))
    return out_nc, out_dn


if __name__ == "__main__":
    # Quick shape check first
    x0, y0, _ = clin.get_donors("cad", 0, 10)
    rp, _ = repeated_lopo(x0, y0, [(0, 1, 2, 3)], 42, n_reps=10)
    print("rep_preds shape:", rp[(0, 1, 2, 3)].shape)

    jobs = [(t, r) for t in clin.ALL_TASKS for r in range(clin.REPEATS)]
    with ProcessPoolExecutor(max_workers=20) as ex:
        results = list(ex.map(eval_rppd, jobs))
    nc_flat = [x for a, _ in results for x in a]
    dn_flat = [x for _, b in results for x in b]
    df = pd.DataFrame(nc_flat, columns=["t", "r", "n", "prior", "rule", "ba", "sz"])
    agg = df.groupby(["t", "n", "prior", "rule"], as_index=False)[["ba", "sz"]].mean()
    piv = agg.groupby(["prior", "rule", "n"], as_index=False)["ba"].mean().pivot(index="rule", columns=["prior", "n"], values="ba") * 100.0
    piv["Aln_avg"] = piv["correct"].mean(axis=1)
    piv["Mis_avg"] = piv["wrong"].mean(axis=1)
    piv["All_avg"] = 0.5 * (piv["Aln_avg"] + piv["Mis_avg"])
    piv["Mean_n30"] = 0.5 * (piv[("correct", 30)] + piv[("wrong", 30)])
    print("=== RP-PD NC BALANCED ACCURACY (%) ===")
    print(piv.sort_values("All_avg", ascending=False).head(20).round(2).to_string())

    dfd = pd.DataFrame(dn_flat, columns=["t", "r", "prior", "rule", "lr", "rf", "sz"])
    ad = dfd.groupby(["t", "prior", "rule"], as_index=False)[["lr", "rf", "sz"]].mean().groupby(["prior", "rule"], as_index=False)[["lr", "rf", "sz"]].mean()
    pd_lr = ad.pivot(index="rule", columns="prior", values="lr") * 100.0
    pd_rf = ad.pivot(index="rule", columns="prior", values="rf") * 100.0
    comb_d = pd.DataFrame({
        "LR_Aln": pd_lr["correct"], "LR_Mis": pd_lr["wrong"], "LR_Mean": 0.5*(pd_lr["correct"]+pd_lr["wrong"]),
        "RF_Aln": pd_rf["correct"], "RF_Mis": pd_rf["wrong"], "RF_Mean": 0.5*(pd_rf["correct"]+pd_rf["wrong"]),
    })
    print("\n=== RP-PD DOWNSTREAM LR & RF AT n=30 ===")
    print(comb_d.sort_values("LR_Mean", ascending=False).round(2).to_string())
