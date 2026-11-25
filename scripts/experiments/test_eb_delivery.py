"""Leak-free evaluation of Empirical-Bayes Prior-Regularized Repeated-Pairing Delivery
on ALL 12 clinical datasets (20 repeats x n in {3,10,20,30}, B=128, K=4, 1-call Rank prior),
evaluating NC (all n) AND Downstream LR / RF (n=30).
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
ROOT = Path('/home/sunxie/paper-experiments-20261124')
sys.path.insert(0, str(ROOT))
import run_clinical_non_llm as clin


def repeated_lopo(x, y, masks, rs, n_reps=10):
    rng = np.random.RandomState(rs + 7777)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    n = len(idx0)
    y_sign = np.where(y == 1, 1.0, -1.0)
    preds = {m: [] for m in masks}
    smarg = {m: [] for m in masks}
    for _ in range(n_reps):
        p0 = rng.permutation(idx0)
        p1 = rng.permutation(idx1)
        m_mat = np.zeros_like(x, dtype=float)
        for f in range(n):
            te = np.array([p0[f], p1[f]])
            tr = np.ones(len(y), dtype=bool)
            tr[te] = False
            m_mat[te] = clin.margins(x[tr], y[tr], x[te])
        for m in masks:
            sv = m_mat[:, list(m)].sum(axis=1)
            preds[m].append((sv > 0) == (y == 1))
            smarg[m].append(sv * y_sign)
    return {m: np.concatenate(preds[m]) for m in masks}, {m: np.concatenate(smarg[m]) for m in masks}


def eval_job(job):
    t, r = job
    q_outer, v_outer, _ = clin.load_role(t, 'outer')
    orders, p = clin.load_gemini_orders(t)
    out_nc = []
    out_dn = []

    for n in clin.NS:
        x, y, ids = clin.get_donors(t, r, n)
        rs = clin.seed('clinical-EA-20261123', t, r, n)
        d_outer = clin.margins(x, y, q_outer)
        d_outer_t = np.ascontiguousarray(d_outer.T)
        outer_cache = {}
        def score_ba(mask):
            mask = tuple(mask)
            if mask not in outer_cache:
                pred = d_outer_t[list(mask)].sum(0) > 0
                outer_cache[mask] = float(clin.measures(pred, v_outer)[0])
            return outer_cache[mask]

        down = clin.Downstream(x, y, q_outer, v_outer, clin.seed('clinical-readout-20261123', t, r, n)) if n == 30 else None

        for prior_kind in ('correct', 'wrong'):
            order = orders[prior_kind]
            bk, st, ret, _ = clin.run_frozen_ext(x, y, clin.CAP, 128, order, rs)
            path = st['path']
            anchor_id = ret['anchor']
            a_mask = tuple(bk.masks[anchor_id])
            eval_ids = list(dict.fromkeys(path))

            # Build shortlist: anchor + top-8 data-driven finalists + top-8 prior-overlapping finalists (all inside B=128!)
            y_sign = np.where(y == 1, 1.0, -1.0)
            col_m = bk.margin * y_sign[:, None]
            acc_sc = {j: bk.scores[j]['fold'] for j in eval_ids}
            cont_sc = {}
            for j in eval_ids:
                sv = col_m[:, list(bk.masks[j])].sum(axis=1)
                cont_sc[j] = float(np.tanh(sv / (0.5 * (np.std(sv) + 1e-6))).mean())

            # Prior affinity phi(m) in [0, 1]: exact fraction of features in anchor (top-K) + 0.5 weight for rank K+1..2K
            topK = set(int(f) for f in order[:clin.CAP])
            top2K = set(int(f) for f in order[:2 * clin.CAP])
            def phi_overlap(m):
                if not m:
                    return 0.0
                return sum(1.0 if f in topK else (0.5 if f in top2K else 0.0) for f in m) / float(len(m))

            # Shortlist: union of top-8 pure-score + top-6 prior-tilted + anchor (typically 10-14 masks, all already evaluated in B=128)
            top_pure = sorted(eval_ids, key=lambda j: (-(acc_sc[j] + 0.15 * cont_sc[j]), len(bk.masks[j])))[:8]
            top_tilt = sorted(eval_ids, key=lambda j: (-(acc_sc[j] + 0.10 * cont_sc[j] + 0.15 * phi_overlap(bk.masks[j])), len(bk.masks[j])))[:6]
            champ_01 = min(eval_ids, key=lambda j: (-acc_sc[j], len(bk.masks[j])))
            shortlist = list(dict.fromkeys([a_mask, tuple(bk.masks[champ_01])] + [tuple(bk.masks[j]) for j in top_pure + top_tilt]))

            rep_preds, rep_smarg = repeated_lopo(x, y, shortlist, rs, n_reps=10)
            rep_acc = {m: float(rep_preds[m].mean()) for m in shortlist}
            rep_sm = {m: float(np.tanh(rep_smarg[m] / (0.5 * (np.std(rep_smarg[m]) + 1e-6))).mean()) for m in shortlist}

            # Evaluate Empirical-Bayes prior-regularized delivery J(m) = rep_acc(m) + 0.03*rep_sm(m) + (lam / sqrt(2n)) * phi(m)
            cands = {}
            cands['anchor'] = a_mask
            # Current baseline tau=1.5
            a_oof = (bk.oof[anchor_id] == y); c_oof = (bk.oof[champ_01] == y)
            a0 = int(np.sum(c_oof & ~a_oof)); b0 = int(np.sum(~c_oof & a_oof))
            cands['cur_tau15'] = tuple(bk.masks[champ_01]) if (a0 > b0 and (a0 - b0) ** 2 > 2.25 * (a0 + b0)) else a_mask

            inv_sqn = 1.0 / np.sqrt(2.0 * n)
            # Also test dimension-aware shrinkage: lam * sqrt(log(p/2)) / sqrt(2n)
            dim_factor = np.sqrt(np.log(max(2.0, p / 2.0))) / np.sqrt(2.0 * n)

            for lam in (0.10, 0.18, 0.25, 0.35, 0.45):
                cands[f'eb_sqn_l{lam}'] = max(shortlist, key=lambda m: (rep_acc[m] + 0.03 * rep_sm[m] + lam * inv_sqn * phi_overlap(m), -len(m)))
            for lam in (0.08, 0.12, 0.16, 0.22, 0.28):
                cands[f'eb_dim_l{lam}'] = max(shortlist, key=lambda m: (rep_acc[m] + 0.03 * rep_sm[m] + lam * dim_factor * phi_overlap(m), -len(m)))

            # Hybrid: EB shortlist winner + paired McNemar gate (tau=0.8, 1.0) against anchor
            for lam in (0.12, 0.16):
                cand_m = cands[f'eb_dim_l{lam}']
                pa = rep_preds[a_mask]; pc = rep_preds[cand_m]
                ab = float(np.sum(pc & ~pa)) / 10.0
                bb = float(np.sum(~pc & pa)) / 10.0
                for tau in (0.6, 0.8, 1.0):
                    ok = (ab > bb) and ((ab - bb) ** 2 > (tau ** 2) * (ab + bb))
                    cands[f'hyb_l{lam}_t{tau}'] = cand_m if ok else a_mask

            for rname, m in cands.items():
                out_nc.append((t, r, n, prior_kind, rname, score_ba(m)))

            if down is not None:
                # Score selected key candidates on downstream LR & RF at n=30
                key_dn = ['anchor', 'cur_tau15', 'eb_sqn_l0.18', 'eb_sqn_l0.25', 'eb_dim_l0.12', 'eb_dim_l0.16', 'eb_dim_l0.22', 'hyb_l0.12_t0.8']
                for rname in key_dn:
                    m = cands[rname]
                    res_d = down.scores(m)
                    out_dn.append((t, r, prior_kind, rname, float(res_d['lr'][0]), float(res_d['rf'][0])))
    return out_nc, out_dn


if __name__ == '__main__':
    jobs = [(t, r) for t in clin.ALL_TASKS for r in range(clin.REPEATS)]
    with ProcessPoolExecutor(max_workers=20) as ex:
        results = list(ex.map(eval_job, jobs))
    nc_flat = [x for a, _ in results for x in a]
    dn_flat = [x for _, b in results for x in b]

    df = pd.DataFrame(nc_flat, columns=['t', 'r', 'n', 'prior', 'rule', 'ba'])
    grp = df.groupby(['t', 'n', 'prior', 'rule'], as_index=False)['ba'].mean()
    agg = grp.groupby(['prior', 'rule', 'n'], as_index=False)['ba'].mean()
    piv = agg.pivot(index='rule', columns=['prior', 'n'], values='ba') * 100.0
    piv['Aln_avg'] = piv['correct'].mean(axis=1)
    piv['Mis_avg'] = piv['wrong'].mean(axis=1)
    piv['All_avg'] = 0.5 * (piv['Aln_avg'] + piv['Mis_avg'])
    piv['Mean_n30'] = 0.5 * (piv[('correct', 30)] + piv[('wrong', 30)])
    print('=== NC BA (%) ACROSS ALL 12 CLINICAL DATASETS ===')
    print(piv.sort_values('All_avg', ascending=False).round(2).to_string())

    dfd = pd.DataFrame(dn_flat, columns=['t', 'r', 'prior', 'rule', 'lr', 'rf'])
    gd = dfd.groupby(['t', 'prior', 'rule'], as_index=False)[['lr', 'rf']].mean()
    ad = gd.groupby(['prior', 'rule'], as_index=False)[['lr', 'rf']].mean()
    pd_lr = ad.pivot(index='rule', columns='prior', values='lr') * 100.0
    pd_rf = ad.pivot(index='rule', columns='prior', values='rf') * 100.0
    comb_d = pd.DataFrame({
        'LR_Aln': pd_lr['correct'], 'LR_Mis': pd_lr['wrong'], 'LR_Mean': 0.5*(pd_lr['correct']+pd_lr['wrong']),
        'RF_Aln': pd_rf['correct'], 'RF_Mis': pd_rf['wrong'], 'RF_Mean': 0.5*(pd_rf['correct']+pd_rf['wrong']),
    })
    print('\n=== DOWNSTREAM LR & RF BA (%) AT n=30 (12 DATASETS) ===')
    print(comb_d.sort_values('LR_Mean', ascending=False).round(2).to_string())
