from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
ROOT = Path('/home/sunxie/paper-experiments-20261124')
sys.path.insert(0, str(ROOT))
import run_clinical_non_llm as clin


def repeated_lopo_margins(x, y, masks, rs, n_reps=10):
    """Compute out-of-fold predictions & signed margins over n_reps fresh stratified pairings
    strictly on training set (x, y) for a small set of masks (e.g. anchor + top-K finalists)."""
    rng = np.random.RandomState(rs + 7777)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    n = len(idx0)
    all_preds = {m: [] for m in masks}
    all_smarg = {m: [] for m in masks}
    y_sign = np.where(y == 1, 1.0, -1.0)
    for _ in range(n_reps):
        p0 = rng.permutation(idx0)
        p1 = rng.permutation(idx1)
        m_mat = np.zeros_like(x, dtype=float)
        for f in range(n):
            te = np.array([p0[f], p1[f]])
            tr_mask = np.ones(len(y), dtype=bool)
            tr_mask[te] = False
            m_mat[te] = clin.margins(x[tr_mask], y[tr_mask], x[te])
        for m in masks:
            sv = m_mat[:, list(m)].sum(axis=1)
            all_preds[m].append((sv > 0) == (y == 1))
            all_smarg[m].append(sv * y_sign)
    return {m: np.concatenate(all_preds[m]) for m in masks}, {m: np.concatenate(all_smarg[m]) for m in masks}


def eval_job(job):
    t, r = job
    q_outer, v_outer, _ = clin.load_role(t, 'outer')
    orders, p = clin.load_gemini_orders(t)
    out = []

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

        for prior_kind in ('correct', 'wrong'):
            order = orders[prior_kind]
            bk, st, ret, _ = clin.run_frozen_ext(x, y, clin.CAP, 128, order, rs)
            path = st['path']
            anchor_id = ret['anchor']
            a_mask = tuple(bk.masks[anchor_id])
            eval_ids = list(dict.fromkeys(path))

            # Current champion (0/1 argmax)
            champ_01 = min(eval_ids, key=lambda j: (-bk.scores[j]['fold'], len(bk.masks[j])))
            c_mask_01 = tuple(bk.masks[champ_01])

            # Continuous signed margin on search SmallBook
            y_sign = np.where(y == 1, 1.0, -1.0)
            col_m = bk.margin * y_sign[:, None]
            acc_sc = {j: bk.scores[j]['fold'] for j in eval_ids}
            cont_sc = {}
            for j in eval_ids:
                sv = col_m[:, list(bk.masks[j])].sum(axis=1)
                sc = np.std(sv) + 1e-6
                cont_sc[j] = float(np.tanh(sv / (0.5 * sc)).mean())

            # Top-8 finalists by smooth score (all inside B=128)
            top8_ids = sorted(eval_ids, key=lambda j: (-(acc_sc[j] + 0.15 * cont_sc[j]), len(bk.masks[j])))[:8]
            top8_masks = list(dict.fromkeys([a_mask, c_mask_01] + [tuple(bk.masks[j]) for j in top8_ids]))

            # Repeated fresh LOPO verification STRICTLY on training set (x, y)
            rep_preds, rep_smarg = repeated_lopo_margins(x, y, top8_masks, rs, n_reps=10)
            rep_acc = {m: float(rep_preds[m].mean()) for m in top8_masks}
            # Smooth repeated margin score
            rep_sm = {m: float(np.tanh(rep_smarg[m] / (0.5 * (np.std(rep_smarg[m]) + 1e-6))).mean()) for m in top8_masks}
            # Pick best finalist by repeated LOPO accuracy (+ small smooth tiebreak)
            best_rep = max(top8_masks, key=lambda m: (rep_acc[m] + 0.05 * rep_sm[m], -len(m)))
            best_rep_no_anc = max([m for m in top8_masks if m != a_mask] or [a_mask],
                                  key=lambda m: (rep_acc[m] + 0.05 * rep_sm[m], -len(m)))

            # Baseline McNemar stats on search pairing
            a_oof = (bk.oof[anchor_id] == y)
            c_oof = (bk.oof[champ_01] == y)
            a0 = int(np.sum(c_oof & ~a_oof))
            b0 = int(np.sum(~c_oof & a_oof))

            cands = {}
            cands['anchor'] = a_mask
            cands['cur_tau15'] = c_mask_01 if (a0 > b0 and (a0 - b0) ** 2 > 2.25 * (a0 + b0)) else a_mask
            cands['raw_01'] = c_mask_01
            cands['rep_best_raw'] = best_rep

            # 1) Dimension-aware McNemar on single pairing
            for c_pen in (0.8, 1.2, 1.6):
                thr2 = 1.0 + c_pen * np.log(max(2.0, p / 4.0))
                ok = (a0 > b0) and ((a0 - b0) ** 2 > thr2 * (a0 + b0))
                cands[f'dim_01_c{c_pen}'] = c_mask_01 if ok else a_mask

            # 2) Repeated-pairing paired McNemar gate (effective sample size = 2n, averaged over 10 pairings!)
            # Average discordant counts per pairing: a_bar = a_tot / 10, b_bar = b_tot / 10
            for tag, cm in [('01', c_mask_01), ('top8', best_rep_no_anc)]:
                pa = rep_preds[a_mask]
                pc = rep_preds[cm]
                a_bar = float(np.sum(pc & ~pa)) / 10.0
                b_bar = float(np.sum(~pc & pa)) / 10.0
                for tau in (0.75, 1.0, 1.25, 1.5):
                    ok = (a_bar > b_bar) and ((a_bar - b_bar) ** 2 > (tau ** 2) * (a_bar + b_bar))
                    cands[f'rep_mc_{tag}_t{tau}'] = cm if ok else a_mask
                # Dimension-aware repeated-pairing gate!
                for c_pen in (0.4, 0.8, 1.2):
                    thr2 = 0.75 + c_pen * np.log(max(2.0, p / 4.0))
                    ok = (a_bar > b_bar) and ((a_bar - b_bar) ** 2 > thr2 * (a_bar + b_bar))
                    cands[f'rep_dim_{tag}_c{c_pen}'] = cm if ok else a_mask

            # 3) Repeated-pairing paired margin t-test across the 2n training instances (averaged over 10 pairings per instance!)
            for tag, cm in [('01', c_mask_01), ('top8', best_rep_no_anc)]:
                # reshape (10, 2n) -> mean over 10 pairings per instance -> 2n-vector!
                ma = rep_smarg[a_mask].reshape(10, -1).mean(axis=0)
                mc = rep_smarg[cm].reshape(10, -1).mean(axis=0)
                # normalize scales
                d_vec = (mc / (np.sqrt(np.mean(mc**2)) + 1e-9)) - (ma / (np.sqrt(np.mean(ma**2)) + 1e-9))
                t_val = float(d_vec.mean() / (d_vec.std(ddof=1) / np.sqrt(len(d_vec)) + 1e-9))
                for t_thr in (1.5, 2.0, 2.5):
                    ok = (t_val > t_thr) and (rep_acc[cm] > rep_acc[a_mask])
                    cands[f'rep_t_{tag}_th{t_thr}'] = cm if ok else a_mask

            for rname, m in cands.items():
                out.append((t, r, n, prior_kind, rname, score_ba(m)))
    return out


if __name__ == '__main__':
    jobs = [(t, r) for t in clin.ALL_TASKS for r in range(clin.REPEATS)]
    with ProcessPoolExecutor(max_workers=20) as ex:
        nested = list(ex.map(eval_job, jobs))
    flat = [row for sub in nested for row in sub]
    df = pd.DataFrame(flat, columns=['t', 'r', 'n', 'prior', 'rule', 'ba'])
    # avg over repeats per task, then over 12 tasks
    grp = df.groupby(['t', 'n', 'prior', 'rule'], as_index=False)['ba'].mean()
    agg = grp.groupby(['prior', 'rule', 'n'], as_index=False)['ba'].mean()
    piv = agg.pivot(index='rule', columns=['prior', 'n'], values='ba') * 100.0
    piv['Mean_all'] = piv.mean(axis=1)
    piv['Aln_mean'] = piv['correct'].mean(axis=1)
    piv['Mis_mean'] = piv['wrong'].mean(axis=1)
    piv['Mean_n30'] = 0.5 * (piv[('correct', 30)] + piv[('wrong', 30)])
    piv = piv.sort_values('Mean_all', ascending=False)
    print(piv.round(2).to_string())
