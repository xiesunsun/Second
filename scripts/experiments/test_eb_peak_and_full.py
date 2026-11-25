"""1) Sweep higher EB prior-regularization lambda in {0.4, 0.6, 0.8, 1.0, 1.4} on n in {3,10,20,30} (NC + LR/RF).
2) Evaluate larger-sample regime (full available train split, capped at 500/class on brf/thyroid for fast 10-fold CV)
   across ALL 12 clinical datasets for SP-PD vs LLM-Select (Rank/Score), LLM-Lasso, LSP, MEL, L1-LR, mRMR, RF-Imp!
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
import run_clinical_llm_baselines as llm_mod
from test_eb_delivery import repeated_lopo


def eval_eb_high(job):
    t, r = job
    q_outer, v_outer, _ = clin.load_role(t, 'outer')
    orders, p = clin.load_gemini_orders(t)
    out_nc, out_dn = [], []

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

            y_sign = np.where(y == 1, 1.0, -1.0)
            col_m = bk.margin * y_sign[:, None]
            acc_sc = {j: bk.scores[j]['fold'] for j in eval_ids}
            cont_sc = {}
            for j in eval_ids:
                sv = col_m[:, list(bk.masks[j])].sum(axis=1)
                cont_sc[j] = float(np.tanh(sv / (0.5 * (np.std(sv) + 1e-6))).mean())

            topK = set(int(f) for f in order[:clin.CAP])
            top2K = set(int(f) for f in order[:2 * clin.CAP])
            def phi(m):
                return sum(1.0 if f in topK else (0.5 if f in top2K else 0.0) for f in m) / max(1.0, float(len(m)))

            top_pure = sorted(eval_ids, key=lambda j: (-(acc_sc[j] + 0.15 * cont_sc[j]), len(bk.masks[j])))[:8]
            top_tilt = sorted(eval_ids, key=lambda j: (-(acc_sc[j] + 0.10 * cont_sc[j] + 0.25 * phi(bk.masks[j])), len(bk.masks[j])))[:6]
            shortlist = list(dict.fromkeys([a_mask] + [tuple(bk.masks[j]) for j in top_pure + top_tilt]))

            rep_preds, rep_smarg = repeated_lopo(x, y, shortlist, rs, n_reps=10)
            rep_acc = {m: float(rep_preds[m].mean()) for m in shortlist}
            rep_sm = {m: float(np.tanh(rep_smarg[m] / (0.5 * (np.std(rep_smarg[m]) + 1e-6))).mean()) for m in shortlist}

            dim_factor = np.sqrt(np.log(max(2.0, p / 2.0))) / np.sqrt(2.0 * n)
            inv_sqn = 1.0 / np.sqrt(2.0 * n)
            cands = {}
            for lam in (0.35, 0.50, 0.65, 0.85, 1.10):
                cands[f'eb_dim_l{lam}'] = max(shortlist, key=lambda m: (rep_acc[m] + 0.03 * rep_sm[m] + lam * dim_factor * phi(m), -len(m)))
            for lam in (0.60, 0.90, 1.20, 1.50):
                cands[f'eb_sqn_l{lam}'] = max(shortlist, key=lambda m: (rep_acc[m] + 0.03 * rep_sm[m] + lam * inv_sqn * phi(m), -len(m)))

            for rname, m in cands.items():
                out_nc.append((t, r, n, prior_kind, rname, score_ba(m)))
            if down is not None:
                for rname in ('eb_dim_l0.35', 'eb_dim_l0.50', 'eb_dim_l0.65', 'eb_sqn_l0.60', 'eb_sqn_l0.90'):
                    res_d = down.scores(cands[rname])
                    out_dn.append((t, r, prior_kind, rname, float(res_d['lr'][0]), float(res_d['rf'][0])))
    return out_nc, out_dn


def eval_full_one(t):
    """Evaluate ALL 12 clinical datasets using full balanced/stratified train split (up to 300/class)
    on outer holdout for NC, LR, RF."""
    x_tr_full, y_tr_full, _ = clin.load_role(t, 'train')
    q_outer, v_outer, _ = clin.load_role(t, 'outer')
    orders, p = clin.load_gemini_orders(t)
    # Load LLM-Select (Score), LLM-Lasso & LSP cards
    sc_cards, pen_cards = llm_mod.load_llm_cards(t)
    score_orders = {pk: np.lexsort((np.arange(p), -sc_cards[pk])) for pk in ('correct', 'wrong')}

    rng = np.random.RandomState(clin.seed('clinical-full-20261124', t))
    # Use all training instances (subsampled to at most 400 per class on huge brf/thyroid so 10-fold CV stays fast)
    idx0 = np.flatnonzero(y_tr_full == 0)
    idx1 = np.flatnonzero(y_tr_full == 1)
    n0 = min(len(idx0), 400); n1 = min(len(idx1), 400)
    sel = np.sort(np.concatenate([rng.choice(idx0, n0, replace=False), rng.choice(idx1, n1, replace=False)]))
    x, y = x_tr_full[sel], y_tr_full[sel]
    rs = clin.seed('clinical-full-eval', t)

    # 5-fold stratified inner CV margins on (x, y) for fast exact NC search
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
    oof_m = np.zeros_like(x, dtype=float)
    fold_ids = np.zeros(len(y), dtype=int)
    for fi, (tr, va) in enumerate(skf.split(x, y)):
        oof_m[va] = clin.margins(x[tr], y[tr], x[va])
        fold_ids[va] = fi

    class FullBook:
        def __init__(self, x_in, y_in, m_in, fids=fold_ids):
            self.x = clin.empirical(x_in, x_in)
            self.y = y_in
            self.classes = np.array([0, 1])
            self.assignment = fids
            self.margin = m_in
            self.masks, self.scores, self.oof = [], [], []
            self.ids = {}
        def buy(self, mask):
            mask = tuple(sorted(map(int, mask)))
            if mask not in self.ids:
                pred = (self.margin[:, mask].sum(1) > 0).astype(np.int32)
                ba = 0.5 * (float(np.mean(pred[self.y == 0] == 0)) + float(np.mean(pred[self.y == 1] == 1)))
                i = len(self.masks); self.ids[mask] = i
                self.masks.append(mask); self.oof.append(pred); self.scores.append({'fold': ba})
            return self.ids[mask]

    d_outer = clin.margins(x, y, q_outer)
    d_outer_t = np.ascontiguousarray(d_outer.T)
    down = clin.Downstream(x, y, q_outer, v_outer, rs)
    def score_all(mask):
        mask = tuple(sorted(map(int, mask)))
        pred = d_outer_t[list(mask)].sum(0) > 0
        nc_ba = float(clin.measures(pred, v_outer)[0])
        ds = down.scores(mask)
        return nc_ba, float(ds['lr'][0]), float(ds['rf'][0])

    x_std = clin.prep_std(x)
    res = []
    # Static non-LLM
    for nm, m in [
        ('mRMR', clin.run_mrmr_static(x_std, y, clin.CAP)),
        ('L1-LR', clin.run_l1_lr_static(x_std, y, clin.CAP, rs)),
        ('RF-Imp', clin.run_rf_imp_static(x_std, y, clin.CAP, rs)),
    ]:
        nc, lr, rf = score_all(m)
        res.append((t, 'both', nm, nc, lr, rf))

    # MEL (B=128) directly on FullBook
    from types import SimpleNamespace
    bk_mel = FullBook(x, y, oof_m)
    rks = clin.tree_rank(bk_mel.x, y, rs), clin.cdf_rank(bk_mel.x, y)
    s_ref = clin.Search(bk_mel.x, y, rs, 128, limit=clin.CAP, book=FullBook(x, y, oof_m), ranks=rks)
    mel_prob = SimpleNamespace(x=bk_mel.x, y=y, seed=rs, book=bk_mel, base=SimpleNamespace(cap=clin.CAP, tree=rks[0], vote=s_ref.menu))
    mel_res = clin.peers.mel_path(mel_prob, 128)
    win_mel = clin.winners_from_path(bk_mel, mel_res['path'][:128])
    nc, lr, rf = score_all(bk_mel.masks[win_mel[-1]])
    res.append((t, 'both', 'MEL', nc, lr, rf))

    # Prior-informed methods
    for pk in ('correct', 'wrong'):
        order = orders[pk]
        a_mask = tuple(sorted(map(int, order[:clin.CAP])))
        sc_mask = tuple(sorted(map(int, score_orders[pk][:clin.CAP])))
        w_grid_lasso = [(a, np.power(pen_cards[pk], a)) for a in (0.5, 1.0, 2.0)]
        m_lasso, _ = llm_mod.run_weighted_l1_native(x_std, y, clin.CAP, w_grid_lasso, rs)
        rel = np.clip(sc_cards[pk], 0.05, 1.0)
        w_grid_lsp = [(eta, np.power(rel, -eta)) for eta in (0.0, 0.5, 1.0, 2.0, 4.0)]
        m_lsp, _ = llm_mod.run_weighted_l1_native(x_std, y, clin.CAP, w_grid_lsp, rs)
        for nm, m in [('LLM-Select (Rank)', a_mask), ('LLM-Select (Score)', sc_mask), ('LLM-Lasso', m_lasso), ('LSP', m_lsp)]:
            nc, lr, rf = score_all(m)
            res.append((t, pk, nm, nc, lr, rf))
        # SP-PD (B=128) on FullBook with EB delivery (lam=0.50) & tau=1.5
        bk = FullBook(x, y, oof_m)
        s = clin.Search(bk.x, y, rs, 128, limit=clin.CAP, book=bk, ranks=rks)
        base, fam = clin.schedule(s)
        menu, fam, _ = clin.third_menu(base, fam, order, clin.CAP)
        st = clin.evolve(s, 'handoff_mut', menu, fam, clin.Recombination(s))
        ret = clin.delivery_any(bk, st, a_mask, tau=1.5)
        nc, lr, rf = score_all(bk.masks[ret['paired'][-1]])
        res.append((t, pk, 'SP-PD (tau1.5)', nc, lr, rf))
        # EB prior-regularized winner over all evaluated
        topK = set(int(f) for f in order[:clin.CAP])
        top2K = set(int(f) for f in order[:2 * clin.CAP])
        def phi(m):
            return sum(1.0 if f in topK else (0.5 if f in top2K else 0.0) for f in m) / max(1.0, float(len(m)))
        dim_f = np.sqrt(np.log(max(2.0, p / 2.0))) / np.sqrt(float(len(y)))
        eval_ids = list(dict.fromkeys(st['path']))
        best_eb = max(eval_ids, key=lambda j: (bk.scores[j]['fold'] + 0.50 * dim_f * phi(bk.masks[j]), -len(bk.masks[j])))
        nc, lr, rf = score_all(bk.masks[best_eb])
        res.append((t, pk, 'SP-PD (EB0.5)', nc, lr, rf))
    return res


if __name__ == '__main__':
    # First run full-train (super fast: 12 tasks!)
    print('=== RUNNING FULL-TRAIN CLINICAL BENCHMARK (ALL 12 DATASETS, B=128, K=4) ===')
    with ProcessPoolExecutor(max_workers=12) as ex:
        full_nested = list(ex.map(eval_full_one, clin.ALL_TASKS))
    fdf = pd.DataFrame([x for sub in full_nested for x in sub], columns=['t', 'prior', 'method', 'nc', 'lr', 'rf'])
    for pk in ('correct', 'wrong'):
        sub = fdf[fdf['prior'].isin(['both', pk])].groupby('method')[['nc', 'lr', 'rf']].mean() * 100.0
        print(f'\n[FULL-TRAIN {pk.upper()} (12 datasets)]')
        print(sub.sort_values('nc', ascending=False).round(2).to_string())

    # Second run high-lambda EB sweep on n in {3,10,20,30}
    print('\n=== RUNNING HIGH-LAMBDA EB SWEEP ON n in {3,10,20,30} ===')
    jobs = [(t, r) for t in clin.ALL_TASKS for r in range(clin.REPEATS)]
    with ProcessPoolExecutor(max_workers=20) as ex:
        results = list(ex.map(eval_eb_high, jobs))
    nc_flat = [x for a, _ in results for x in a]
    dn_flat = [x for _, b in results for x in b]
    df = pd.DataFrame(nc_flat, columns=['t', 'r', 'n', 'prior', 'rule', 'ba'])
    piv = df.groupby(['t', 'n', 'prior', 'rule'], as_index=False)['ba'].mean().groupby(['prior', 'rule', 'n'], as_index=False)['ba'].mean().pivot(index='rule', columns=['prior', 'n'], values='ba') * 100.0
    piv['Aln_avg'] = piv['correct'].mean(axis=1)
    piv['Mis_avg'] = piv['wrong'].mean(axis=1)
    piv['All_avg'] = 0.5 * (piv['Aln_avg'] + piv['Mis_avg'])
    piv['Mean_n30'] = 0.5 * (piv[('correct', 30)] + piv[('wrong', 30)])
    print(piv.sort_values('All_avg', ascending=False).round(2).to_string())

    dfd = pd.DataFrame(dn_flat, columns=['t', 'r', 'prior', 'rule', 'lr', 'rf'])
    ad = dfd.groupby(['t', 'prior', 'rule'], as_index=False)[['lr', 'rf']].mean().groupby(['prior', 'rule'], as_index=False)[['lr', 'rf']].mean()
    pd_lr = ad.pivot(index='rule', columns='prior', values='lr') * 100.0
    pd_rf = ad.pivot(index='rule', columns='prior', values='rf') * 100.0
    comb_d = pd.DataFrame({
        'LR_Aln': pd_lr['correct'], 'LR_Mis': pd_lr['wrong'], 'LR_Mean': 0.5*(pd_lr['correct']+pd_lr['wrong']),
        'RF_Aln': pd_rf['correct'], 'RF_Mis': pd_rf['wrong'], 'RF_Mean': 0.5*(pd_rf['correct']+pd_rf['wrong']),
    })
    print('\n=== DOWNSTREAM LR & RF AT n=30 ===')
    print(comb_d.round(2).to_string())
