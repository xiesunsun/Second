"""Additive sync for high-dimensional benchmarks:
1) Wide-K (42 folds): adds native multi-class ReliefF (Z-score, Manhattan, k=min(5, n_min-1)).
2) Tight-K (72 folds across author_13 + prospective_10): adds EDA (B=1..1000), RF-Imp, ReliefF,
   and strictly native 3-fold stratified CV L1-LR (zero wrapper oracle access).
Writes additive TSVs without modifying any frozen directory or existing raw files.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

ROOT = Path('/home/sunxie/paper-experiments-20261124')
PAIRED_ROOT = Path('/home/sunxie/paired-delivery-20261123.lCTbZvTQ')
sys.path.insert(0, str(PAIRED_ROOT))
sys.path.insert(0, str(ROOT))

from engine import projected, read
from local_study import runner
from exact_five import kernel
import run_numerical_wide_baselines as wide_mod
import run_numerical_baselines as tight_mod


def run_relieff_multiclass(x, y, cap):
    t0 = perf_counter()
    xs = StandardScaler().fit_transform(x)
    n, p = xs.shape
    classes, counts = np.unique(y, return_counts=True)
    priors = {c: cnt / float(n) for c, cnt in zip(classes, counts)}
    n_min = int(counts.min())
    k = max(1, min(5, n_min - 1))
    dmat = cdist(xs, xs, metric='cityblock')
    np.fill_diagonal(dmat, np.inf)
    weights = np.zeros(p, dtype=float)
    cls_idx = {c: np.flatnonzero(y == c) for c in classes}
    for i in range(n):
        yi = y[i]
        same = cls_idx[yi]
        if len(same) > 1:
            ds = dmat[i, same]
            kh = min(k, len(same) - 1)
            nh = same[np.argpartition(ds, kh)[:kh]]
            weights -= np.abs(xs[i] - xs[nh]).mean(axis=0)
        denom = max(1e-12, 1.0 - priors[yi])
        for c in classes:
            if c == yi:
                continue
            other = cls_idx[c]
            do = dmat[i, other]
            km = min(k, len(other))
            nm = other[np.argpartition(do, km - 1)[:km]] if km < len(other) else other
            weights += (priors[c] / denom) * np.abs(xs[i] - xs[nm]).mean(axis=0)
    order = np.lexsort((np.arange(p), -weights))
    mask = tuple(sorted(map(int, order[:cap])))
    return mask, perf_counter() - t0


def group_of(ds):
    return 'lymphoma-family' if ds in ('DLBCL', 'lymphoma') else ds


def eval_wide_relieff(key):
    kernel()
    folder = PAIRED_ROOT / 'width_wide'
    a = read(folder / 'train' / f'{key}.json.gz')
    meta = read(ROOT / 'raw_numerical_wide' / f'{key}.json.gz')
    c = a['case']
    cap = int(a['cap'])
    p = int(a['p'])
    x, y = projected.arrays(c)
    q, v = projected.arrays(c, True)
    mask, dt = run_relieff_multiclass(x, y, cap)
    om = runner.outer_metrics(x, y, q, v, [np.asarray(mask, dtype=int)])
    return dict(
        key=key,
        group=meta['group'],
        dataset=meta['task'],
        p=p,
        cap=cap,
        precision=float(om['precision'][0]),
        ba=float(om['ba'][0]),
        f1=float(om['f1'][0]),
        width=float(len(mask)),
        cpu=float(dt),
    )


def eval_tight_extra(job):
    suite, source_group, task_key, rep, x, y, q, v, rs, cap = job
    kernel()
    bk_eda, pth_eda, win_eda, dt_eda = wide_mod.run_eda_wide(x, y, rs, cap, 1000)
    m_rf, dt_rf = wide_mod.run_rf_wide(x, y, rs, cap)
    m_rel, dt_rel = run_relieff_multiclass(x, y, cap)
    m_l1, dt_l1 = wide_mod.run_l1_wide(x, y, rs, cap)

    eda_masks = list(dict.fromkeys(tuple(bk_eda.masks[j]) for j in set(win_eda)))
    all_masks = list(dict.fromkeys(eda_masks + [m_rf, m_rel, m_l1]))
    om = runner.outer_metrics(x, y, q, v, [np.asarray(m, dtype=int) for m in all_masks])
    m2o = {m: dict(precision=float(om['precision'][i]), ba=float(om['ba'][i]), f1=float(om['f1'][i])) for i, m in enumerate(all_masks)}

    out = []
    eda_curve = [m2o[tuple(bk_eda.masks[j])] for j in win_eda]
    out.append(dict(
        suite=suite, group=source_group, task_key=task_key, rep=rep, method='eda',
        b128_prec=eda_curve[127]['precision'], b500_prec=eda_curve[499]['precision'],
        b1000_prec=eda_curve[999]['precision'], b1000_ba=eda_curve[999]['ba'],
        b1000_f1=eda_curve[999]['f1'],
        curve_prec=[d['precision'] for d in eda_curve],
        curve_ba=[d['ba'] for d in eda_curve],
        curve_f1=[d['f1'] for d in eda_curve],
    ))
    for mname, mask in [('rf_imp_static', m_rf), ('relieff_static', m_rel), ('l1_lr_native', m_l1)]:
        o = m2o[mask]
        out.append(dict(
            suite=suite, group=source_group, task_key=task_key, rep=rep, method=mname,
            b128_prec=o['precision'], b500_prec=o['precision'],
            b1000_prec=o['precision'], b1000_ba=o['ba'], b1000_f1=o['f1'],
            curve_prec=None, curve_ba=None, curve_f1=None,
        ))
    return out


if __name__ == '__main__':
    t0 = perf_counter()
    wide_keys = sorted(p.name[:-8] for p in (PAIRED_ROOT / 'width_wide' / 'train').glob('*.json.gz'))
    with ProcessPoolExecutor(max_workers=16) as ex:
        w_rows = list(ex.map(eval_wide_relieff, wide_keys))
    df_w = pd.DataFrame(w_rows)
    df_w_ds = df_w.groupby(['group', 'dataset', 'p', 'cap'], as_index=False)[['precision', 'ba', 'f1', 'width', 'cpu']].mean()
    df_w_grp = df_w_ds.groupby('group', as_index=False)[['p', 'cap', 'precision', 'ba', 'f1', 'width', 'cpu']].mean()
    df_w_grp['method'] = 'relieff_static'
    df_w_grp.to_csv(ROOT / 'results' / 'numerical_wide_relieff.tsv', sep='\t', index=False)
    print(f'[Wide-K ReliefF done in {perf_counter()-t0:.1f}s] Prec={df_w_grp["precision"].mean()*100:.2f}%, BA={df_w_grp["ba"].mean()*100:.2f}%, F1={df_w_grp["f1"].mean()*100:.2f}%, #Feat={df_w_grp["width"].mean():.1f}')

    t1 = perf_counter()
    jobs = [j for j in tight_mod.build_jobs() if j[0] in ('author_13', 'prospective_10')]
    print(f'Launching Tight-K extra baselines on {len(jobs)} folds...')
    with ProcessPoolExecutor(max_workers=16) as ex:
        nested = list(ex.map(eval_tight_extra, jobs))
    flat = [r for sub in nested for r in sub]
    df_t = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('curve_')} for r in flat])
    df_t_ds = df_t.groupby(['suite', 'group', 'task_key', 'method'], as_index=False)[['b128_prec', 'b500_prec', 'b1000_prec', 'b1000_ba', 'b1000_f1']].mean()
    df_t_grp = df_t_ds.groupby(['suite', 'group', 'method'], as_index=False)[['b128_prec', 'b500_prec', 'b1000_prec', 'b1000_ba', 'b1000_f1']].mean()
    df_t_grp.to_csv(ROOT / 'results' / 'numerical_b1000_extra_per_source.tsv', sep='\t', index=False)

    eda_rows = [r for r in flat if r['method'] == 'eda']
    curve_out = []
    for suite in ('author_13', 'prospective_10'):
        s_rows = [r for r in eda_rows if r['suite'] == suite]
        by_grp = {}
        for r in s_rows:
            by_grp.setdefault((r['group'], r['task_key']), []).append((r['curve_prec'], r['curve_ba'], r['curve_f1']))
        grp_curves = {}
        for (g, tk), vals in by_grp.items():
            p_tk = np.mean([v[0] for v in vals], axis=0)
            b_tk = np.mean([v[1] for v in vals], axis=0)
            f_tk = np.mean([v[2] for v in vals], axis=0)
            grp_curves.setdefault(g, []).append((p_tk, b_tk, f_tk))
        p_all = np.mean([np.mean([v[0] for v in vs], axis=0) for vs in grp_curves.values()], axis=0)
        b_all = np.mean([np.mean([v[1] for v in vs], axis=0) for vs in grp_curves.values()], axis=0)
        f_all = np.mean([np.mean([v[2] for v in vs], axis=0) for vs in grp_curves.values()], axis=0)
        for b_idx in range(len(p_all)):
            curve_out.append(dict(suite=suite, method='eda', budget=b_idx + 1, precision=float(p_all[b_idx]), ba=float(b_all[b_idx]), f1=float(f_all[b_idx])))
    pd.DataFrame(curve_out).to_csv(ROOT / 'results' / 'numerical_b1000_eda_curves.tsv', sep='\t', index=False)
    print(f'[Tight-K extra done in {perf_counter()-t1:.1f}s]')
    for m in ['eda', 'rf_imp_static', 'relieff_static', 'l1_lr_native']:
        sub = df_t_grp[df_t_grp['method'] == m]
        print(f'  all_23 {m:15s}: B128={sub["b128_prec"].mean()*100:.2f}%, B500={sub["b500_prec"].mean()*100:.2f}%, B1000={sub["b1000_prec"].mean()*100:.2f}%, BA={sub["b1000_ba"].mean()*100:.2f}%, F1={sub["b1000_f1"].mean()*100:.2f}%')
