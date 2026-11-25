"""Step 3: Exact paired Holm-Bonferroni statistics + 5-run LLM stability for SP-PD (tau=1.5 vs tau=1.0).

Runs strictly on scroot-server (`lab`) over the sealed raw_clinical_non_llm / raw_clinical_llm / llm_cards files.
Zero new LLM API calls.
"""
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from time import perf_counter
import gzip
import json
import sys
import warnings

import numpy as np
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_clinical_non_llm import (
    ALL_TASKS, CAP, EARLIER_SIX, NEW_SIX, REPEATS, delivery_any, get_donors,
    load_gemini_orders, load_role, log_weights, margins, measures, read,
    require_lab, run_frozen_ext, seed,
)


def exact_signflip_p(diffs):
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    signs = np.asarray(list(product((-1.0, 1.0), repeat=n)), dtype=float)
    return float(np.mean(np.abs(signs @ d / n) >= abs(d.mean()) - 1e-12))


def run_stability_tau15_one(job):
    t, r = job
    q_outer, v_outer, _ = load_role(t, "outer")
    frozen_orders, p = load_gemini_orders(t)
    x, y, _ = get_donors(t, r, 3)
    rs = seed("clinical-EA-20261123", t, r, 3)
    d_outer = margins(x, y, q_outer)
    d_outer_t = np.ascontiguousarray(d_outer.T)
    cache = {}

    def score_ba(mask):
        mask = tuple(mask)
        if mask not in cache:
            cache[mask] = measures(d_outer_t[list(mask)].sum(0) > 0, v_outer)[0]
        return cache[mask]

    w216 = log_weights(2, 16)
    out = {}
    for kind in ("correct", "wrong"):
        out[kind] = {}
        for rep in range(5):
            if rep == 0:
                ord_vec = frozen_orders[kind]
            else:
                card = read(ROOT / "llm_cards/stability" / t / kind / f"rep-{rep}" / "card.json")
                ol = list(map(int, card["order"]))
                ord_vec = np.array(ol + [j for j in range(p) if j not in set(ol)], dtype=int)
            bk, st, ret10, _ = run_frozen_ext(x, y, CAP, 32, ord_vec, rs)
            anc = tuple(sorted(map(int, ord_vec[:CAP])))
            ret15 = delivery_any(bk, st, anc, tau=1.5)
            ba10 = float(np.asarray([score_ba(bk.masks[j]) for j in ret10["paired"][1:16]]) @ w216)
            ba15 = float(np.asarray([score_ba(bk.masks[j]) for j in ret15["paired"][1:16]]) @ w216)
            ba_st = float(score_ba(anc))
            out[kind][rep] = (ba10, ba15, ba_st)
    return t, r, out


def compute_holm_stats():
    raw1 = ROOT / "raw_clinical_non_llm"
    raw2 = ROOT / "raw_clinical_llm"
    res_dir = ROOT / "results"
    w216 = log_weights(2, 16)

    # Load per-task means across 20 repeats for n in (3, 10, 20, 30)
    # data[(n, band_or_down, method)][task] = mean BA over 20 repeats
    data = {}
    for t in ALL_TASKS:
        acc = {}
        for r in range(REPEATS):
            for raw_dir in (raw1, raw2):
                recs = json.loads(gzip.decompress((raw_dir / f"{t}-{r:02d}.json.gz").read_bytes()))
                for rec in recs:
                    n = rec["n"]
                    for m, payload in rec["methods"].items():
                        arr = np.asarray(payload["ba"], dtype=float)
                        acc.setdefault((n, "b2_16", m), []).append(float(arr[1:16] @ w216))
                        acc.setdefault((n, "b16", m), []).append(float(arr[15]))
                        acc.setdefault((n, "b32", m), []).append(float(arr[31]))
                        acc.setdefault((n, "b128", m), []).append(float(arr[127]))
                        if n >= 10 and payload.get("down"):
                            for clf_name in ("lr", "rf"):
                                acc.setdefault((n, f"down_{clf_name}_b16", m), []).append(
                                    float(payload["down"]["16"][clf_name][0])
                                )
        for k, vals in acc.items():
            data.setdefault(k, {})[t] = float(np.mean(vals))

    competitors_aligned = [
        "pibo_correct", "llm_lasso_correct", "lsp_adapted_correct",
        "mel", "eda", "bpso", "ga", "numeric_ext",
        "relieff_static", "rf_imp_static", "mrmr_static", "l1_lr_static",
        "gemini_static", "llm_select_score_correct",
    ]
    competitors_wrong = [
        "pibo_wrong", "llm_lasso_wrong", "lsp_adapted_wrong", "wrong_static", "llm_select_score_wrong",
    ]

    lines = ["ref_method\tn\tmetric\tcompetitor\tref_mean\tcomp_mean\tdiff_pp\tp_raw\tp_holm\twtl"]
    for ref_m, comps in (("gemini_paired_tau15", competitors_aligned), ("gemini_paired_ext", competitors_aligned),
                         ("wrong_paired_tau15", competitors_wrong), ("wrong_paired_ext", competitors_wrong)):
        for n in (3, 10, 20, 30):
            metrics_list = ["b2_16", "b128"] + (["down_lr_b16", "down_rf_b16"] if n >= 10 else [])
            for met in metrics_list:
                rows_tmp = []
                pvals = []
                for c in comps:
                    rv = np.array([data[(n, met, ref_m)][t] for t in ALL_TASKS])
                    cv = np.array([data[(n, met, c)][t] for t in ALL_TASKS])
                    diff = rv - cv
                    pv = exact_signflip_p(diff)
                    wi = int(np.sum(diff > 1e-10))
                    ti = int(np.sum(np.abs(diff) <= 1e-10))
                    lo = int(np.sum(diff < -1e-10))
                    rows_tmp.append((c, float(rv.mean() * 100), float(cv.mean() * 100), float(diff.mean() * 100), pv, f"{wi}/{ti}/{lo}"))
                    pvals.append(pv)
                _, pholm, _, _ = multipletests(pvals, method="holm")
                for (c, rm, cm, dm, pv, wtl), ph in zip(rows_tmp, pholm):
                    lines.append(f"{ref_m}\t{n}\t{met}\t{c}\t{rm:.2f}\t{cm:.2f}\t{dm:+.2f}\t{pv:.6f}\t{ph:.6f}\t{wtl}")

    (res_dir / "clinical_paired_holm_stats.tsv").write_text("\n".join(lines) + "\n")
    print("HOLM STATS WRITTEN:", len(lines) - 1, "comparisons", flush=True)


def compute_stability_tau15():
    t0 = perf_counter()
    jobs = [(t, r) for t in NEW_SIX for r in range(REPEATS)]
    with ProcessPoolExecutor(max_workers=16) as pool:
        rows = list(pool.map(run_stability_tau15_one, jobs))
    lines = ["task\tkind\ttau10_mean\ttau10_sd\ttau15_mean\ttau15_sd\tstatic_mean\tstatic_sd"]
    for kind in ("correct", "wrong"):
        p10, p15, pst = {rep: [] for rep in range(5)}, {rep: [] for rep in range(5)}, {rep: [] for rep in range(5)}
        for t in NEW_SIX:
            m10, m15, mst = [], [], []
            for rep in range(5):
                v10 = float(np.mean([o[kind][rep][0] for tt, rr, o in rows if tt == t]))
                v15 = float(np.mean([o[kind][rep][1] for tt, rr, o in rows if tt == t]))
                vst = float(np.mean([o[kind][rep][2] for tt, rr, o in rows if tt == t]))
                m10.append(v10); m15.append(v15); mst.append(vst)
                p10[rep].append(v10); p15[rep].append(v15); pst[rep].append(vst)
            lines.append(
                f"{t}\t{kind}\t{np.mean(m10)*100:.2f}\t{np.std(m10, ddof=1)*100:.2f}\t"
                f"{np.mean(m15)*100:.2f}\t{np.std(m15, ddof=1)*100:.2f}\t"
                f"{np.mean(mst)*100:.2f}\t{np.std(mst, ddof=1)*100:.2f}"
            )
        pm10 = [float(np.mean(p10[rep])) for rep in range(5)]
        pm15 = [float(np.mean(p15[rep])) for rep in range(5)]
        pmst = [float(np.mean(pst[rep])) for rep in range(5)]
        lines.append(
            f"NEW6_MEAN\t{kind}\t{np.mean(pm10)*100:.2f}\t{np.std(pm10, ddof=1)*100:.2f}\t"
            f"{np.mean(pm15)*100:.2f}\t{np.std(pm15, ddof=1)*100:.2f}\t"
            f"{np.mean(pmst)*100:.2f}\t{np.std(pmst, ddof=1)*100:.2f}"
        )
    (ROOT / "results/llm_stability_tau15_new6.tsv").write_text("\n".join(lines) + "\n")
    print(f"STABILITY TAU15 COMPLETE in {perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    require_lab()
    compute_holm_stats()
    compute_stability_tau15()
