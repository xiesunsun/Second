"""Phase 2 Evaluation: LLM-Select (Score), LLM-Lasso, LSP-adapted, and new-6 5-run LLM stability.

Runs strictly on scroot-server (`lab`) using the exact frozen donor/outer splits and SmallBook/margins/Downstream
evaluators from Phase 1. Reads acquired gemini-3.8-flash cards from ./llm_cards/.
"""
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
from time import perf_counter
import gzip
import json
import sys
import warnings

import numpy as np
from scipy.stats import kendalltau
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_clinical_non_llm import (
    ALL_TASKS, BANDS, CAP, CHECKPOINTS, DOWN_BUDGETS, EARLIER_SIX, MAX_BUDGET,
    NEW_SIX, NS, REPEATS, Downstream, SmallBook, canonical, digest, get_donors,
    load_gemini_orders, load_role, log_weights, margins, measures, prep_std, read,
    require_lab, run_frozen_ext, seed, task_root, write,
)


def load_llm_cards(t):
    sc = {}
    pen = {}
    for kind in ("correct", "wrong"):
        s_obj = read(ROOT / "llm_cards/score" / t / kind / "summary.json")
        l_obj = read(ROOT / "llm_cards/lasso" / t / kind / "summary.json")
        assert s_obj["model"] == "gemini-3.8-flash" and l_obj["model"] == "gemini-3.8-flash"
        sc[kind] = np.asarray(s_obj["scores"], dtype=float)
        pen[kind] = np.asarray(l_obj["penalties"], dtype=float)
    return sc, pen


def run_weighted_l1_native(x_std, y, cap, weights_grid, rs):
    """Faithful native weighted L1 logistic regression via column scaling x_tilde_j = x_j / w_j.
    Selects (w_cfg, C) strictly via native 3-fold stratified L1-LR CV on Z-scored features.
    No SmallBook wrapper oracle evaluations and no silent static LLM fallback on all-zero L1 models.
    """
    p = x_std.shape[1]
    c_grid = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=rs)
    splits = list(skf.split(x_std, y))
    best_key = None
    best_cfg = None

    for tag, w in weights_grid:
        w = np.asarray(w, dtype=float)
        inv_w = 1.0 / np.maximum(w, 1e-6)
        x_scaled = x_std * inv_w[None, :]
        for c_val in c_grid:
            scs = []
            for tr, va in splits:
                clf = LogisticRegression(
                    penalty="l1", C=c_val, solver="liblinear",
                    class_weight="balanced", max_iter=500, random_state=rs,
                )
                clf.fit(x_scaled[tr], y[tr])
                pr = clf.predict(x_scaled[va])
                b0 = float(np.mean(pr[y[va] == 0] == 0))
                b1 = float(np.mean(pr[y[va] == 1] == 1))
                scs.append(0.5 * (b0 + b1))
            sc = float(np.mean(scs))
            key = (-sc, tag, c_val)
            if best_key is None or key < best_key:
                best_key = key
                best_cfg = (tag, c_val, inv_w)

    tag, c_val, inv_w = best_cfg
    x_scaled = x_std * inv_w[None, :]
    clf = LogisticRegression(
        penalty="l1", C=c_val, solver="liblinear",
        class_weight="balanced", max_iter=500, random_state=rs,
    )
    clf.fit(x_scaled, y)
    beta = np.abs(clf.coef_.ravel() * inv_w)
    nz_idx = np.flatnonzero(beta > 1e-8)
    nz = int(len(nz_idx))
    if nz > 0:
        order = nz_idx[np.argsort(-beta[nz_idx])]
        mask = tuple(sorted(map(int, order[:cap])))
        zero_fb = 0
    else:
        # Pure statistical fallback (univariate Z-score class mean diff), never static LLM
        corr = np.abs(x_std[y == 1].mean(axis=0) - x_std[y == 0].mean(axis=0))
        order = np.lexsort((np.arange(p), -corr))
        mask = tuple(sorted(map(int, order[:cap])))
        zero_fb = 1
    return mask, (tag, c_val, nz, zero_fb)


def eval_llm_task_repeat(job):
    t, r = job
    q_outer, v_outer, _ = load_role(t, "outer")
    sc_cards, pen_cards = load_llm_cards(t)
    p = len(sc_cards["correct"])
    records = []

    for n in NS:
        x, y, ids = get_donors(t, r, n)
        rs = seed("clinical-EA-20261123", t, r, n)
        book = SmallBook(x, y)
        x_std = prep_std(x)
        d_outer = margins(x, y, q_outer)
        d_outer_t = np.ascontiguousarray(d_outer.T)
        outer_cache = {}

        def score_outer(mask):
            mask = tuple(mask)
            if mask not in outer_cache:
                pred = d_outer_t[list(mask)].sum(0) > 0
                outer_cache[mask] = measures(pred, v_outer)
            return outer_cache[mask]

        down = Downstream(x, y, q_outer, v_outer, seed("clinical-readout-20261123", t, r, n)) if n >= 10 else None
        methods_out = {}

        def pack_static(name, mask, dt, extra=None):
            idx = book.buy(mask)
            ba, pr, f1 = score_outer(mask)
            inner_sc = book.scores[idx]["fold"]
            down_scores = {}
            if down is not None:
                sc_d = down.scores(mask)
                for b in DOWN_BUDGETS:
                    down_scores[str(b)] = sc_d
            methods_out[name] = dict(
                ba=[round(ba, 6)] * MAX_BUDGET,
                precision=[round(pr, 6)] * MAX_BUDGET,
                f1=[round(f1, 6)] * MAX_BUDGET,
                inner=[round(inner_sc, 6)] * MAX_BUDGET,
                ceiling=[round(ba, 6)] * MAX_BUDGET,
                size=[len(mask)] * MAX_BUDGET,
                cpu=[round(dt, 6)] * MAX_BUDGET,
                masks={str(b): list(mask) for b in CHECKPOINTS},
                down=down_scores,
                meta=extra or {},
            )

        def pack_search(name, bk, path, win_list, times):
            ba_arr, prec_arr, f1_arr, inner_arr, ceil_arr, sz_arr = [], [], [], [], [], []
            best_ceil = -1.0
            for i in range(MAX_BUDGET):
                pm = bk.masks[path[i]]
                p_ba = score_outer(pm)[0]
                if p_ba > best_ceil:
                    best_ceil = p_ba
                wm = bk.masks[win_list[i]]
                w_ba, w_pr, w_f1 = score_outer(wm)
                ba_arr.append(round(w_ba, 6))
                prec_arr.append(round(w_pr, 6))
                f1_arr.append(round(w_f1, 6))
                inner_arr.append(round(bk.scores[win_list[i]]["fold"], 6))
                ceil_arr.append(round(best_ceil, 6))
                sz_arr.append(len(wm))
            ck_masks = {str(b): list(bk.masks[win_list[b - 1]]) for b in CHECKPOINTS}
            down_scores = {}
            if down is not None:
                for b in DOWN_BUDGETS:
                    down_scores[str(b)] = down.scores(bk.masks[win_list[b - 1]])
            methods_out[name] = dict(
                ba=ba_arr, precision=prec_arr, f1=f1_arr,
                inner=inner_arr, ceiling=ceil_arr, size=sz_arr,
                cpu=[round(v, 6) for v in times],
                masks=ck_masks, down=down_scores, meta={},
            )

        for kind in ("correct", "wrong"):
            scores_vec = sc_cards[kind]
            pen_vec = pen_cards[kind]

            # 1. LLM-Select (Score) static top-CAP
            t0 = perf_counter()
            score_order = np.lexsort((np.arange(p), -scores_vec))
            m_score = tuple(sorted(map(int, score_order[:CAP])))
            pack_static(f"llm_select_score_{kind}", m_score, perf_counter() - t0)

            # 2. LLM-Select (Score) plugged into our paired search (bonus prior-form comparison)
            bk_s, st_s, ret_s, tms_s = run_frozen_ext(x, y, CAP, MAX_BUDGET, score_order, rs)
            pack_search(f"score_paired_{kind}", bk_s, st_s["path"], ret_s["paired"], tms_s)

            # 3. Faithful native LLM-Lasso (weighted L1 with alpha in {0.5, 1.0, 2.0}, 3-fold native CV)
            t0 = perf_counter()
            w_grid_lasso = [(a, np.power(pen_vec, a)) for a in (0.5, 1.0, 2.0)]
            m_lasso, meta_l = run_weighted_l1_native(x_std, y, CAP, w_grid_lasso, rs)
            pack_static(f"llm_lasso_{kind}", m_lasso, perf_counter() - t0,
                        extra=dict(alpha=meta_l[0], C=meta_l[1], nz=meta_l[2], zero_fb=meta_l[3]))

            # 4. Faithful native LSP-adapted (adaptive prior discount eta in {0, 0.5, 1, 2, 4}, 3-fold native CV)
            t0 = perf_counter()
            rel = np.clip(scores_vec, 0.05, 1.0)
            w_grid_lsp = [(eta, np.power(rel, -eta)) for eta in (0.0, 0.5, 1.0, 2.0, 4.0)]
            m_lsp, meta_lsp = run_weighted_l1_native(x_std, y, CAP, w_grid_lsp, rs)
            pack_static(f"lsp_adapted_{kind}", m_lsp, perf_counter() - t0,
                        extra=dict(eta=meta_lsp[0], C=meta_lsp[1], nz=meta_lsp[2], zero_fb=meta_lsp[3]))

        records.append(dict(t=t, r=r, n=n, k=CAP, p=p, ids=ids.tolist(), methods=methods_out))

    out_path = ROOT / "raw_clinical_llm" / f"{t}-{r:02d}.json.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(canonical(records), compresslevel=4, mtime=0))
    tmp.replace(out_path)
    return out_path.name


def eval_stability_task_repeat(job):
    """Evaluate all 5 independent gemini-3.8-flash cards (rep 0=frozen T=0, rep 1..4=T=0.3) on new-6."""
    t, r = job
    q_outer, v_outer, _ = load_role(t, "outer")
    frozen_orders, p = load_gemini_orders(t)
    x, y, ids = get_donors(t, r, 3)
    rs = seed("clinical-EA-20261123", t, r, 3)
    d_outer = margins(x, y, q_outer)
    d_outer_t = np.ascontiguousarray(d_outer.T)
    cache = {}

    def score_ba(mask):
        mask = tuple(mask)
        if mask not in cache:
            cache[mask] = measures(d_outer_t[list(mask)].sum(0) > 0, v_outer)[0]
        return cache[mask]

    out = {}
    for kind in ("correct", "wrong"):
        out[kind] = {}
        for rep in range(5):
            if rep == 0:
                ord_vec = frozen_orders[kind]
            else:
                card = read(ROOT / "llm_cards/stability" / t / kind / f"rep-{rep}" / "card.json")
                assert card["model"] == "gemini-3.8-flash"
                ol = list(map(int, card["order"]))
                ord_vec = np.array(ol + [j for j in range(p) if j not in set(ol)], dtype=int)
            bk, st, ret, _ = run_frozen_ext(x, y, CAP, 32, ord_vec, rs)
            anc = tuple(sorted(map(int, ord_vec[:CAP])))
            paired_ba = [round(score_ba(bk.masks[j]), 6) for j in ret["paired"]]
            static_ba = round(score_ba(anc), 6)
            out[kind][str(rep)] = dict(paired_ba=paired_ba, static_ba=static_ba, top4=list(anc))

    out_path = ROOT / "raw_llm_stability" / f"{t}-{r:02d}.json.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(canonical(out), compresslevel=4, mtime=0))
    tmp.replace(out_path)
    return out_path.name


def aggregate_llm():
    raw_dir = ROOT / "raw_clinical_llm"
    res_dir = ROOT / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    ba_data = {n: {} for n in NS}
    cpu_data = {n: {} for n in NS}
    zero_data = {n: {} for n in NS}
    down_data = {}

    for t in ALL_TASKS:
        for r in range(REPEATS):
            recs = json.loads(gzip.decompress((raw_dir / f"{t}-{r:02d}.json.gz").read_bytes()))
            for rec in recs:
                n = rec["n"]
                for m, payload in rec["methods"].items():
                    ba_data[n].setdefault(m, {}).setdefault(t, []).append(np.array(payload["ba"], dtype=float))
                    cpu_data[n].setdefault(m, {}).setdefault(t, []).append(np.array(payload["cpu"], dtype=float))
                    if "zero_fb" in payload.get("meta", {}):
                        zero_data[n].setdefault(m, []).append(int(payload["meta"]["zero_fb"]))
                    if n >= 10 and payload["down"]:
                        for b_str, dvals in payload["down"].items():
                            for clf_name in ("lr", "rf"):
                                (down_data.setdefault(n, {})
                                 .setdefault(m, {})
                                 .setdefault(int(b_str), {})
                                 .setdefault(clf_name, {})
                                 .setdefault(t, []).append(dvals[clf_name][0]))

    panels = {
        "earlier_six": EARLIER_SIX,
        "new_six": NEW_SIX,
        "zero_history_eight": ("cad", "gbsg") + NEW_SIX,
        "all_twelve": ALL_TASKS,
    }
    methods = sorted(ba_data[3].keys())

    # 1. bands
    lines = ["panel\tn\tband\tmethod\tba\tcpu_end"]
    for pname, tasks in panels.items():
        for n in NS:
            for bname, (lo, hi) in BANDS.items():
                w = log_weights(lo, hi)
                for m in methods:
                    s_ba, s_cp = [], []
                    for t in tasks:
                        arr_ba = np.asarray(ba_data[n][m][t])[:, lo - 1:hi]
                        arr_cp = np.asarray(cpu_data[n][m][t])[:, hi - 1]
                        s_ba.append(float((arr_ba @ w).mean()))
                        s_cp.append(float(arr_cp.mean()))
                    lines.append(f"{pname}\t{n}\t{bname}\t{m}\t{np.mean(s_ba):.6f}\t{np.mean(s_cp):.6f}")
    (res_dir / "clinical_llm_bands.tsv").write_text("\n".join(lines) + "\n")

    # 2. per source n=3
    w216 = log_weights(2, 16)
    lines = ["panel\ttask\tmethod\tb2_16\tb2\tb16\tb32\tb128"]
    for t in ALL_TASKS:
        pl = "earlier_six" if t in EARLIER_SIX else "new_six"
        for m in methods:
            arr = np.asarray(ba_data[3][m][t])
            v216 = float((arr[:, 1:16] @ w216).mean())
            vals = [float(arr[:, b - 1].mean()) for b in (2, 16, 32, 128)]
            lines.append(f"{pl}\t{t}\t{m}\t{v216:.6f}\t" + "\t".join(f"{v:.6f}" for v in vals))
    (res_dir / "clinical_llm_per_source_n3.tsv").write_text("\n".join(lines) + "\n")

    # 3. downstream n=10 (backward compatible) and all n in (10, 20, 30)
    lines10 = ["panel\tmodel\tbudget\tmethod\tba"]
    lines_all = ["panel\tn\tmodel\tbudget\tmethod\tba"]
    for pname, tasks in panels.items():
        for n_d in (10, 20, 30):
            if n_d not in down_data:
                continue
            for clf_name in ("lr", "rf"):
                for b in DOWN_BUDGETS:
                    for m in methods:
                        val = np.mean([np.mean(down_data[n_d][m][b][clf_name][t]) for t in tasks])
                        lines_all.append(f"{pname}\t{n_d}\t{clf_name}\t{b}\t{m}\t{val:.6f}")
                        if n_d == 10:
                            lines10.append(f"{pname}\t{clf_name}\t{b}\t{m}\t{val:.6f}")
    (res_dir / "clinical_llm_downstream_n10.tsv").write_text("\n".join(lines10) + "\n")
    (res_dir / "clinical_llm_downstream_all.tsv").write_text("\n".join(lines_all) + "\n")

    # 3b. L1 zero-degeneration rate ledger
    zlines = ["n\tmethod\tzero_count\ttotal\tzero_rate"]
    for n in NS:
        for m, arr in sorted(zero_data[n].items()):
            zc = int(sum(arr))
            tot = len(arr)
            zlines.append(f"{n}\t{m}\t{zc}\t{tot}\t{zc / max(1, tot):.4f}")
    (res_dir / "clinical_llm_l1_zero_rate.tsv").write_text("\n".join(zlines) + "\n")

    # 4. Stability summary on new-6
    stab_dir = ROOT / "raw_llm_stability"
    stab_lines = ["task\tkind\ttop4_jaccard_mean\ttop8_jaccard_mean\ttop8_kendall_mean\tpaired_b2_16_mean\tpaired_b2_16_sd\tpaired_b2_16_min\tpaired_b2_16_max\tstatic_ba_mean\tstatic_ba_sd"]
    per_rep_all = {kind: {rep: [] for rep in range(5)} for kind in ("correct", "wrong")}
    per_rep_static = {kind: {rep: [] for rep in range(5)} for kind in ("correct", "wrong")}

    for t in NEW_SIX:
        frozen_orders, p = load_gemini_orders(t)
        for kind in ("correct", "wrong"):
            # Order similarity across 5 cards
            cards5 = []
            for rep in range(5):
                if rep == 0:
                    cards5.append(list(map(int, frozen_orders[kind][:min(8, p)])))
                else:
                    c = read(ROOT / "llm_cards/stability" / t / kind / f"rep-{rep}" / "card.json")
                    cards5.append(list(map(int, c["order"])))
            j4, j8, kt = [], [], []
            for a, b in combinations(range(5), 2):
                s4a, s4b = set(cards5[a][:4]), set(cards5[b][:4])
                s8a, s8b = set(cards5[a]), set(cards5[b])
                j4.append(len(s4a & s4b) / len(s4a | s4b))
                j8.append(len(s8a & s8b) / len(s8a | s8b))
                union = sorted(s8a | s8b)
                ra = [cards5[a].index(u) if u in s8a else len(cards5[a]) for u in union]
                rb = [cards5[b].index(u) if u in s8b else len(cards5[b]) for u in union]
                tau, _ = kendalltau(ra, rb)
                kt.append(0.0 if np.isnan(tau) else float(tau))

            rep_b216 = []
            rep_stat = []
            for rep in range(5):
                r_vals, s_vals = [], []
                for r in range(REPEATS):
                    obj = json.loads(gzip.decompress((stab_dir / f"{t}-{r:02d}.json.gz").read_bytes()))
                    arr = np.asarray(obj[kind][str(rep)]["paired_ba"][1:16], dtype=float)
                    r_vals.append(float(arr @ w216))
                    s_vals.append(float(obj[kind][str(rep)]["static_ba"]))
                mv = float(np.mean(r_vals))
                sv = float(np.mean(s_vals))
                rep_b216.append(mv)
                rep_stat.append(sv)
                per_rep_all[kind][rep].append(mv)
                per_rep_static[kind][rep].append(sv)

            stab_lines.append(
                f"{t}\t{kind}\t{np.mean(j4):.4f}\t{np.mean(j8):.4f}\t{np.mean(kt):.4f}\t"
                f"{np.mean(rep_b216):.6f}\t{np.std(rep_b216, ddof=1):.6f}\t{np.min(rep_b216):.6f}\t{np.max(rep_b216):.6f}\t"
                f"{np.mean(rep_stat):.6f}\t{np.std(rep_stat, ddof=1):.6f}"
            )

    for kind in ("correct", "wrong"):
        panel_means = [float(np.mean(per_rep_all[kind][rep])) for rep in range(5)]
        stat_means = [float(np.mean(per_rep_static[kind][rep])) for rep in range(5)]
        stab_lines.append(
            f"NEW6_MEAN\t{kind}\t-\t-\t-\t"
            f"{np.mean(panel_means):.6f}\t{np.std(panel_means, ddof=1):.6f}\t{np.min(panel_means):.6f}\t{np.max(panel_means):.6f}\t"
            f"{np.mean(stat_means):.6f}\t{np.std(stat_means, ddof=1):.6f}"
        )
    (res_dir / "llm_stability_new6.tsv").write_text("\n".join(stab_lines) + "\n")

    # 5. Unified LLM Cost & Efficiency Ledger (including frozen top-8 cards)
    cost_lines = ["method\tpanel\ttask\tkind\tp\toffline_calls\tnative_calls\tprompt_tokens\toutput_tokens\tthoughts_tokens\ttotal_tokens\tapi_seconds"]
    for t in ALL_TASKS:
        pl = "earlier_six" if t in EARLIER_SIX else "new_six"
        base = task_root(t)
        for kind in ("correct", "wrong"):
            # Frozen Ours / LLM-Select (Rank)
            card = read(base / "cards" / t / kind / "card.json")
            # Locate matching attempt log
            att = None
            for ap in sorted((base / "cards" / t / kind).glob("attempt-*.json")):
                cand = read(ap)
                if cand.get("response_id") == card.get("response_id"):
                    att = cand
                    break
            u = (att or {}).get("usage", {})
            p_dim = len(read(base / "metadata" / f"{t}.json")["fields"])
            cost_lines.append(
                f"ours_top8_rank\t{pl}\t{t}\t{kind}\t{p_dim}\t1\t1\t"
                f"{u.get('promptTokenCount',0)}\t{u.get('candidatesTokenCount',0)}\t{u.get('thoughtsTokenCount',0)}\t{u.get('totalTokenCount',0)}\t{(att or {}).get('seconds',0.0):.3f}"
            )
            for mname, subdir in (("llm_select_score", "score"), ("llm_lasso", "lasso")):
                s = read(ROOT / "llm_cards" / subdir / t / kind / "summary.json")
                nat = s.get("native_per_feature_calls", s["n_calls"])
                cost_lines.append(
                    f"{mname}\t{pl}\t{t}\t{kind}\t{s['p']}\t{s['n_calls']}\t{nat}\t"
                    f"{s['prompt_tokens']}\t{s['output_tokens']}\t{s['thoughts_tokens']}\t{s['total_tokens']}\t{s['seconds']:.3f}"
                )
    (res_dir / "llm_cost_ledger.tsv").write_text("\n".join(cost_lines) + "\n")

    print("PHASE 2 AGGREGATION COMPLETE.", flush=True)


if __name__ == "__main__":
    require_lab()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("run", "all"):
        t0 = perf_counter()
        jobs_llm = [(t, r) for r in range(REPEATS) for t in ALL_TASKS]
        jobs_stab = [(t, r) for r in range(REPEATS) for t in NEW_SIX]
        with ProcessPoolExecutor(max_workers=20) as pool:
            d1 = list(pool.map(eval_llm_task_repeat, jobs_llm))
            d2 = list(pool.map(eval_stability_task_repeat, jobs_stab))
        print(f"PHASE 2 RUN COMPLETE: {len(d1)} llm + {len(d2)} stability in {perf_counter() - t0:.1f}s", flush=True)
    if cmd in ("aggregate", "all"):
        aggregate_llm()
