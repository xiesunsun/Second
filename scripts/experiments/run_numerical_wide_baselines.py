"""Step 2 (Extended): High-Dimensional Author-13 Wide-K (K = 0.15p, p = 500..10,000, B = 1..1000).

Runs strictly on scroot-server (`lab`) using the exact 42 frozen folds, ExactBook (10-fold 5-NN C kernel),
and runner.outer_metrics from paired-delivery-20261123/width_wide.
Verifies exact B=1..128 parity against frozen geometry (87.5818%) and mel (84.5264%),
then extends all iterative methods (geometry, mel, ga, bpso, eda) cleanly to B=1,000
alongside native static baselines (mRMR, L1-LR, RF-Imp).
"""
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from itertools import product
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import gzip
import json
import sys
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
PAIRED_ROOT = Path("/home/sunxie/paired-delivery-20261123.lCTbZvTQ")
sys.path.insert(0, str(PAIRED_ROOT))

from engine import ExactBook, ProposalSearch, projected, read, require_lab, run_seed, schedule
from handoff import Recombination, evolve
from local_study import runner
from exact_five import kernel

MAX_BUDGET = 1000
BANDS = {
    "early": (1, 8),
    "b8": (8, 8),
    "main": (8, 128),
    "b128": (128, 128),
    "b256": (256, 256),
    "b500": (500, 500),
    "b1000": (1000, 1000),
    "b8_500": (8, 500),
    "b8_1000": (8, 1000),
}


def canonical(v):
    return (json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def weights_wide(lo, hi):
    idx = np.arange(lo - 1, hi, dtype=float)
    w = np.log1p(1.0 / (idx + 1.0))
    return w / w.sum()


def winners_from_path(book, path):
    win = []
    best = path[0]
    for j in path:
        best = min((best, j), key=lambda idx: (-book.scores[idx]["fold"], len(book.masks[idx])))
        win.append(best)
    return win


def repair_mask(vec, scores, cap, rng):
    p = len(vec)
    ones = np.flatnonzero(vec > 0)
    if len(ones) == 0:
        if scores is not None:
            return (int(np.argmax(scores + rng.uniform(0, 1e-9, size=p))),)
        return (int(rng.randint(0, p)),)
    if len(ones) > cap:
        if scores is not None:
            chosen = ones[np.argsort(-(scores[ones] + rng.uniform(0, 1e-9, size=len(ones))))[:cap]]
        else:
            chosen = rng.choice(ones, size=cap, replace=False)
        return tuple(sorted(map(int, chosen)))
    return tuple(sorted(map(int, ones)))


def dedup_mask(mask, seen, p, cap, rng):
    if mask not in seen:
        return mask
    cur = set(mask)
    for _ in range(48):
        cand = set(cur)
        op = rng.randint(0, 3)
        if op == 0 and len(cand) < cap:
            j = int(rng.randint(0, p))
            if j not in cand:
                cand.add(j)
        elif op == 1 and len(cand) > 1:
            cand.remove(int(sorted(cand)[rng.randint(len(cand))]))
        else:
            j = int(rng.randint(0, p))
            if j not in cand and cand:
                cand.remove(int(sorted(cand)[rng.randint(len(cand))]))
                cand.add(j)
        t = tuple(sorted(cand))
        if t and t not in seen:
            return t
    for _ in range(256):
        k = int(rng.randint(1, min(cap, p) + 1))
        t = tuple(sorted(map(int, rng.choice(p, size=k, replace=False))))
        if t not in seen:
            return t
    return mask


def f_stat_scores(x, y):
    classes = np.unique(y)
    mu = x.mean(axis=0)
    ssb = np.zeros(x.shape[1], dtype=float)
    ssw = np.zeros(x.shape[1], dtype=float)
    for c in classes:
        xc = x[y == c]
        muc = xc.mean(axis=0)
        ssb += len(xc) * (muc - mu) ** 2
        ssw += ((xc - muc) ** 2).sum(axis=0)
    return ssb / (ssw + 1e-12)


def run_geometry_wide(x, y, rs, cap, budget):
    t0 = perf_counter()
    bk = ExactBook(x, y, rs)
    s = ProposalSearch(x, y, rs, budget, limit=cap, book=bk)
    menu, family = schedule(s)
    geo = evolve(s, "handoff_mut", menu, family, Recombination(s))
    dt = perf_counter() - t0
    return bk, geo["path"], geo["win"], dt


def run_mel_wide(x, y, rs, cap, budget):
    t0 = perf_counter()
    bk = ExactBook(x, y, rs)
    s = ProposalSearch(x, y, rs, budget, limit=cap, book=bk)
    adapter = SimpleNamespace(
        x=s.x, y=s.y, seed=rs, book=s.book,
        base=SimpleNamespace(cap=s.cap, tree=s.tree, vote=s.menu),
    )
    ref = projected.mel_path(adapter, budget)
    dt = perf_counter() - t0
    return bk, ref["path"], ref["winners"], dt


def run_ga_wide(x, y, rs, cap, budget):
    t0 = perf_counter()
    bk = ExactBook(x, y, rs)
    p = x.shape[1]
    rng = np.random.RandomState(rs + 101)
    pop_size = 20
    seen, path = set(), []

    def m2v(m):
        v = np.zeros(p, dtype=np.int8)
        v[list(m)] = 1
        return v

    pop, fit = [], []
    for _ in range(pop_size):
        k = int(rng.randint(1, min(cap, p) + 1))
        m = dedup_mask(tuple(sorted(map(int, rng.choice(p, size=k, replace=False)))), seen, p, cap, rng)
        seen.add(m)
        j = bk.buy(m)
        path.append(j)
        pop.append(m2v(m))
        fit.append(bk.scores[j]["fold"])

    pm = max(1.0 / p, 0.005)
    while len(path) < budget:
        new_pop, new_fit = [], []
        order = np.argsort(-np.asarray(fit))
        for idx in order[:2]:
            new_pop.append(pop[idx].copy())
            new_fit.append(fit[idx])
        while len(new_pop) < pop_size and len(path) < budget:
            i1, i2 = rng.randint(len(pop)), rng.randint(len(pop))
            p1 = pop[i1] if fit[i1] >= fit[i2] else pop[i2]
            j1, j2 = rng.randint(len(pop)), rng.randint(len(pop))
            p2 = pop[j1] if fit[j1] >= fit[j2] else pop[j2]
            c1 = p1.copy()
            if rng.rand() < 0.9 and p > 2:
                a, b = sorted(rng.choice(p, size=2, replace=False))
                c1[a:b] = p2[a:b]
            flips = rng.rand(p) < pm
            c1[flips] ^= 1
            m = dedup_mask(repair_mask(c1, None, cap, rng), seen, p, cap, rng)
            seen.add(m)
            j = bk.buy(m)
            path.append(j)
            new_pop.append(m2v(m))
            new_fit.append(bk.scores[j]["fold"])
        pop, fit = new_pop, new_fit
    dt = perf_counter() - t0
    return bk, path, winners_from_path(bk, path), dt


def run_bpso_wide(x, y, rs, cap, budget):
    t0 = perf_counter()
    bk = ExactBook(x, y, rs)
    p = x.shape[1]
    rng = np.random.RandomState(rs + 202)
    n_part = 20
    seen, path = set(), []
    vel = rng.uniform(-1.0, 1.0, size=(n_part, p))
    pos = np.zeros((n_part, p), dtype=np.int8)
    pbest = np.zeros((n_part, p), dtype=np.int8)
    pbest_fit = np.full(n_part, -1.0)
    gbest = np.zeros(p, dtype=np.int8)
    gbest_fit = -1.0

    n_iters = max(1, (budget + n_part - 1) // n_part)
    for it in range(n_iters):
        w = 0.9 - 0.5 * (it / max(1, n_iters - 1))
        for i in range(n_part):
            if len(path) >= budget:
                break
            if it == 0:
                k = int(rng.randint(1, min(cap, p) + 1))
                m = tuple(sorted(map(int, rng.choice(p, size=k, replace=False))))
            else:
                r1, r2 = rng.rand(p), rng.rand(p)
                vel[i] = np.clip(w * vel[i] + 1.5 * r1 * (pbest[i] - pos[i]) + 1.5 * r2 * (gbest - pos[i]), -4.0, 4.0)
                probs = 1.0 / (1.0 + np.exp(-vel[i]))
                cand = (rng.rand(p) < probs).astype(np.int8)
                m = repair_mask(cand, probs, cap, rng)
            m = dedup_mask(m, seen, p, cap, rng)
            seen.add(m)
            v = np.zeros(p, dtype=np.int8)
            v[list(m)] = 1
            pos[i] = v
            j = bk.buy(m)
            path.append(j)
            f = bk.scores[j]["fold"]
            if f > pbest_fit[i]:
                pbest_fit[i] = f
                pbest[i] = v.copy()
            if f > gbest_fit:
                gbest_fit = f
                gbest = v.copy()
    dt = perf_counter() - t0
    return bk, path, winners_from_path(bk, path), dt


def run_eda_wide(x, y, rs, cap, budget):
    t0 = perf_counter()
    bk = ExactBook(x, y, rs)
    p = x.shape[1]
    rng = np.random.RandomState(rs + 303)
    gen_size = 16
    seen, path = set(), []
    probs = np.full(p, min(0.5, float(cap) / float(p)), dtype=float)

    while len(path) < budget:
        g_vecs, g_fits = [], []
        for _ in range(gen_size):
            if len(path) >= budget:
                break
            cand = (rng.rand(p) < probs).astype(np.int8)
            m = dedup_mask(repair_mask(cand, probs, cap, rng), seen, p, cap, rng)
            seen.add(m)
            j = bk.buy(m)
            path.append(j)
            v = np.zeros(p, dtype=float)
            v[list(m)] = 1.0
            g_vecs.append(v)
            g_fits.append(bk.scores[j]["fold"])
        if g_vecs:
            order = np.argsort(-np.asarray(g_fits))
            n_el = max(1, len(g_vecs) // 4)
            elite_mean = np.mean([g_vecs[idx] for idx in order[:n_el]], axis=0)
            probs = np.clip(0.8 * probs + 0.2 * elite_mean, 0.005, 0.95)
    dt = perf_counter() - t0
    return bk, path, winners_from_path(bk, path), dt


def run_mrmr_wide(x, y, cap):
    t0 = perf_counter()
    xs = StandardScaler().fit_transform(x)
    p = xs.shape[1]
    fsc = f_stat_scores(xs, y)
    pool_k = min(p, max(512, 2 * cap))
    pool = np.argsort(-fsc)[:pool_k]
    xp = xs[:, pool]
    norms = np.linalg.norm(xp, axis=0)
    norms[norms < 1e-12] = 1.0
    x_unit = xp / norms[None, :]
    rel = fsc[pool]
    rel_norm = rel / (rel.max() + 1e-12)

    selected_local = []
    avail = np.ones(pool_k, dtype=bool)
    sum_corr = np.zeros(pool_k, dtype=float)

    first = int(np.argmax(rel_norm))
    selected_local.append(first)
    avail[first] = False
    for step in range(1, min(cap, pool_k)):
        last = selected_local[-1]
        sum_corr += np.abs(x_unit.T @ x_unit[:, last])
        score = rel_norm - (sum_corr / float(step))
        score[~avail] = -1e18
        nxt = int(np.argmax(score))
        selected_local.append(nxt)
        avail[nxt] = False

    mask = tuple(sorted(map(int, pool[selected_local])))
    return mask, perf_counter() - t0


def run_l1_wide(x, y, rs, cap):
    t0 = perf_counter()
    xs = StandardScaler().fit_transform(x)
    min_cls = int(np.min(np.bincount(y.astype(int))))
    n_splits = min(3, max(2, min_cls))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)
    splits = list(skf.split(xs, y))
    c_grid = (0.1, 0.5, 1.0, 5.0)
    best_sc, best_c = -1.0, 1.0
    for c_val in c_grid:
        scs = []
        for tr, va in splits:
            clf = LogisticRegression(
                penalty="l1", C=c_val, solver="liblinear",
                class_weight="balanced", max_iter=200, tol=1e-3, random_state=rs,
            )
            clf.fit(xs[tr], y[tr])
            pr = clf.predict(xs[va])
            cls_va = np.unique(y[va])
            scs.append(float(np.mean([np.mean(pr[y[va] == c] == c) for c in cls_va])))
        sc = float(np.mean(scs))
        if sc > best_sc:
            best_sc, best_c = sc, c_val

    clf = LogisticRegression(
        penalty="l1", C=best_c, solver="liblinear",
        class_weight="balanced", max_iter=200, tol=1e-3, random_state=rs,
    )
    clf.fit(xs, y)
    coef = np.abs(clf.coef_)
    imp = coef.max(axis=0) if coef.ndim == 2 else coef.ravel()
    nz = np.flatnonzero(imp > 1e-8)
    if len(nz) > 0:
        ord_nz = nz[np.argsort(-imp[nz])]
        mask = tuple(sorted(map(int, ord_nz[:cap])))
    else:
        fsc = f_stat_scores(xs, y)
        mask = tuple(sorted(map(int, np.argsort(-fsc)[:cap])))
    return mask, perf_counter() - t0


def run_rf_wide(x, y, rs, cap):
    t0 = perf_counter()
    xs = StandardScaler().fit_transform(x)
    p = xs.shape[1]
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=rs, n_jobs=1)
    clf.fit(xs, y)
    imp = clf.feature_importances_
    fsc = f_stat_scores(xs, y)
    order = np.lexsort((np.arange(p), -fsc, -imp))
    mask = tuple(sorted(map(int, order[:cap])))
    return mask, perf_counter() - t0


def eval_one_wide(key):
    kernel()
    folder = PAIRED_ROOT / "width_wide"
    tr_f = folder / "train" / f"{key}.json.gz"
    a = read(tr_f)
    c = a["case"]
    rs = run_seed(c)
    cap = int(a["cap"])
    p = int(a["p"])
    x, y = projected.arrays(c)
    q, v = projected.arrays(c, True)

    with np.load(tr_f.with_name(key + ".npz")) as tz:
        old_masks = [tuple(map(int, np.flatnonzero(b))) for b in np.unpackbits(tz["masks"], axis=1)[:, :p]]

    unique_masks = []
    mask_to_uid = {}

    def register(mask):
        if mask not in mask_to_uid:
            mask_to_uid[mask] = len(unique_masks)
            unique_masks.append(mask)
        return mask_to_uid[mask]

    search_traces = {}
    for mname, fn in (
        ("geometry", lambda: run_geometry_wide(x, y, rs, cap, MAX_BUDGET)),
        ("mel", lambda: run_mel_wide(x, y, rs, cap, MAX_BUDGET)),
        ("ga", lambda: run_ga_wide(x, y, rs, cap, MAX_BUDGET)),
        ("bpso", lambda: run_bpso_wide(x, y, rs, cap, MAX_BUDGET)),
        ("eda", lambda: run_eda_wide(x, y, rs, cap, MAX_BUDGET)),
    ):
        bk, path, win, dt = fn()
        # Strict B=1..128 mask parity check against frozen width_wide for geometry and mel
        if mname in ("geometry", "mel"):
            old_win = [old_masks[j] for j in a["arms"][mname]["raw"]]
            new_win128 = [bk.masks[j] for j in win[:128]]
            assert old_win == new_win128, f"Wide-K B128 parity mismatch on {key} {mname}"
        uids = [register(bk.masks[j]) for j in win]
        widths = [len(bk.masks[j]) for j in win]
        search_traces[mname] = (uids, widths, float(dt))

    static_traces = {}
    for mname, fn in (
        ("mrmr_static", lambda: run_mrmr_wide(x, y, cap)),
        ("l1_lr_static", lambda: run_l1_wide(x, y, rs, cap)),
        ("rf_imp_static", lambda: run_rf_wide(x, y, rs, cap)),
    ):
        m, dt = fn()
        uid = register(m)
        static_traces[mname] = (uid, len(m), float(dt))

    np_masks = [np.asarray(m, dtype=int) for m in unique_masks]
    om = runner.outer_metrics(x, y, q, v, np_masks)

    arms_out = {}
    for mname, (uids, widths, dt) in search_traces.items():
        arms_out[mname] = dict(
            precision=[round(float(om["precision"][u]), 6) for u in uids],
            ba=[round(float(om["ba"][u]), 6) for u in uids],
            f1=[round(float(om["f1"][u]), 6) for u in uids],
            width=widths,
            cpu=round(dt, 4),
        )
    for mname, (uid, wlen, dt) in static_traces.items():
        pr_v = round(float(om["precision"][uid]), 6)
        ba_v = round(float(om["ba"][uid]), 6)
        f1_v = round(float(om["f1"][uid]), 6)
        arms_out[mname] = dict(
            precision=[pr_v] * MAX_BUDGET,
            ba=[ba_v] * MAX_BUDGET,
            f1=[f1_v] * MAX_BUDGET,
            width=[wlen] * MAX_BUDGET,
            cpu=round(dt, 4),
        )

    rec = dict(
        key=key,
        cohort=c[0],
        task=c[1],
        group=a["group"],
        fold=int(c[3]),
        n=int(a["n"]),
        p=p,
        cap=cap,
        arms=arms_out,
    )
    out_dir = ROOT / "raw_numerical_wide"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{key}.json.gz"
    tmp = out_path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(canonical(rec), compresslevel=4, mtime=0))
    tmp.replace(out_path)
    return key, p, cap, len(unique_masks)


def paired_exact(diff):
    d = np.asarray(diff, dtype=float) * 100.0
    n = len(d)
    signs = np.asarray(list(product((-1.0, 1.0), repeat=n)), dtype=float)
    p_val = float(np.mean(np.abs(signs @ d / n) >= abs(d.mean()) - 1e-12))
    wins = int(np.sum(d > 1e-10))
    ties = int(np.sum(np.abs(d) <= 1e-10))
    losses = int(np.sum(d < -1e-10))
    return float(d.mean()), p_val, wins, ties, losses


def aggregate_wide():
    raw_dir = ROOT / "raw_numerical_wide"
    res_dir = ROOT / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(gzip.decompress(fp.read_bytes())) for fp in sorted(raw_dir.glob("*.json.gz"))]
    assert len(recs) == 42, f"Expected 42 folds, got {len(recs)}"

    methods = sorted(recs[0]["arms"].keys())
    groups = sorted(set(r["group"] for r in recs))
    assert len(groups) == 13, f"Expected 13 independent groups, got {len(groups)}"

    band_lines = ["band\tmethod\tprecision\tba\tf1\twidth\tcpu_end\tdiff_vs_geom_prec\tp_exact_prec\twtl_prec"]
    per_group_lines = ["group\tp\tcap\tband\tmethod\tprecision\tba\tf1\twidth"]

    for bname, (lo, hi) in BANDS.items():
        w = weights_wide(lo, hi)
        grp_vals = {m: {} for m in methods}
        for m in methods:
            for g in groups:
                tasks_in_g = sorted(set(r["task"] for r in recs if r["group"] == g))
                t_prec, t_ba, t_f1, t_w, t_cpu = [], [], [], [], []
                for tk in tasks_in_g:
                    f_recs = [r for r in recs if r["group"] == g and r["task"] == tk]
                    t_prec.append(np.mean([float(np.asarray(r["arms"][m]["precision"][lo - 1:hi]) @ w) for r in f_recs]))
                    t_ba.append(np.mean([float(np.asarray(r["arms"][m]["ba"][lo - 1:hi]) @ w) for r in f_recs]))
                    t_f1.append(np.mean([float(np.asarray(r["arms"][m]["f1"][lo - 1:hi]) @ w) for r in f_recs]))
                    t_w.append(np.mean([float(np.asarray(r["arms"][m]["width"][lo - 1:hi]) @ w) for r in f_recs]))
                    t_cpu.append(np.mean([float(r["arms"][m]["cpu"]) for r in f_recs]))
                grp_vals[m][g] = (
                    float(np.mean(t_prec)), float(np.mean(t_ba)),
                    float(np.mean(t_f1)), float(np.mean(t_w)), float(np.mean(t_cpu)),
                )
                if bname in ("main", "b128", "b500", "b1000"):
                    r0 = next(r for r in recs if r["group"] == g)
                    per_group_lines.append(
                        f"{g}\t{r0['p']}\t{r0['cap']}\t{bname}\t{m}\t"
                        f"{grp_vals[m][g][0]:.6f}\t{grp_vals[m][g][1]:.6f}\t{grp_vals[m][g][2]:.6f}\t{grp_vals[m][g][3]:.2f}"
                    )

        if bname == "main":
            g_prec = float(np.mean([grp_vals["geometry"][g][0] for g in groups]))
            m_prec = float(np.mean([grp_vals["mel"][g][0] for g in groups]))
            assert abs(g_prec - 0.8758176) < 1e-5, f"Geometry parity mismatch: {g_prec}"
            assert abs(m_prec - 0.8452644) < 1e-5, f"MEL parity mismatch: {m_prec}"
            print(f"WIDE-K B=1000 PARITY PASS (B8-128): geometry={g_prec*100:.4f}%, mel={m_prec*100:.4f}%", flush=True)

        for m in methods:
            pr = float(np.mean([grp_vals[m][g][0] for g in groups]))
            ba = float(np.mean([grp_vals[m][g][1] for g in groups]))
            f1 = float(np.mean([grp_vals[m][g][2] for g in groups]))
            wd = float(np.mean([grp_vals[m][g][3] for g in groups]))
            cp = float(np.mean([grp_vals[m][g][4] for g in groups]))
            if m == "geometry":
                diff_str, p_str, wtl_str = "0.00", "-", "-"
            else:
                diffs = [grp_vals["geometry"][g][0] - grp_vals[m][g][0] for g in groups]
                eff, pv, wi, ti, lo_n = paired_exact(diffs)
                diff_str = f"{eff:+.2f}"
                p_str = f"{pv:.6f}"
                wtl_str = f"{wi}/{ti}/{lo_n}"
            band_lines.append(f"{bname}\t{m}\t{pr:.6f}\t{ba:.6f}\t{f1:.6f}\t{wd:.2f}\t{cp:.3f}\t{diff_str}\t{p_str}\t{wtl_str}")

    (res_dir / "numerical_wide_bands.tsv").write_text("\n".join(band_lines) + "\n")
    (res_dir / "numerical_wide_per_group.tsv").write_text("\n".join(per_group_lines) + "\n")

    # Full B=1..1000 curves (group-averaged) for IEEE plotting
    curve_lines = ["method\tbudget\tprecision\tba\tf1\twidth"]
    for m in methods:
        g_pr, g_ba, g_f1, g_w = [], [], [], []
        for g in groups:
            tasks_in_g = sorted(set(r["task"] for r in recs if r["group"] == g))
            tp, tb, tf, tw = [], [], [], []
            for tk in tasks_in_g:
                f_recs = [r for r in recs if r["group"] == g and r["task"] == tk]
                tp.append(np.mean([r["arms"][m]["precision"] for r in f_recs], axis=0))
                tb.append(np.mean([r["arms"][m]["ba"] for r in f_recs], axis=0))
                tf.append(np.mean([r["arms"][m]["f1"] for r in f_recs], axis=0))
                tw.append(np.mean([r["arms"][m]["width"] for r in f_recs], axis=0))
            g_pr.append(np.mean(tp, axis=0))
            g_ba.append(np.mean(tb, axis=0))
            g_f1.append(np.mean(tf, axis=0))
            g_w.append(np.mean(tw, axis=0))
        c_pr = np.mean(g_pr, axis=0)
        c_ba = np.mean(g_ba, axis=0)
        c_f1 = np.mean(g_f1, axis=0)
        c_w = np.mean(g_w, axis=0)
        for b in range(MAX_BUDGET):
            curve_lines.append(f"{m}\t{b+1}\t{c_pr[b]:.6f}\t{c_ba[b]:.6f}\t{c_f1[b]:.6f}\t{c_w[b]:.2f}")
    (res_dir / "numerical_wide_curves.tsv").write_text("\n".join(curve_lines) + "\n")
    print("WIDE-K B=1000 AGGREGATION COMPLETE.", flush=True)


if __name__ == "__main__":
    require_lab()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("run", "all"):
        t0 = perf_counter()
        keys = sorted(p.name.replace(".json.gz", "") for p in (PAIRED_ROOT / "width_wide/train").glob("*.json.gz"))
        assert len(keys) == 42
        with ProcessPoolExecutor(max_workers=14) as pool:
            done = list(pool.map(eval_one_wide, keys))
        print(f"WIDE-K B=1000 RUN COMPLETE: {len(done)} folds in {perf_counter() - t0:.1f}s", flush=True)
    if cmd in ("aggregate", "all"):
        aggregate_wide()
