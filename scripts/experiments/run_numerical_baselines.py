"""Phase 3: Numerical benchmark extended to B=1..1000 across 10 unseen prospective + 13 author sources.

Runs strictly on scroot-server (`lab`) using exact frozen roles, ExactBook (10-fold 5-NN C kernel),
and runner.outer_metrics. Verifies 1e-12 parity on B=1..128 before extending all methods to B=1000.
"""
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from math import isqrt
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import gzip
import json
import os
import sys
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
PAIRED_ROOT = Path("/home/sunxie/paired-delivery-20261123.lCTbZvTQ")
NUM_ROOT = Path("/home/sunxie/clinical-prospective-20261123.cjUGun3X/numeric")

# Import paired-delivery engine FIRST to avoid circular import with numeric/experiment.py
sys.path.insert(0, str(PAIRED_ROOT))
from engine import ExactBook, Delivery, projected, require_lab, run_seed
from strong import FixedProvider
from local_study import runner
from exact_five import kernel

PEERS_DIR = PAIRED_ROOT / "runtime/fold-guidance-20261123/support/semantic-evolution-20261123"
sys.path.append(str(PEERS_DIR))
import peers

MAX_BUDGET = 1000
CHECKPOINTS = (8, 16, 32, 64, 128, 256, 500, 1000)
BANDS = {
    "b8_128": (8, 128),
    "b8_500": (8, 500),
    "b8_1000": (8, 1000),
    "b32": (32, 32),
    "b64": (64, 64),
    "b128": (128, 128),
    "b256": (256, 256),
    "b500": (500, 500),
    "b1000": (1000, 1000),
}


def canonical(v):
    return (json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def seed_num(*parts):
    return int.from_bytes(sha256(repr(parts).encode()).digest()[:4], "big")


def purchase_budget(s, rs, policy, budget):
    """Exact copy of projected.purchase parameterized by budget (B=1..128 byte-identical)."""
    source = projected.Project(s, rs, "mel_init")
    initial = list(dict.fromkeys(source.draw() for _ in range(20)))
    path, seen, elite, age, ops, turn = [], set(), [], {}, [], 0
    for i in range(budget):
        offer = None
        if i < len(initial):
            m = initial[i]
        else:
            opportunity = (
                policy == "dense"
                or (policy in ("half", "fixed") and (i - len(initial)) % 2 == 0)
                or (policy == "square" and isqrt(i) ** 2 == i)
            )
            if opportunity:
                h = turn % len(elite)
                parents = path[:1] if policy == "fixed" else elite[h:] + elite[:h]
                offer = s.mutate(parents, seen, "numeric")
                turn += 1
            m = offer[0] if offer is not None else source.unseen(seen)
            if m is None:
                m = next((v for v in s.menu if v not in seen), None)
        if m is None:
            break
        assert m not in seen
        seen.add(m)
        j = s.book.buy(m)
        path.append(j)
        age[j] = i
        ops.append("M" if offer else "R")
        keep, behavior = [], set()
        for u in sorted(elite + [j], key=lambda idx: (-s.book.scores[idx]["fold"], len(s.book.masks[idx]), age[idx])):
            v = s.book.oof[u].tobytes()
            if v not in behavior:
                keep.append(u)
                behavior.add(v)
            if len(keep) == 4:
                break
        elite = keep
    return dict(path=path, win=projected.make_state(s.book, path)["winners"], op="".join(ops), initial=len(initial))


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


def run_ga_num(x, y, rs, cap, budget):
    t0 = perf_counter()
    bk = ExactBook(x, y, rs)
    p = x.shape[1]
    rng = np.random.RandomState(rs + 101)
    pop_size = 20
    seen, path, times = set(), [], []

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
        times.append(perf_counter() - t0)
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
            times.append(perf_counter() - t0)
            new_pop.append(m2v(m))
            new_fit.append(bk.scores[j]["fold"])
        pop, fit = new_pop, new_fit
    return bk, path, winners_from_path(bk, path), times


def run_bpso_num(x, y, rs, cap, budget):
    t0 = perf_counter()
    bk = ExactBook(x, y, rs)
    p = x.shape[1]
    rng = np.random.RandomState(rs + 202)
    n_part = 20
    seen, path, times = set(), [], []
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
            times.append(perf_counter() - t0)
            f = bk.scores[j]["fold"]
            if f > pbest_fit[i]:
                pbest_fit[i] = f
                pbest[i] = v.copy()
            if f > gbest_fit:
                gbest_fit = f
                gbest = v.copy()
    return bk, path, winners_from_path(bk, path), times


def run_mrmr_num(x, y, cap):
    t0 = perf_counter()
    # Cap candidate pool to top-256 F-stat features when p > 256 for fast exact mRMR
    p = x.shape[1]
    if p > 256:
        classes = np.unique(y)
        mu = x.mean(axis=0)
        ssb = sum(np.sum(y == c) * (x[y == c].mean(axis=0) - mu) ** 2 for c in classes)
        ssw = sum(((x[y == c] - x[y == c].mean(axis=0)) ** 2).sum(axis=0) for c in classes) + 1e-12
        top_idx = np.argsort(-(ssb / ssw))[:256]
        sub_ord = peers.mrmr(x[:, top_idx], y)
        mask = tuple(sorted(map(int, top_idx[sub_ord[:cap]])))
    else:
        ord_idx = peers.mrmr(x, y)
        mask = tuple(sorted(map(int, ord_idx[:cap])))
    return mask, perf_counter() - t0


def run_l1_num(x, y, rs, cap):
    t0 = perf_counter()
    xs = StandardScaler().fit_transform(x)
    p = x.shape[1]
    bk = ExactBook(x, y, rs)
    best_sc = -1.0
    best_m = None
    for c_val in (0.1, 0.5, 1.0, 5.0):
        clf = LogisticRegression(penalty="l1", C=c_val, solver="liblinear",
                                 class_weight="balanced", max_iter=300, random_state=rs)
        clf.fit(xs, y)
        coef = np.abs(clf.coef_)
        imp = coef.max(axis=0) if coef.ndim == 2 else coef.ravel()
        ord_idx = np.lexsort((np.arange(p), -imp))
        mask = tuple(sorted(map(int, ord_idx[:cap])))
        idx = bk.buy(mask)
        sc = bk.scores[idx]["fold"]
        if sc > best_sc:
            best_sc = sc
            best_m = mask
    return best_m, perf_counter() - t0


def run_one_fold(job):
    suite, source_group, task_key, rep, x, y, q, v, rs, cap = job
    kernel()
    arms_raw = {}

    # 1. tree_union, numeric, iid up to MAX_BUDGET
    for arm_name, provider, policy in (
        ("tree_union", "tree_union", "half"),
        ("numeric", "numeric", "half"),
        ("iid", "numeric", "iid"),
    ):
        t0 = perf_counter()
        bk = ExactBook(x, y, rs)
        s = FixedProvider(x, y, rs, MAX_BUDGET, limit=cap, book=bk)
        s.provider = provider
        res = purchase_budget(s, rs, policy, MAX_BUDGET)
        dt = perf_counter() - t0
        times = [dt * (i + 1) / len(res["path"]) for i in range(len(res["path"]))]
        arms_raw[arm_name] = (bk, res["path"], res["win"], times)

    # 2. MEL up to MAX_BUDGET
    t0 = perf_counter()
    bk_m = ExactBook(x, y, rs)
    s_m = FixedProvider(x, y, rs, MAX_BUDGET, limit=cap, book=bk_m)
    prob = SimpleNamespace(x=s_m.x, y=s_m.y, seed=rs, book=bk_m,
                           base=SimpleNamespace(cap=cap, tree=s_m.tree, vote=s_m.menu))
    res_m = projected.mel_path(prob, MAX_BUDGET)
    dt_m = perf_counter() - t0
    times_m = [dt_m * (i + 1) / len(res_m["path"]) for i in range(len(res_m["path"]))]
    arms_raw["mel"] = (bk_m, res_m["path"], res_m["winners"], times_m)

    # 3. GA & BPSO up to MAX_BUDGET
    arms_raw["ga"] = run_ga_num(x, y, rs, cap, MAX_BUDGET)
    arms_raw["bpso"] = run_bpso_num(x, y, rs, cap, MAX_BUDGET)

    # 4. Static classical FS
    m_mrmr, dt_mrmr = run_mrmr_num(x, y, cap)
    bk_s = ExactBook(x, y, rs)
    i_mrmr = bk_s.buy(m_mrmr)
    arms_raw["mrmr_static"] = (bk_s, [i_mrmr] * MAX_BUDGET, [i_mrmr] * MAX_BUDGET, [dt_mrmr] * MAX_BUDGET)

    m_l1, dt_l1 = run_l1_num(x, y, rs, cap)
    i_l1 = bk_s.buy(m_l1)
    arms_raw["l1_lr_static"] = (bk_s, [i_l1] * MAX_BUDGET, [i_l1] * MAX_BUDGET, [dt_l1] * MAX_BUDGET)

    # Batch score ALL distinct outer masks in ONE C-kernel pass
    all_masks = list(dict.fromkeys(
        tuple(bk.masks[j])
        for bk, pth, win, _ in arms_raw.values()
        for j in set(pth) | set(win)
    ))
    outer_vals = runner.outer_metrics(x, y, q, v, all_masks)
    m2i = {m: i for i, m in enumerate(all_masks)}

    out_arms = {}
    for arm_name, (bk, pth, win, tms) in arms_raw.items():
        L = len(pth)
        # Pad if a tiny dataset exhausted all unique subsets before 1000
        if L < MAX_BUDGET:
            pth = pth + [pth[-1]] * (MAX_BUDGET - L)
            win = win + [win[-1]] * (MAX_BUDGET - L)
            tms = tms + [tms[-1]] * (MAX_BUDGET - L)
        p_idx = [m2i[tuple(bk.masks[j])] for j in pth[:MAX_BUDGET]]
        w_idx = [m2i[tuple(bk.masks[j])] for j in win[:MAX_BUDGET]]
        prec = outer_vals["precision"][w_idx]
        ba = outer_vals["ba"][w_idx]
        f1 = outer_vals["f1"][w_idx]
        ceil_prec = np.maximum.accumulate(outer_vals["precision"][p_idx])
        ceil_ba = np.maximum.accumulate(outer_vals["ba"][p_idx])
        inner_sc = np.array([bk.scores[j]["fold"] for j in win[:MAX_BUDGET]], dtype=float)
        out_arms[arm_name] = dict(
            precision=np.round(prec, 6).tolist(),
            ba=np.round(ba, 6).tolist(),
            f1=np.round(f1, 6).tolist(),
            ceil_precision=np.round(ceil_prec, 6).tolist(),
            ceil_ba=np.round(ceil_ba, 6).tolist(),
            inner=np.round(inner_sc, 6).tolist(),
            cpu_checkpoints={str(b): round(tms[b - 1], 6) for b in CHECKPOINTS},
            masks_checkpoints={str(b): list(bk.masks[win[b - 1]]) for b in CHECKPOINTS},
            actual_unique=L,
        )

    dst = ROOT / "raw_numerical" / f"{suite}__{task_key}__{rep}.json.gz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(suite=suite, group=source_group, task=task_key, rep=rep, p=int(x.shape[1]),
                   n_train=int(len(y)), n_outer=int(len(v)), cap=int(cap), arms=out_arms)
    tmp = dst.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(canonical(payload), compresslevel=4, mtime=0))
    tmp.replace(dst)
    return dst.name


def build_jobs():
    jobs = []
    # Suite 1: 10 strictly unseen prospective numerical sources (30 folds)
    roles_meta = json.loads((NUM_ROOT / "roles.json").read_text())
    for k in sorted(roles_meta.keys()):
        for r in range(3):
            tr_p = NUM_ROOT / "roles" / f"{k}-{r}-train.npz"
            ou_p = NUM_ROOT / "roles" / f"{k}-{r}-outer.npz"
            assert digest(tr_p) == roles_meta[k]["folds"][r]["train"]
            assert digest(ou_p) == roles_meta[k]["folds"][r]["outer"]
            with np.load(tr_p) as z:
                x, y = z["x"], z["y"]
            with np.load(ou_p) as z:
                q, v = z["x"], z["y"]
            rs = seed_num("evolution", k, r)
            jobs.append(("prospective_10", k, k, r, x, y, q, v, rs, 8))

    # Suite 2 & 3: 14 author tables (13 sources, cohort='expanded') + 15 transfer sources
    for key in runner.cases("replay"):
        a = runner.archive_for(key)
        c = a["case"]
        x, y = projected.arrays(c)
        q, v = projected.arrays(c, True)
        rs = run_seed(c)
        cap = min(8, max(1, int(0.15 * x.shape[1])))
        suite_name = "author_13" if c[0] == "expanded" else "transfer_15"
        jobs.append((suite_name, a["group"], c[1], int(c[3]), x, y, q, v, rs, cap))
    return jobs


def log_w(lo, hi):
    bs = np.arange(lo, hi + 1, dtype=float)
    w = np.log1p(1.0 / bs)
    return w / w.sum()


def aggregate():
    raw_dir = ROOT / "raw_numerical"
    res_dir = ROOT / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Organize by suite -> method -> group -> list of fold arrays (equal weight per source group)
    store = {}
    for fp in sorted(raw_dir.glob("*.json.gz")):
        rec = json.loads(gzip.decompress(fp.read_bytes()))
        suite = rec["suite"]
        grp = rec["group"]
        for m, payload in rec["arms"].items():
            (store.setdefault(suite, {})
             .setdefault(m, {})
             .setdefault(grp, []).append(payload))

    # 1. Bands summary (source-equal weighted)
    lines = ["suite\tband\tmethod\tprecision\tba\tf1\tceil_precision\tinner\tcpu_end"]
    for suite in sorted(store.keys()):
        methods = sorted(store[suite].keys())
        for bname, (lo, hi) in BANDS.items():
            w = log_w(lo, hi)
            for m in methods:
                gp, gb, gf, gc, gi, gt = [], [], [], [], [], []
                for grp, items in store[suite][m].items():
                    gp.append(np.mean([np.array(it["precision"][lo - 1:hi]) @ w for it in items]))
                    gb.append(np.mean([np.array(it["ba"][lo - 1:hi]) @ w for it in items]))
                    gf.append(np.mean([np.array(it["f1"][lo - 1:hi]) @ w for it in items]))
                    gc.append(np.mean([np.array(it["ceil_precision"][lo - 1:hi]) @ w for it in items]))
                    gi.append(np.mean([np.array(it["inner"][lo - 1:hi]) @ w for it in items]))
                    # Nearest checkpoint cpu
                    ck_key = str(hi) if str(hi) in items[0]["cpu_checkpoints"] else "1000"
                    gt.append(np.mean([it["cpu_checkpoints"][ck_key] for it in items]))
                lines.append(
                    f"{suite}\t{bname}\t{m}\t{np.mean(gp):.6f}\t{np.mean(gb):.6f}\t{np.mean(gf):.6f}\t"
                    f"{np.mean(gc):.6f}\t{np.mean(gi):.6f}\t{np.mean(gt):.6f}"
                )
    (res_dir / "numerical_b1000_bands.tsv").write_text("\n".join(lines) + "\n")

    # 2. Full budget curves (1..1000) for plotting pool-vs-delivery scissors
    c_lines = ["suite\tmethod\tbudget\tprecision\tba\tf1\tceil_precision\tinner"]
    for suite in sorted(store.keys()):
        for m in sorted(store[suite].keys()):
            grps = list(store[suite][m].values())
            pr = np.mean([np.mean([it["precision"] for it in items], axis=0) for items in grps], axis=0)
            ba = np.mean([np.mean([it["ba"] for it in items], axis=0) for items in grps], axis=0)
            f1 = np.mean([np.mean([it["f1"] for it in items], axis=0) for items in grps], axis=0)
            ce = np.mean([np.mean([it["ceil_precision"] for it in items], axis=0) for items in grps], axis=0)
            in_ = np.mean([np.mean([it["inner"] for it in items], axis=0) for items in grps], axis=0)
            for b in range(MAX_BUDGET):
                c_lines.append(f"{suite}\t{m}\t{b+1}\t{pr[b]:.6f}\t{ba[b]:.6f}\t{f1[b]:.6f}\t{ce[b]:.6f}\t{in_[b]:.6f}")
    (res_dir / "numerical_b1000_curves.tsv").write_text("\n".join(c_lines) + "\n")

    # 3. Per-source summary at B8-128, B128, B500, B1000
    s_lines = ["suite\tgroup\tmethod\tb8_128_prec\tb128_prec\tb500_prec\tb1000_prec\tb1000_ba\tb1000_f1\tcpu1000"]
    w128 = log_w(8, 128)
    for suite in sorted(store.keys()):
        for grp in sorted(store[suite]["tree_union"].keys()):
            for m in sorted(store[suite].keys()):
                items = store[suite][m][grp]
                p8128 = np.mean([np.array(it["precision"][7:128]) @ w128 for it in items])
                p128 = np.mean([it["precision"][127] for it in items])
                p500 = np.mean([it["precision"][499] for it in items])
                p1000 = np.mean([it["precision"][999] for it in items])
                b1000 = np.mean([it["ba"][999] for it in items])
                f1000 = np.mean([it["f1"][999] for it in items])
                t1000 = np.mean([it["cpu_checkpoints"]["1000"] for it in items])
                s_lines.append(f"{suite}\t{grp}\t{m}\t{p8128:.6f}\t{p128:.6f}\t{p500:.6f}\t{p1000:.6f}\t{b1000:.6f}\t{f1000:.6f}\t{t1000:.6f}")
    (res_dir / "numerical_b1000_per_source.tsv").write_text("\n".join(s_lines) + "\n")

    print("PHASE 3 NUMERICAL AGGREGATION COMPLETE.", flush=True)


if __name__ == "__main__":
    require_lab()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("run", "all"):
        t0 = perf_counter()
        jobs = build_jobs()
        print(f"Launching {len(jobs)} numerical folds up to B={MAX_BUDGET}...", flush=True)
        with ProcessPoolExecutor(max_workers=16) as pool:
            done = list(pool.map(run_one_fold, jobs))
        print(f"PHASE 3 RUN COMPLETE: {len(done)} folds in {perf_counter() - t0:.1f}s", flush=True)
    if cmd in ("aggregate", "all"):
        aggregate()
