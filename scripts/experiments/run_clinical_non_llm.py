"""Phase 1: Clinical 12-cohort non-LLM external baselines + extended B=1..128 trajectories.

Runs strictly on scroot-server (`lab`) using the exact frozen donor/outer splits,
SmallBook inner evaluator, margins outer evaluator, and Downstream LR/RF evaluator
from clinical-prospective-20261123 and clinical-extension-20261124.
Zero modifications to any frozen directory or selector logic.
"""
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import gzip
import json
import os
import sys
import warnings

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OLD_CLIN = Path("/home/sunxie/clinical-prospective-20261123.cjUGun3X")
NEW_CLIN = Path("/home/sunxie/clinical-extension-20261124")
PEERS_DIR = Path("/home/sunxie/paired-delivery-20261123.lCTbZvTQ/runtime/fold-guidance-20261123/support/semantic-evolution-20261123")

sys.path.insert(0, str(OLD_CLIN))
sys.path.insert(0, str(PEERS_DIR))

from panel_data import canonical, digest, read, require_lab, write
from legacy import SmallBook, Search, Recombination, cdf_rank, empirical, evolve, margins, schedule, seed, third_menu, tree_rank
from evaluation import Downstream, measures
import peers

EARLIER_SIX = ("brf", "cad", "gbsg", "thor", "thyroid", "voice")
NEW_SIX = ("whas", "colontrial", "actg", "ercp", "opttrial", "gargle")
ALL_TASKS = EARLIER_SIX + NEW_SIX

REPEATS = 20
NS = (3, 10, 20, 30)
CAP = 4
MAX_BUDGET = 128
CHECKPOINTS = (1, 2, 4, 8, 16, 32, 64, 128)
DOWN_BUDGETS = (2, 8, 16, 32, 64, 128)
BANDS = {
    "b2_16": (2, 16),
    "b2_32": (2, 32),
    "b2_64": (2, 64),
    "b2_128": (2, 128),
    "b1": (1, 1),
    "b2": (2, 2),
    "b8": (8, 8),
    "b16": (16, 16),
    "b32": (32, 32),
    "b64": (64, 64),
    "b128": (128, 128),
}


def task_root(t):
    return OLD_CLIN if t in EARLIER_SIX else NEW_CLIN


def load_role(t, part):
    base = task_root(t)
    manifest = read(base / "roles/manifest.json")
    fp = base / "roles" / f"{t}-{part}.npz"
    assert digest(fp) == manifest[t][part + "_sha"], f"SHA mismatch {fp}"
    with np.load(fp) as z:
        return z["x"], z["y"], z["ids"]


def get_donors(t, r, n):
    x, y, ids = load_role(t, "train")
    rng = np.random.RandomState(seed("clinical-donor-20261123", t, r))
    selected = np.asarray([rng.permutation(np.flatnonzero(y == c))[:n] for c in (0, 1)]).T.ravel()
    assert len(set(ids[selected])) == 2 * n
    return np.ascontiguousarray(x[selected]), y[selected], ids[selected]


def load_gemini_orders(t):
    base = task_root(t)
    schema = read(base / "metadata" / f"{t}.json")
    p = len(schema["fields"])
    orders = {}
    for kind in ("correct", "wrong"):
        card = read(base / "cards" / t / kind / "card.json")
        assert card["model"] == "gemini-3.8-flash"
        ord_list = list(map(int, card["order"]))
        orders[kind] = np.array(ord_list + [j for j in range(p) if j not in set(ord_list)], dtype=int)
    return orders, p


def delivery_any(book, st, anchor, tau=1.0):
    """Exact copy of training.delivery without the <=32 budget assertion (tau=1.0 is frozen default)."""
    path = st["path"]
    raw = st["win"]
    assert len(path) == len(set(path)) == len(raw) and len(path) > 0
    if anchor is None:
        return dict(raw=raw, paired=raw, tied=raw, anchor=None)
    aset = set(anchor)
    anchor_id = book.ids.get(anchor)
    age = {j: i for i, j in enumerate(path)}
    champion = None
    paired, tied = [], []
    best_key = None
    tau2 = float(tau) ** 2
    for i, j in enumerate(path):
        correct = book.oof[j] == book.y
        key = (-int(correct.sum()), len(aset.symmetric_difference(book.masks[j])), len(book.masks[j]), i)
        if best_key is None or key < best_key:
            champion, best_key = j, key
        if anchor_id not in age or i < age[anchor_id]:
            paired.append(raw[i])
            tied.append(raw[i])
            continue
        new = book.oof[champion] == book.y
        ref = book.oof[anchor_id] == book.y
        a, b = int((new & ~ref).sum()), int((ref & ~new).sum())
        paired.append(champion if a > b and (a - b) ** 2 > tau2 * (a + b) else anchor_id)
        tied.append(champion)
    return dict(raw=raw, paired=paired, tied=tied, anchor=anchor_id)


def winners_from_path(book, path):
    win = []
    best = path[0]
    for j in path:
        best = min((best, j), key=lambda idx: (-book.scores[idx]["fold"], len(book.masks[idx])))
        win.append(best)
    return win


# ---------------- Standard Z-score preprocessing & Static classical FS baselines ----------------

def prep_std(x):
    """Standard median imputation + Z-score standardization on training split only."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for j in range(x.shape[1]):
        col = x[:, j]
        ok = np.isfinite(col)
        med = float(np.median(col[ok])) if ok.any() else 0.0
        c = np.where(ok, col, med)
        sd = float(np.std(c))
        out[:, j] = (c - float(np.mean(c))) / (sd if sd > 1e-12 else 1.0)
    return out


def run_mrmr_static(x_std, y, cap):
    order = peers.mrmr(x_std, y)
    return tuple(sorted(map(int, order[:cap])))


def run_relieff_static(x_std, y, cap):
    n_samples, p = x_std.shape
    k = max(1, min(int(np.sum(y == 0)) - 1, int(np.sum(y == 1)) - 1, 5))
    weights = np.zeros(p, dtype=float)
    dists = np.abs(x_std[:, None, :] - x_std[None, :, :]).sum(axis=2)
    for i in range(n_samples):
        same = np.flatnonzero((y == y[i]) & (np.arange(n_samples) != i))
        diff = np.flatnonzero(y != y[i])
        nh = same[np.argsort(dists[i, same])[:k]]
        nm = diff[np.argsort(dists[i, diff])[:k]]
        weights -= np.abs(x_std[i] - x_std[nh]).mean(axis=0)
        weights += np.abs(x_std[i] - x_std[nm]).mean(axis=0)
    order = np.lexsort((np.arange(p), -weights))
    return tuple(sorted(map(int, order[:cap])))


def run_l1_lr_static(x_std, y, cap, rs):
    """Faithful native L1 logistic regression with 3-fold stratified CV on Z-scored features."""
    p = x_std.shape[1]
    c_grid = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=rs)
    best_sc, best_c = -1.0, 1.0
    for c_val in c_grid:
        scs = []
        for tr, va in skf.split(x_std, y):
            clf = LogisticRegression(penalty="l1", C=c_val, solver="liblinear",
                                     class_weight="balanced", max_iter=500, random_state=rs)
            clf.fit(x_std[tr], y[tr])
            pr = clf.predict(x_std[va])
            b0 = float(np.mean(pr[y[va] == 0] == 0))
            b1 = float(np.mean(pr[y[va] == 1] == 1))
            scs.append(0.5 * (b0 + b1))
        sc = float(np.mean(scs))
        if sc > best_sc:
            best_sc, best_c = sc, c_val
    clf = LogisticRegression(penalty="l1", C=best_c, solver="liblinear",
                             class_weight="balanced", max_iter=500, random_state=rs)
    clf.fit(x_std, y)
    coef = np.abs(clf.coef_.ravel())
    nz = np.flatnonzero(coef > 1e-8)
    if len(nz) > 0:
        order = nz[np.argsort(-coef[nz])]
        return tuple(sorted(map(int, order[:cap])))
    corr = np.abs(x_std[y == 1].mean(axis=0) - x_std[y == 0].mean(axis=0))
    order = np.lexsort((np.arange(p), -corr))
    return tuple(sorted(map(int, order[:cap])))


def run_rf_imp_static(x_std, y, cap, rs):
    p = x_std.shape[1]
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=rs, n_jobs=1)
    clf.fit(x_std, y)
    imp = clf.feature_importances_
    corr = np.abs(x_std[y == 1].mean(axis=0) - x_std[y == 0].mean(axis=0))
    order = np.lexsort((np.arange(p), -corr, -imp))
    return tuple(sorted(map(int, order[:cap])))


# ---------------- Wrapper EFS & BO baselines (B = 1..128) ----------------

def repair_mask(vec, scores, cap, rng):
    p = len(vec)
    ones = np.flatnonzero(vec > 0)
    if len(ones) == 0:
        if scores is not None:
            j = int(np.argmax(scores + rng.uniform(0, 1e-9, size=p)))
        else:
            j = int(rng.randint(0, p))
        return (j,)
    if len(ones) > cap:
        if scores is not None:
            sub = scores[ones] + rng.uniform(0, 1e-9, size=len(ones))
            chosen = ones[np.argsort(-sub)[:cap]]
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
            avail = [j for j in range(p) if j not in cand]
            if avail:
                cand.add(int(avail[rng.randint(len(avail))]))
        elif op == 1 and len(cand) > 1:
            cand.remove(int(sorted(cand)[rng.randint(len(cand))]))
        else:
            avail = [j for j in range(p) if j not in cand]
            if avail and cand:
                cand.remove(int(sorted(cand)[rng.randint(len(cand))]))
                cand.add(int(avail[rng.randint(len(avail))]))
        t = tuple(sorted(cand))
        if t and t not in seen:
            return t
    # Guaranteed unseen fallback
    for _ in range(256):
        k = int(rng.randint(1, min(cap, p) + 1))
        t = tuple(sorted(map(int, rng.choice(p, size=k, replace=False))))
        if t not in seen:
            return t
    # Exhaustive fallback for very small p
    from itertools import combinations
    for k in range(1, min(cap, p) + 1):
        for comb in combinations(range(p), k):
            if comb not in seen:
                return comb
    return mask


def eval_mask_timed(book, mask, path, times, t0):
    j = book.buy(mask)
    path.append(j)
    times.append(perf_counter() - t0)
    return book.scores[j]["fold"], j


def run_ga(book, p, cap, budget, rs):
    t0 = perf_counter()
    rng = np.random.RandomState(rs + 101)
    pop_size = 16
    seen = set()
    path, times = [], []

    def mask_to_vec(m):
        v = np.zeros(p, dtype=np.int8)
        v[list(m)] = 1
        return v

    # Initial population
    pop, fit = [], []
    for _ in range(pop_size):
        if len(path) >= budget:
            break
        k = int(rng.randint(1, min(cap, p) + 1))
        m = tuple(sorted(map(int, rng.choice(p, size=k, replace=False))))
        m = dedup_mask(m, seen, p, cap, rng)
        seen.add(m)
        f, _ = eval_mask_timed(book, m, path, times, t0)
        pop.append(mask_to_vec(m))
        fit.append(f)

    pm = max(1.0 / p, 0.05)
    while len(path) < budget:
        new_pop, new_fit = [], []
        # Elitism: carry top 2 without re-evaluating
        order = np.argsort(-np.array(fit))
        for idx in order[:2]:
            new_pop.append(pop[idx].copy())
            new_fit.append(fit[idx])
        while len(new_pop) < pop_size and len(path) < budget:
            # Binary tournament
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
            m = repair_mask(c1, None, cap, rng)
            m = dedup_mask(m, seen, p, cap, rng)
            seen.add(m)
            f, _ = eval_mask_timed(book, m, path, times, t0)
            new_pop.append(mask_to_vec(m))
            new_fit.append(f)
        pop, fit = new_pop, new_fit
    return path, winners_from_path(book, path), times


def run_bpso(book, p, cap, budget, rs):
    t0 = perf_counter()
    rng = np.random.RandomState(rs + 202)
    n_part = 16
    seen = set()
    path, times = [], []

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
                vel[i] = w * vel[i] + 1.5 * r1 * (pbest[i] - pos[i]) + 1.5 * r2 * (gbest - pos[i])
                vel[i] = np.clip(vel[i], -4.0, 4.0)
                probs = 1.0 / (1.0 + np.exp(-vel[i]))
                cand = (rng.rand(p) < probs).astype(np.int8)
                m = repair_mask(cand, probs, cap, rng)
            m = dedup_mask(m, seen, p, cap, rng)
            seen.add(m)
            v = np.zeros(p, dtype=np.int8)
            v[list(m)] = 1
            pos[i] = v
            f, _ = eval_mask_timed(book, m, path, times, t0)
            if f > pbest_fit[i]:
                pbest_fit[i] = f
                pbest[i] = v.copy()
            if f > gbest_fit:
                gbest_fit = f
                gbest = v.copy()
    return path, winners_from_path(book, path), times


def run_eda(book, p, cap, budget, rs):
    t0 = perf_counter()
    rng = np.random.RandomState(rs + 303)
    pop_size = 16
    seen = set()
    path, times = [], []
    probs = np.full(p, min(0.5, cap / float(p)), dtype=float)

    while len(path) < budget:
        batch_vecs, batch_fits = [], []
        for _ in range(pop_size):
            if len(path) >= budget:
                break
            cand = (rng.rand(p) < probs).astype(np.int8)
            m = repair_mask(cand, probs, cap, rng)
            m = dedup_mask(m, seen, p, cap, rng)
            seen.add(m)
            v = np.zeros(p, dtype=np.int8)
            v[list(m)] = 1
            f, _ = eval_mask_timed(book, m, path, times, t0)
            batch_vecs.append(v)
            batch_fits.append(f)
        if len(batch_vecs) >= 4:
            elite_n = max(2, len(batch_vecs) // 2)
            idx = np.argsort(-np.array(batch_fits))[:elite_n]
            elite = np.array([batch_vecs[k] for k in idx], dtype=float)
            probs = np.clip(0.5 * probs + 0.5 * elite.mean(axis=0), 0.05, 0.95)
    return path, winners_from_path(book, path), times


def run_mel(book, x_emp, y, tree_ord, menu, cap, budget, rs):
    t0 = perf_counter()
    path_orig_buy = book.buy
    times = []
    seen_len = len(book.masks)

    class TimedWrapper:
        def __init__(self, inner):
            self._inner = inner
            self.masks = inner.masks
            self.scores = inner.scores
            self.oof = inner.oof
            self.ids = inner.ids
        def buy(self, mask):
            before = len(self._inner.masks)
            idx = self._inner.buy(mask)
            if len(self._inner.masks) > before:
                times.append(perf_counter() - t0)
            return idx

    # Use an isolated book so MEL's cache matches its own unique evaluations cleanly
    local_book = SmallBook(book.raw_x, book.y)
    tw = TimedWrapper(local_book)
    prob = SimpleNamespace(x=x_emp, y=y, seed=rs, book=tw, base=SimpleNamespace(cap=cap, tree=tree_ord, vote=menu))
    res = peers.mel_path(prob, budget)
    # Map masks into shared book
    path = [book.buy(local_book.masks[j]) for j in res["path"]]
    if len(times) < len(path):
        dt = perf_counter() - t0
        times.extend([dt] * (len(path) - len(times)))
    return path[:budget], winners_from_path(book, path[:budget]), times[:budget]


def run_pibo(book, p, cap, budget, order, rs):
    """Prior-guided Bayesian Optimization (Hvarfner et al., ICLR 2022) over binary masks."""
    t0 = perf_counter()
    rng = np.random.RandomState(rs + 404)
    seen = set()
    path, times = [], []

    # Build smooth marginal prior from frozen LLM ranking
    marg = np.full(p, 0.05, dtype=float)
    for rank_idx, feat_id in enumerate(order):
        if rank_idx < cap:
            marg[feat_id] = 0.80 - 0.05 * rank_idx
        elif rank_idx < 2 * cap:
            marg[feat_id] = 0.45 - 0.04 * (rank_idx - cap)
        else:
            marg[feat_id] = max(0.02, 0.15 * np.exp(-0.15 * (rank_idx - 2 * cap)))
    marg = np.clip(marg, 0.02, 0.95)
    log_m = np.log(marg)
    log_1m = np.log(1.0 - marg)

    def mask_to_vec(m):
        v = np.zeros(p, dtype=np.float32)
        v[list(m)] = 1.0
        return v

    def prior_prob(vec):
        # Normalized geometric mean probability per coordinate to avoid dim collapse on p=753
        lp = (vec * log_m + (1.0 - vec) * log_1m).mean()
        return float(np.exp(lp))

    X_obs, y_obs = [], []
    # Faithful PiBO (Hvarfner et al., ICLR 2022): 4 initial random draws from the prior marginal
    for _ in range(4):
        if len(path) >= budget:
            break
        cand = (rng.rand(p) < marg).astype(np.int8)
        m = repair_mask(cand, marg, cap, rng)
        m = dedup_mask(m, seen, p, cap, rng)
        seen.add(m)
        f, _ = eval_mask_timed(book, m, path, times, t0)
        X_obs.append(mask_to_vec(m))
        y_obs.append(f)

    beta = 10.0
    n_cand = 96
    while len(path) < budget:
        t = len(path)
        X_arr = np.asarray(X_obs, dtype=np.float32)
        y_arr = np.asarray(y_obs, dtype=float)
        best_y = float(y_arr.max())
        best_vec = X_obs[int(np.argmax(y_arr))]
        best_mask = tuple(np.flatnonzero(best_vec > 0.5).tolist())

        # Fit RF surrogate
        rf = RandomForestRegressor(n_estimators=32, max_depth=6, random_state=rs + t, n_jobs=1)
        rf.fit(X_arr, y_arr)

        # Generate diverse candidate pool
        cands = []
        cand_seen = set(seen)
        # 1) 48 prior draws
        for _ in range(n_cand // 2):
            v = (rng.rand(p) < marg).astype(np.int8)
            m = repair_mask(v, marg, cap, rng)
            if m not in cand_seen:
                cand_seen.add(m)
                cands.append(m)
        # 2) 24 local neighbors of current best
        for _ in range(n_cand // 4):
            m = dedup_mask(best_mask, cand_seen, p, cap, rng)
            if m not in cand_seen:
                cand_seen.add(m)
                cands.append(m)
        # 3) 24 uniform feasible
        for _ in range(n_cand // 4):
            k = int(rng.randint(1, min(cap, p) + 1))
            m = tuple(sorted(map(int, rng.choice(p, size=k, replace=False))))
            if m not in cand_seen:
                cand_seen.add(m)
                cands.append(m)
        if not cands:
            m = dedup_mask(best_mask, seen, p, cap, rng)
            cands.append(m)

        C_mat = np.asarray([mask_to_vec(m) for m in cands], dtype=np.float32)
        tree_preds = np.asarray([est.predict(C_mat) for est in rf.estimators_], dtype=float)
        mu = tree_preds.mean(axis=0)
        sigma = tree_preds.std(axis=0) + 1e-6
        z = (mu - best_y) / sigma
        ei = (mu - best_y) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 1e-9)
        pi_vals = np.array([prior_prob(C_mat[idx]) for idx in range(len(cands))], dtype=float)
        acq = ei * (pi_vals ** (beta / float(t)))
        chosen = cands[int(np.argmax(acq))]
        seen.add(chosen)
        f, _ = eval_mask_timed(book, chosen, path, times, t0)
        X_obs.append(mask_to_vec(chosen))
        y_obs.append(f)

    return path, winners_from_path(book, path), times


def run_frozen_ext(x, y, cap, budget, order, rs):
    """Run exact frozen clinical Search+evolve+delivery up to `budget` (B=1..32 byte-identical)."""
    t0 = perf_counter()
    bk = SmallBook(x, y)
    rks = tree_rank(bk.x, bk.y, rs), cdf_rank(bk.x, bk.y)
    s = Search(bk.x, bk.y, rs, budget, limit=cap, book=bk, ranks=rks)
    base, fam = schedule(s)
    menu, fam, _ = third_menu(base, fam, order, cap)
    st = evolve(s, "handoff_mut", menu, fam, Recombination(s))
    anc = None if order is None else tuple(sorted(map(int, order[:cap])))
    ret = delivery_any(bk, st, anc)
    dt = perf_counter() - t0
    times = [dt * (i + 1) / len(st["path"]) for i in range(len(st["path"]))]
    return bk, st, ret, times


# ---------------- Worker per (task, repeat) ----------------

def run_task_repeat(job):
    t, r = job
    q_outer, v_outer, _ = load_role(t, "outer")
    orders, p = load_gemini_orders(t)
    records = []

    for n in NS:
        x, y, ids = get_donors(t, r, n)
        rs = seed("clinical-EA-20261123", t, r, n)

        # Shared book for external baselines
        book = SmallBook(x, y)
        book.raw_x = x
        x_emp = book.x
        rks = tree_rank(x_emp, y, rs), cdf_rank(x_emp, y)
        s_ref = Search(x_emp, y, rs, MAX_BUDGET, limit=CAP, book=SmallBook(x, y), ranks=rks)

        # Precompute outer margins for fast evaluation of any mask
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
        x_std = prep_std(x)

        methods_out = {}

        def pack_trace(name, local_book, path, win_list, times, static_mask=None):
            if static_mask is not None:
                idx = local_book.buy(static_mask)
                path = [idx] * MAX_BUDGET
                win_list = [idx] * MAX_BUDGET
                times = [times[0]] * MAX_BUDGET
            assert len(path) == MAX_BUDGET and len(win_list) == MAX_BUDGET
            ba_arr, prec_arr, f1_arr = [], [], []
            inner_arr, ceil_arr, sz_arr = [], [], []
            best_ceil = -1.0
            for i in range(MAX_BUDGET):
                pm = local_book.masks[path[i]]
                p_ba = score_outer(pm)[0]
                if p_ba > best_ceil:
                    best_ceil = p_ba
                wm = local_book.masks[win_list[i]]
                w_ba, w_pr, w_f1 = score_outer(wm)
                ba_arr.append(round(w_ba, 6))
                prec_arr.append(round(w_pr, 6))
                f1_arr.append(round(w_f1, 6))
                inner_arr.append(round(local_book.scores[win_list[i]]["fold"], 6))
                ceil_arr.append(round(best_ceil, 6))
                sz_arr.append(len(wm))
            ck_masks = {str(b): list(local_book.masks[win_list[b - 1]]) for b in CHECKPOINTS}
            down_scores = {}
            if down is not None:
                for b in DOWN_BUDGETS:
                    wm = local_book.masks[win_list[b - 1]]
                    down_scores[str(b)] = down.scores(wm)
            methods_out[name] = dict(
                ba=ba_arr, precision=prec_arr, f1=f1_arr,
                inner=inner_arr, ceiling=ceil_arr, size=sz_arr,
                cpu=[round(v, 6) for v in times],
                masks=ck_masks, down=down_scores,
            )

        # 1. Static classical FS (standard Z-scored features, native 3-fold CV for L1-LR)
        t0 = perf_counter(); m_mrmr = run_mrmr_static(x_std, y, CAP); dt = perf_counter() - t0
        pack_trace("mrmr_static", book, None, None, [dt], static_mask=m_mrmr)

        t0 = perf_counter(); m_rel = run_relieff_static(x_std, y, CAP); dt = perf_counter() - t0
        pack_trace("relieff_static", book, None, None, [dt], static_mask=m_rel)

        t0 = perf_counter(); m_l1 = run_l1_lr_static(x_std, y, CAP, rs); dt = perf_counter() - t0
        pack_trace("l1_lr_static", book, None, None, [dt], static_mask=m_l1)

        t0 = perf_counter(); m_rf = run_rf_imp_static(x_std, y, CAP, rs); dt = perf_counter() - t0
        pack_trace("rf_imp_static", book, None, None, [dt], static_mask=m_rf)

        # 2. Wrapper EFS & PiBO up to B=128
        for name, fn in (
            ("ga", lambda: run_ga(book, p, CAP, MAX_BUDGET, rs)),
            ("bpso", lambda: run_bpso(book, p, CAP, MAX_BUDGET, rs)),
            ("eda", lambda: run_eda(book, p, CAP, MAX_BUDGET, rs)),
            ("mel", lambda: run_mel(book, x_emp, y, s_ref.tree, s_ref.menu, CAP, MAX_BUDGET, rs)),
            ("pibo_correct", lambda: run_pibo(book, p, CAP, MAX_BUDGET, orders["correct"], rs)),
            ("pibo_wrong", lambda: run_pibo(book, p, CAP, MAX_BUDGET, orders["wrong"], rs)),
        ):
            pth, win, tms = fn()
            pack_trace(name, book, pth, win, tms)

        # 3. Extended B=1..128 trajectories of our frozen algorithm
        bk_num, st_num, ret_num, tms_num = run_frozen_ext(x, y, CAP, MAX_BUDGET, None, rs)
        pack_trace("numeric_ext", bk_num, st_num["path"], ret_num["raw"], tms_num)

        bk_gc, st_gc, ret_gc, tms_gc = run_frozen_ext(x, y, CAP, MAX_BUDGET, orders["correct"], rs)
        pack_trace("gemini_paired_ext", bk_gc, st_gc["path"], ret_gc["paired"], tms_gc)
        pack_trace("gemini_tied_ext", bk_gc, st_gc["path"], ret_gc["tied"], tms_gc)
        pack_trace("gemini_raw_ext", bk_gc, st_gc["path"], ret_gc["raw"], tms_gc)
        anc_gc = tuple(sorted(map(int, orders["correct"][:CAP])))
        pack_trace("gemini_paired_tau15", bk_gc, st_gc["path"], delivery_any(bk_gc, st_gc, anc_gc, tau=1.5)["paired"], tms_gc)
        pack_trace("gemini_paired_tau173", bk_gc, st_gc["path"], delivery_any(bk_gc, st_gc, anc_gc, tau=1.7320508)["paired"], tms_gc)
        pack_trace("gemini_static", bk_gc, None, None, [0.0], static_mask=anc_gc)

        bk_gw, st_gw, ret_gw, tms_gw = run_frozen_ext(x, y, CAP, MAX_BUDGET, orders["wrong"], rs)
        pack_trace("wrong_paired_ext", bk_gw, st_gw["path"], ret_gw["paired"], tms_gw)
        pack_trace("wrong_tied_ext", bk_gw, st_gw["path"], ret_gw["tied"], tms_gw)
        pack_trace("wrong_raw_ext", bk_gw, st_gw["path"], ret_gw["raw"], tms_gw)
        anc_gw = tuple(sorted(map(int, orders["wrong"][:CAP])))
        pack_trace("wrong_paired_tau15", bk_gw, st_gw["path"], delivery_any(bk_gw, st_gw, anc_gw, tau=1.5)["paired"], tms_gw)
        pack_trace("wrong_paired_tau173", bk_gw, st_gw["path"], delivery_any(bk_gw, st_gw, anc_gw, tau=1.7320508)["paired"], tms_gw)
        pack_trace("wrong_static", bk_gw, None, None, [0.0], static_mask=anc_gw)

        records.append(dict(t=t, r=r, n=n, k=CAP, p=p, ids=ids.tolist(), methods=methods_out))

    out_path = ROOT / "raw_clinical_non_llm" / f"{t}-{r:02d}.json.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(canonical(records), compresslevel=4, mtime=0))
    tmp.replace(out_path)
    return out_path.name


def verify_parity():
    """Verify exact 1e-12 parity with frozen training/external files on both waves."""
    checked = 0
    for t in ("gbsg", "cad", "whas", "ercp"):
        base = task_root(t)
        train_recs = json.loads(gzip.decompress((base / "training" / f"{t}-00.json.gz").read_bytes()))
        ext_recs = json.loads(gzip.decompress((base / "external" / f"{t}-00.json.gz").read_bytes()))["records"]
        orders, _ = load_gemini_orders(t)
        q_outer, v_outer, _ = load_role(t, "outer")
        for tr_r, ex_r in zip(train_recs, ext_recs):
            if tr_r["n"] not in NS or tr_r["k"] != CAP:
                continue
            n = tr_r["n"]
            x, y, ids = get_donors(t, 0, n)
            assert ids.tolist() == tr_r["ids"]
            rs = seed("clinical-EA-20261123", t, 0, n)
            d_outer = margins(x, y, q_outer)
            for arm_name, ord_vec in (("numeric", None), ("gemini", orders["correct"]), ("wrong", orders["wrong"])):
                bk, st, ret, _ = run_frozen_ext(x, y, CAP, MAX_BUDGET, ord_vec, rs)
                old_arm = tr_r["arms"][arm_name]
                old_masks = [tuple(m) for m in tr_r["masks"]]
                for b in range(32):
                    m_new = bk.masks[ret["paired"][b]]
                    m_old = old_masks[old_arm["paired"][b]]
                    assert m_new == m_old, f"Mask mismatch {t} n={n} {arm_name} b={b+1}: {m_new} vs {m_old}"
                    ba_new = measures(d_outer[:, m_new].sum(1) > 0, v_outer)[0]
                    ba_old = ex_r["metrics"][old_arm["paired"][b]][0]
                    assert abs(ba_new - ba_old) < 1e-12, f"BA mismatch {ba_new} vs {ba_old}"
                checked += 1
    out = dict(status="PASS", checked_trajectories=checked, max_budget=MAX_BUDGET, tol=1e-12)
    write(ROOT / "results/parity_check.json", out)
    print("PARITY CHECK PASS:", out, flush=True)


def log_weights(lo, hi):
    bs = np.arange(lo, hi + 1, dtype=float)
    w = 1.0 / bs
    return w / w.sum()


def aggregate():
    raw_dir = ROOT / "raw_clinical_non_llm"
    res_dir = ROOT / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Load all records: data[n][method][task] = list of 20 arrays of shape (128,)
    ba_data = {n: {} for n in NS}
    ceil_data = {n: {} for n in NS}
    inner_data = {n: {} for n in NS}
    cpu_data = {n: {} for n in NS}
    down_data = {}  # down_data[method][b][model][task] = list of 20 BA

    for t in ALL_TASKS:
        for r in range(REPEATS):
            recs = json.loads(gzip.decompress((raw_dir / f"{t}-{r:02d}.json.gz").read_bytes()))
            for rec in recs:
                n = rec["n"]
                for m, payload in rec["methods"].items():
                    ba_data[n].setdefault(m, {}).setdefault(t, []).append(np.array(payload["ba"], dtype=float))
                    ceil_data[n].setdefault(m, {}).setdefault(t, []).append(np.array(payload["ceiling"], dtype=float))
                    inner_data[n].setdefault(m, {}).setdefault(t, []).append(np.array(payload["inner"], dtype=float))
                    cpu_data[n].setdefault(m, {}).setdefault(t, []).append(np.array(payload["cpu"], dtype=float))
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

    # 1. means_bands.tsv
    lines = ["panel\tn\tband\tmethod\tba\tceiling\tinner\tcpu_end"]
    for pname, tasks in panels.items():
        for n in NS:
            for bname, (lo, hi) in BANDS.items():
                w = log_weights(lo, hi)
                for m in methods:
                    s_ba, s_ce, s_in, s_cp = [], [], [], []
                    for t in tasks:
                        arr_ba = np.asarray(ba_data[n][m][t])[:, lo - 1:hi]
                        arr_ce = np.asarray(ceil_data[n][m][t])[:, lo - 1:hi]
                        arr_in = np.asarray(inner_data[n][m][t])[:, lo - 1:hi]
                        arr_cp = np.asarray(cpu_data[n][m][t])[:, hi - 1]
                        s_ba.append(float((arr_ba @ w).mean()))
                        s_ce.append(float((arr_ce @ w).mean()))
                        s_in.append(float((arr_in @ w).mean()))
                        s_cp.append(float(arr_cp.mean()))
                    lines.append(f"{pname}\t{n}\t{bname}\t{m}\t{np.mean(s_ba):.6f}\t{np.mean(s_ce):.6f}\t{np.mean(s_in):.6f}\t{np.mean(s_cp):.6f}")
    (res_dir / "clinical_non_llm_bands.tsv").write_text("\n".join(lines) + "\n")

    # 2. per_source_checkpoints.tsv (n=3)
    lines = ["panel\ttask\tmethod\tb2_16\tb2\tb8\tb16\tb32\tb64\tb128\tceil128\tcpu128"]
    w216 = log_weights(2, 16)
    for t in ALL_TASKS:
        p_lbl = "earlier_six" if t in EARLIER_SIX else "new_six"
        for m in methods:
            arr = np.asarray(ba_data[3][m][t])
            ce = np.asarray(ceil_data[3][m][t])[:, 127].mean()
            cp = np.asarray(cpu_data[3][m][t])[:, 127].mean()
            v216 = float((arr[:, 1:16] @ w216).mean())
            vals = [float(arr[:, b - 1].mean()) for b in (2, 8, 16, 32, 64, 128)]
            lines.append(f"{p_lbl}\t{t}\t{m}\t{v216:.6f}\t" + "\t".join(f"{v:.6f}" for v in vals) + f"\t{ce:.6f}\t{cp:.6f}")
    (res_dir / "clinical_non_llm_per_source_n3.tsv").write_text("\n".join(lines) + "\n")

    # 3. budget_curves.tsv (full 1..128 for plotting)
    lines = ["panel\tn\tmethod\tbudget\tba\tceiling\tinner"]
    for pname, tasks in panels.items():
        for n in NS:
            for m in methods:
                curve_ba = np.mean([np.mean(ba_data[n][m][t], axis=0) for t in tasks], axis=0)
                curve_ce = np.mean([np.mean(ceil_data[n][m][t], axis=0) for t in tasks], axis=0)
                curve_in = np.mean([np.mean(inner_data[n][m][t], axis=0) for t in tasks], axis=0)
                for b in range(MAX_BUDGET):
                    lines.append(f"{pname}\t{n}\t{m}\t{b+1}\t{curve_ba[b]:.6f}\t{curve_ce[b]:.6f}\t{curve_in[b]:.6f}")
    (res_dir / "clinical_non_llm_curves.tsv").write_text("\n".join(lines) + "\n")

    # 4. downstream_n10.tsv (backward compatible) and downstream_all.tsv (n=10, 20, 30)
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
    (res_dir / "clinical_non_llm_downstream_n10.tsv").write_text("\n".join(lines10) + "\n")
    (res_dir / "clinical_non_llm_downstream_all.tsv").write_text("\n".join(lines_all) + "\n")

    # Seal all deliverables
    seals = {}
    for fp in sorted(raw_dir.glob("*.json.gz")) + sorted(res_dir.glob("*.tsv")):
        seals[str(fp.relative_to(ROOT))] = digest(fp)
    write(res_dir / "phase1_seal.json", dict(files=seals, count=len(seals)))
    print("AGGREGATION COMPLETE. Sealed files:", len(seals), flush=True)


if __name__ == "__main__":
    require_lab()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("verify", "all"):
        verify_parity()
    if cmd in ("run", "all"):
        t0 = perf_counter()
        jobs = [(t, r) for r in range(REPEATS) for t in ALL_TASKS]
        with ProcessPoolExecutor(max_workers=20) as pool:
            done = list(pool.map(run_task_repeat, jobs))
        assert len(set(done)) == len(ALL_TASKS) * REPEATS
        print(f"RUN COMPLETE: {len(done)} files in {perf_counter() - t0:.1f}s", flush=True)
    if cmd in ("aggregate", "all"):
        aggregate()
