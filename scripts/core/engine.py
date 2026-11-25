"""Train-side policies, evaluation receipts and shared source definitions."""

from collections import Counter
import gzip
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parent
RUNTIME = Path(os.environ.get("FS_REPLAY_RUNTIME", ROOT / "runtime"))
PREFIX = RUNTIME / "prefix-handoff-20261123"
os.environ["WIDTH_SEM_INPUT"] = str(RUNTIME / "fold-guidance-20261123/support/semantic-evolution-20261123")
sys.path[:0] = [str(PREFIX), str(RUNTIME / "capped-search-20261122/deps")]
import projected
from core import seed, signature
from handoff import Search, schedule, rotated
from numerical_search import initial_widths, metrics, confusion
from selector import ExactBook


def require_lab():
    if socket.gethostname().split(".")[0] != "scroot-server":
        raise RuntimeError("Use SSH lab for evaluation.")


def read(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    return json.loads(path.read_bytes())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if path.suffix == ".gz":
        data = gzip.compress(data, compresslevel=3, mtime=0)
    tmp = path.with_name("." + path.name)
    tmp.write_bytes(data)
    tmp.replace(path)


def run_seed(case):
    tag = {"expanded": "trial", "signal": "signal-EA", "med": "width-transfer-EA",
           "new": "handoff-transfer-20261123", "new2": "projected-new"}[case[0]]
    return seed(tag, case[1], case[3])


def dominance(book, candidate):
    return -book.scores[candidate]["fold"], len(book.masks[candidate])


def replacement(fold_differences):
    return bool(fold_differences.mean() > fold_differences.std(ddof=1) / np.sqrt(len(fold_differences)) + 1e-12)


class Delivery:
    def __init__(self, book):
        self.book = book
        self.profiles = {}

    def folds(self, j):
        if j in self.profiles:
            return self.profiles[j]
        bk = self.book
        y, pred, splits = bk.y, bk.oof[j], bk.assignment
        nclasses = len(bk.classes)
        vals = np.array([
            metrics(confusion(y[splits == f], pred[splits == f], nclasses))["precision"]
            for f in range(10)
        ])
        assert abs(vals.mean() - bk.scores[j]["fold"]) < 1e-13
        self.profiles[j] = vals
        return vals

    def trace(self, path, kinds):
        raw, ordinary, gated = [], [], []
        best = anchor = None
        for j, kind in zip(path, kinds):
            if best is None or dominance(self.book, j) < dominance(self.book, best):
                best = j
            if kind != "M" and (anchor is None or dominance(self.book, j) < dominance(self.book, anchor)):
                anchor = j
            assert anchor is not None
            keep = best == anchor or replacement(self.folds(best) - self.folds(anchor))
            raw.append(best)
            ordinary.append(anchor)
            gated.append(best if keep else anchor)
        assert len(path) == len(kinds) == len(set(path)) == 128
        a, z = raw[-1], ordinary[-1]
        y = self.book.y
        smallest = np.bincount(y).min()
        small = np.isin(y, np.flatnonzero(np.bincount(y) == smallest))
        ca, cz = self.book.oof[a] == y, self.book.oof[z] == y
        return dict(path=path, op=kinds, raw=raw, ordinary=ordinary, gated=gated,
                    end_diff=(self.folds(a) - self.folds(z)).tolist(),
                    small_class_wins=int(np.sum(ca & ~cz & small)),
                    small_class_losses=int(np.sum(cz & ~ca & small)))


class ProposalSearch(Search):
    """Use the same parent, neighbours and scoring; vary only the stated control."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unweighted = False
        self.unweighted_pairs = {}
        self.first_pair_changes = set()
        self.pair_visits = set()
        self.proxy_fields = 0

    def errors(self, idx):
        weighted = super().errors(idx)
        if not self.unweighted or len(weighted) < 2:
            return weighted
        if idx not in self.unweighted_pairs:
            pred = self.book.oof[idx]
            wrong = self.y != pred
            counts = Counter(tuple(sorted((int(a), int(b)))) for a, b in zip(self.y[wrong], pred[wrong]))
            ordered = sorted(weighted, key=lambda item: (-counts[item[1]], item[1]))
            if weighted[0][1] != ordered[0][1]:
                self.first_pair_changes.add(idx)
            self.unweighted_pairs[idx] = ordered
        self.pair_visits.add(idx)
        return self.unweighted_pairs[idx]

    def support(self, pair, mask, provider, old):
        base, _ = super().support(pair, mask, "numeric", old)
        if provider == "numeric":
            self.proxy_fields += len(base)
            return base, ()
        if provider not in ("tree_union", "mw_equal"):
            raise ValueError(provider)
        outside = np.ones(self.p, dtype=bool)
        outside[list(mask)] = False
        tree = self.tree[outside[self.tree]][:4]
        outside[base] = False
        extra = tree[outside[tree]]
        if provider == "mw_equal":
            rank, _ = self.order(pair)
            extra = rank[outside[rank]][:len(extra)]
        self.proxy_fields += len(base) + len(extra)
        return np.r_[base, extra], tuple(map(int, extra))


def buy_archived(book, archive, policy, delivery):
    ref = archive["arms"][policy]
    path = [book.buy(archive["masks"][j]) for j in ref["path"]]
    tr = delivery.trace(path, ref["op"])
    for j, old in zip(tr["raw"], ref["win"]):
        assert book.masks[j] == tuple(archive["masks"][old])
    return tr


def projected_draw(s, rs, slot, view, seen, menu):
    rng = np.random.RandomState(seed("q-review-ordinary", rs, slot))
    order = (s.tree, s.cdf)[view]
    for _ in range(64):
        kept = rng.uniform(size=s.p) > .6
        cols = order[kept[order]][:s.cap]
        mask = tuple(sorted(map(int, cols))) if len(cols) else menu[0]
        if mask not in seen:
            return mask
    return next(m for m in menu if m not in seen)


def mixed(s, rs, provider, delivery):
    menu, _ = schedule(s)
    widths = set(map(int, initial_widths(s.cap, len(s.y))[0]))
    priority = list(dict.fromkeys(len(m) for m in menu if len(m) in widths))
    assert set(priority) == widths
    q = len(widths)
    cursor = turn = 0
    seen, ages, population = set(), {}, {}
    path, ops = [], []
    fields_before = s.proxy_fields
    for slot in range(1, 129):
        pos = slot - 2 * q
        mask = None
        op = "R"
        if pos <= 0 and slot % 2:
            view = ((slot - 1) // 2) % 2
            ranks = (s.tree, s.cdf) if view == 0 else (s.cdf, s.tree)
            while cursor < len(priority) and mask is None:
                k = priority[cursor]
                cursor += 1
                for rank in ranks:
                    candidate = tuple(sorted(map(int, rank[:k])))
                    if candidate not in seen:
                        mask = candidate
                        break
            if mask is None:
                mask = next(m for m in menu if m not in seen)
            op = "G"
        elif pos > 0 and pos % 2 and provider != "iid":
            niches = rotated(sorted(population), turn)
            parents = [j for place in range(4) for g in niches for j in population[g][place:place+1]]
            if len(niches) == 1:
                parents = rotated(parents, turn)
            offer = s.mutate(parents, seen, provider)
            turn += 1
            if offer is not None:
                mask, op = offer[0], "M"
        if mask is None:
            view = ((slot // 2 - 1) if pos <= 0 else (q + (pos - 1) // 2)) % 2
            mask = projected_draw(s, rs, slot, view, seen, menu)
        assert mask not in seen and len(mask) <= s.cap
        seen.add(mask)
        j = s.book.buy(mask)
        ages[j] = slot
        path.append(j)
        ops.append(op)
        niche = len(mask).bit_length() - 1
        keep, behaviors = [], set()
        for c in sorted(population.get(niche, []) + [j], key=lambda z: (*dominance(s.book, z), ages[z])):
            behavior = s.book.oof[c].tobytes()
            if behavior not in behaviors:
                keep.append(c)
                behaviors.add(behavior)
            if len(keep) == 4:
                break
        population[niche] = keep
    result = delivery.trace(path, "".join(ops))
    result.update(warm=2*q, attempts=turn, proxy_fields=s.proxy_fields-fields_before)
    return result


def mel(s, rs, delivery):
    adapter = SimpleNamespace(x=s.x, y=s.y, seed=rs, book=s.book, base=SimpleNamespace(cap=s.cap, tree=s.tree, vote=s.menu))
    ref = projected.mel_path(adapter, 128)
    tr = delivery.trace(ref["path"], "P" * 128)
    assert tr["raw"] == ref["winners"]
    return tr
