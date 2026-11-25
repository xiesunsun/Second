"""Protected steady-state feature selection. Training labels and purchased parents determine suggestions."""
import argparse
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
import json
from math import isqrt
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "supported-ranks-20261123"))
from budget_selector import purchase_path
from first_delivery import tree_rank, cdf_rank, first_widths
from rank_envelope import ExactBook, lab
from numerical_search import confusion, metrics, predictions

METHODS = ("vote", "old_flat", "fixed", "mutation", "cross", "full", "dense")
PROVIDERS = ("numeric", "union", "wide", "lexical", "shuffled")


def column_ranks(x):
    """One-based average ranks, vectorized across columns, including ties."""
    n = len(x)
    order = np.argsort(x, axis=0, kind="stable")
    sorted_x = np.take_along_axis(x, order, axis=0)
    boundary = sorted_x[:-1] != sorted_x[1:]
    pos = np.arange(1, n + 1)[:, None]
    start = np.maximum.accumulate(np.where(np.vstack((np.ones((1, x.shape[1]), bool), boundary)), pos, 0), axis=0)
    end = np.minimum.accumulate(np.where(np.vstack((boundary, np.ones((1, x.shape[1]), bool))), pos, n + 1)[::-1], axis=0)[::-1]
    ranks = np.empty_like(x, dtype=float)
    np.put_along_axis(ranks, order, (start + end) / 2., axis=0)
    return ranks


def read_card(schema, card, p, classes):
    payload = (json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if (card.get("model") != "gemini-3.8-flash" or card.get("schema") != sha256(payload).hexdigest()
            or len(schema["fields"]) != p or not np.array_equal(classes, np.arange(len(schema["classes"])))):
        raise ValueError("Frozen Gemini card must match ordered fields and integer class roster")
    pairs = list(combinations(range(len(classes)), 2))
    if len(pairs) != len(card["pairs"]):
        raise ValueError("Wrong number of class pairs")
    named = {}
    for pair, claims in zip(pairs, card["pairs"]):
        fields = [j for j, weight in claims[:4]]
        if (len(set(fields)) != len(fields) or any(type(j) != int or not 0 <= j < p or abs(w) not in (1, 2)
                                                    for j, w in claims[:4])):
            raise ValueError("Invalid field in Gemini card")
        named[pair] = fields
    return named


class Search:
    def __init__(self, x, labels, seed, budget, *, limit=None, schema=None, card=None, book=None, ranks=None):
        self.book = ExactBook(x, labels, seed) if book is None else book
        self.x, self.y = self.book.x, self.book.y
        self.p = self.x.shape[1]
        self.cap = max(1, int(.15 * self.p)) if limit is None else limit
        self.tree, self.cdf = ranks if ranks is not None else (
            tree_rank(self.x, self.y, seed), cdf_rank(self.x, self.y))
        self.menu = purchase_path(self.tree, self.cdf, len(self.y), self.cap, budget, "vote")
        self.tie = np.round(np.log1p(np.argsort(self.tree)) + np.log1p(np.argsort(self.cdf)), 12)
        self.count = np.bincount(self.y)
        self.schema = schema
        self.named = {} if card is None else read_card(schema, card, self.p, self.book.classes)
        self.perm = None
        self.fields = self.encoder = None
        self.rank_x = None
        self.orders, self.profiles, self.lexical_orders, self.fold_values = {}, {}, {}, {}

    def order(self, pair):
        if pair not in self.orders:
            a, b = pair
            ii = (self.y == a) | (self.y == b)
            aa = self.y[ii] == a
            n, m = int(aa.sum()), int((~aa).sum())
            ranks = column_ranks(self.x[ii])
            magnitude = np.abs(2*ranks[aa].sum(axis=0) - n*(n+1) - n*m)
            self.orders[pair] = (np.argsort(-magnitude, kind="stable"), magnitude)
        return self.orders[pair]

    def support(self, pair, mask, provider, old):
        order, effect = self.order(pair)
        inside = np.zeros(self.p, bool)
        inside[list(mask)] = True
        base = order[:4] if old else order[~inside[order]][:4]
        if provider == "numeric":
            return base, ()
        if not self.named:
            raise ValueError("Semantic controls require a validated card")
        added = [j for j in self.named[pair] if not inside[j] and j not in base]
        n = len(added)
        if n == 0:
            return base, ()
        if provider == "union":
            return np.r_[base, added], tuple(added)
        occupied = inside.copy()
        occupied[base] = True
        if provider == "wide":
            priority = order
        elif provider == "lexical":
            if pair not in self.lexical_orders:
                from sklearn.feature_extraction.text import TfidfVectorizer
                if self.encoder is None:
                    self.encoder = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5))
                    try:
                        self.fields = self.encoder.fit_transform(self.schema["fields"])
                    except ValueError as exc:
                        if "empty vocabulary" not in str(exc):
                            raise
                a, b = pair
                query = " ".join([self.schema["task"], self.schema["classes"][a], self.schema["classes"][b]])
                sim = np.zeros(self.p) if self.fields is None else (self.encoder.transform([query]) @ self.fields.T).toarray()[0]
                numbers = rankdata(-effect)
                self.lexical_orders[pair] = np.lexsort((np.arange(self.p), numbers, numbers + rankdata(-np.round(sim, 12))))
            priority = self.lexical_orders[pair]
        elif provider == "shuffled":
            if self.perm is None:
                raw = json.dumps(self.schema["fields"], ensure_ascii=False).encode()
                self.perm = np.random.RandomState(int.from_bytes(sha256(raw).digest()[:4], "big")).permutation(self.p)
            mapped = self.perm[self.named[pair]]
            priority = np.r_[mapped, self.perm[~np.isin(self.perm, mapped)]]
        else:
            raise ValueError(provider)
        extra = priority[~occupied[priority]][:n]
        assert len(extra) == n and not np.any(inside[extra])
        return np.r_[base, extra], tuple(map(int, extra))

    def errors(self, idx):
        if idx in self.profiles:
            return self.profiles[idx]
        mask, y = self.book.masks[idx], self.y
        pred = self.book.oof[idx]
        wrong = np.flatnonzero(pred != y)
        if not len(wrong):
            self.profiles[idx] = []
            return []
        if self.rank_x is None:
            self.rank_x = (column_ranks(self.x) - .5) / len(y)
        dist = cdist(self.x[wrong][:, mask], self.x[:, mask], "sqeuclidean")
        dist[self.book.assignment[wrong, None] == self.book.assignment[None, :]] = np.inf
        pairs = defaultdict(list)
        for i, row in enumerate(wrong):
            a, b = int(y[row]), int(pred[row])
            pairs[tuple(sorted((a, b)))].append((i, row, a, b))
        out = []
        for pair, rows in pairs.items():
            groups = defaultdict(list)
            priority = 0.
            for i, row, a, b in rows:
                priority += 1. / self.count[a]
                hit = np.where(y == a, dist[i], np.inf)
                miss = np.where(y == b, dist[i], np.inf)
                h, m = hit.argmin(), miss.argmin()
                if np.isfinite(hit[h]) and np.isfinite(miss[m]):
                    groups[a].append((row, h, m))
            if groups:
                out.append((-round(priority, 12), pair, [np.array(v).T for v in groups.values()]))
        self.profiles[idx] = sorted(out, key=lambda row: row[:2])
        return self.profiles[idx]

    def mutate(self, parents, seen, provider, old=False, swap=False):
        for parent in parents:
            mask = self.book.masks[parent]
            for _, pair, groups in self.errors(parent):
                fields, extras = self.support(pair, mask, provider, old)
                replace = len(mask) == self.cap or swap
                cols = np.unique(np.r_[fields, mask] if replace else fields)
                if not len(cols):
                    continue
                contrasts = []
                for r, h, m in groups:
                    current = self.rank_x[np.ix_(r, cols)]
                    contrasts.append((abs(current-self.rank_x[np.ix_(m, cols)])
                                      - abs(current-self.rank_x[np.ix_(h, cols)])).mean(axis=0))
                value = dict(zip(cols, np.round(np.mean(contrasts, axis=0), 12)))
                drop = min(mask, key=lambda j: (value[j], j)) if replace else None
                threshold = max(0., value[drop]) if replace else 0.
                for j in sorted(map(int, fields), key=lambda k: (-value[k], k)):
                    if value[j] <= threshold:
                        break
                    if j not in mask:
                        child = tuple(sorted([k for k in mask if k != drop] + [j]))
                        if child not in seen:
                            return child, (parent,), j in extras
        return None

    def cross(self, parents, seen):
        for a in parents:
            x = set(self.book.masks[a])
            for b in sorted((i for i in parents if i != a), key=lambda i: abs(len(self.book.masks[i])-len(x))):
                y = set(self.book.masks[b])
                left, right = x-y, y-x
                width = (len(x)+len(y))//2
                both = x & y
                slots = width-len(both)
                if not left or not right or slots < 2:
                    continue
                rank = lambda j: (self.tie[j], j)
                # All contested columns have one vote. Require biparental inheritance among equal-cost optima.
                anchors = {min(left, key=rank), min(right, key=rank)}
                take = sorted((left|right)-anchors, key=rank)[:slots-2]
                child = tuple(sorted(both|anchors|set(take)))
                if child not in seen:
                    return child, (a, b), False
        return None

    def fold_profile(self, idx):
        if idx not in self.fold_values:
            assignment = self.book.assignment
            c = len(self.count)
            self.fold_values[idx] = np.array([metrics(confusion(self.y[assignment==f], self.book.oof[idx][assignment==f], c))["precision"]
                                              for f in np.unique(assignment)])
        return self.fold_values[idx]


def rotated(items, turn):
    return items[turn % len(items):] + items[:turn % len(items)] if items else []


def walk(problem, method, provider="numeric", *, guard=True, warm_width=False):
    if method not in METHODS or provider not in PROVIDERS:
        raise ValueError("Unknown method or field proposal")
    book, menu = problem.book, problem.menu
    path, wins, gated, kinds, parents, generations, flags = [], [], [], [], [], [], []
    population, seen, age, depth = {}, set(), {}, {}
    native = getattr(book, "native", False)
    def key(j):
        return (-book.scores[j]["fold"], 0 if native else len(book.masks[j]), age[j])
    ordinary = champion = None
    cursor = turn = covered = 0
    for i in range(len(menu)):
        paid = i+1
        opportunity = (i > 0 and isqrt(i)**2 == i) if method != "dense" else i >= 4
        offer = None
        if method != "vote" and opportunity:
            groups = rotated(sorted(population), turn)
            active = [idx for rank in range(4) for g in groups for idx in population[g][rank:rank+1]]
            if method == "fixed":
                active = path[:1]
            # Width groups rotate through their rare purchase opportunities. Within a single group rotate parents.
            if len(groups) == 1 and method != "fixed":
                active = rotated(active, turn)
            crossing = method == "cross" or (method in ("full", "dense") and turn % 2 == 1)
            op = "X" if crossing else "M"
            if not warm_width or (cursor < len(menu) and sum(len(book.masks[j]) == len(menu[cursor]) for j in path) >= 2):
                offer = problem.cross(active, seen) if crossing else problem.mutate(active, seen, provider, old=method=="old_flat")
            turn += 1
        if offer is None:
            while cursor < len(menu) and menu[cursor] in seen:
                cursor += 1
            if cursor == len(menu):
                break
            mask, ancestry, extra = menu[cursor], (), False
            cursor += 1
            op = "I"
        else:
            mask, ancestry, extra = offer
        assert mask not in seen and 1 <= len(mask) <= problem.cap
        seen.add(mask)
        idx = book.buy(mask)
        if ancestry:
            assert all(j in age for j in ancestry)
            if op == "X":
                left, right = (set(book.masks[j]) for j in ancestry)
                assert left & right <= set(mask) <= left | right and set(mask)&(left-right) and set(mask)&(right-left)
        parents.append([age[j] for j in ancestry])
        depth[idx] = 1+max(depth[j] for j in ancestry) if ancestry else 0
        generations.append(depth[idx])
        age[idx] = i
        path.append(idx)
        flags.append(int(extra))
        kinds.append(op)
        if op == "I":
            ordinary = idx if ordinary is None else min((ordinary,idx), key=key)
        champion = idx if champion is None else min((champion,idx), key=key)
        wins.append(champion)
        delivered = champion
        if guard and champion != ordinary and not native:
            d = problem.fold_profile(champion)-problem.fold_profile(ordinary)
            if d.sum()-d.max() <= 1e-12:
                delivered = ordinary
        gated.append(delivered)
        if method not in ("vote","fixed"):
            niche = 0 if method == "old_flat" else len(mask).bit_length()-1
            candidates = sorted(population.get(niche, [])+[idx], key=key)
            distinct, survivors = set(), []
            for j in candidates:
                behavior = book.oof[j].tobytes()
                if behavior not in distinct:
                    survivors.append(j)
                    distinct.add(behavior)
                if len(survivors) == 4:
                    break
            population[niche] = survivors
        while covered < len(menu) and menu[covered] in seen:
            covered += 1
        if method != "dense":
            assert covered >= paid-isqrt(i)
    return dict(path=path, win=wins, gated=gated, op="".join(kinds), parents=parents, depth=generations, extra=flags)


def select(x, labels, *, budget=32, seed=7319, limit=None, method="full", schema=None, card=None,
           provider="numeric", cautious=False, trace=False, first_callback=None, query=None, warm_width=False):
    lab()
    x, labels = np.asarray(x, float), np.asarray(labels)
    if (x.ndim != 2 or x.shape[1] == 0 or len(x) < 10 or labels.shape != (len(x),)
            or not np.isfinite(x).all() or labels.dtype.kind not in "iufbUS"
            or (labels.dtype.kind == "f" and not np.isfinite(labels).all())):
        raise ValueError("Expected a finite training matrix with at least ten rows and matching labels")
    def valid_int(v):
        return isinstance(v,(int,np.integer)) and not isinstance(v,(bool,np.bool_))
    if not valid_int(budget) or budget < 1 or (limit is not None and (not valid_int(limit) or not 1 <= limit <= x.shape[1])):
        raise ValueError("Invalid budget or feature limit")
    if (provider != "numeric" and (card is None or schema is None)) or (card is not None and schema is None):
        raise ValueError("Semantic field providers require their frozen card and schema")
    if method not in METHODS or provider not in PROVIDERS:
        raise ValueError("Unknown search or nomination rule")
    if card is not None:
        read_card(schema, card, x.shape[1], np.unique(labels))
    cap = max(1,int(.15*x.shape[1])) if limit is None else int(limit)
    start = perf_counter()
    book = ExactBook(x, labels, seed)
    tree = tree_rank(x,book.y,seed)
    k, _ = first_widths(len(x),cap)
    first = tuple(sorted(map(int,tree[:k])))
    idx = book.buy(first)
    first_s = perf_counter()-start
    if first_callback is not None:
        first_callback(dict(columns=list(first),paid=1,score=book.scores[idx]["fold"],seconds=first_s))
    cdf = cdf_rank(x,book.y) if budget>1 else tree
    problem = Search(x,labels,seed,int(budget),limit=cap,schema=schema,card=card,book=book,ranks=(tree,cdf))
    state = walk(problem,method,provider,guard=cautious,warm_width=warm_width)
    assert book.masks[state["path"][0]] == first
    key = "gated" if cautious else "win"
    idx = state[key][-1]
    mask = book.masks[idx]
    out = dict(columns=list(mask),score=book.scores[idx]["fold"],requested=int(budget),paid=len(state["path"]),
               offspring=len(state["path"])-state["op"].count("I"),crossovers=state["op"].count("X"),
               max_depth=max(state["depth"]),first_seconds=first_s,seconds=perf_counter()-start)
    if trace:
        out["trace"] = [dict(columns=list(book.masks[j]),operation=op,parents=[v+1 for v in ps],depth=g,extra=extra)
                        for j,op,ps,g,extra in zip(state["path"],state["op"],state["parents"],state["depth"],state["extra"])]
    if query is not None:
        q = np.asarray(query,float)
        if q.ndim != 2 or q.shape[1] != x.shape[1] or not np.isfinite(q).all():
            raise ValueError("Finite query rows must use the same columns")
        out["predictions"] = []
        for i in range(0,len(q),256):
            d = cdist(q[i:i+256,mask],x[:,mask],"sqeuclidean")
            p = predictions(d,np.arange(len(x)),np.arange(len(d)),book.y,len(book.classes))
            out["predictions"].extend(book.classes[p].tolist())
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train"); parser.add_argument("--budget",type=int,default=32); parser.add_argument("--seed",type=int,default=7319)
    parser.add_argument("--method",choices=METHODS,default="full"); parser.add_argument("--provider",choices=PROVIDERS,default="numeric")
    parser.add_argument("--schema",type=Path); parser.add_argument("--card",type=Path); parser.add_argument("--query",type=Path)
    parser.add_argument("--cautious",action="store_true"); parser.add_argument("--trace",action="store_true"); parser.add_argument("--warm-width", action="store_true")
    args=vars(parser.parse_args()); train=args.pop("train")
    for name in ("schema","card"):
        if args[name]: args[name]=json.loads(args[name].read_bytes())
    if args["query"]:
        with np.load(args["query"],allow_pickle=False) as data: args["query"]=data["x"]
    with np.load(train,allow_pickle=False) as data: result=select(data["x"],data["y"],**args)
    print(json.dumps(result,allow_nan=False))
