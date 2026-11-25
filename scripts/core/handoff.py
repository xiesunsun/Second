"""Width-balanced rank continuation and state-dependent steady-state feature selection."""
from math import isqrt
from numbers import Integral
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'width-evolution-20261123'))
from selector import Search, rotated, lab
from founders import Founders
from compromise import WidthMenus
from rank_neighbours import continuation, prefix_size
from numerical_search import capped_path

POLICIES = ('shell','sparse_fixed','sparse_mut','sparse_full','handoff_fixed','handoff_mut',
            'handoff_half','handoff_full','handoff_width','handoff_disagree','handoff_both')

def schedule(search):
    orders = (search.tree,search.cdf)
    cap, n, budget = search.cap, len(search.y), len(search.menu)
    count = prefix_size(orders,cap)
    if count >= budget:
        return search.menu, None
    dual = capped_path(orders,n,cap,count)
    menus = WidthMenus(*orders)
    vote = list(menus.candidates(capped_path((menus.mid,),n,cap,cap),'vote'))
    family = set(dual) | set(vote)
    direct, compromise = iter(dual), iter(vote)
    seen, path = set(), []
    for i in range(min(budget,len(family))):
        choice = None
        if i and isqrt(i)**2 == i:
            choice = next((m for m in compromise if m not in seen),None)
        if choice is None:
            choice = next((m for m in direct if m not in seen),None)
        if choice is None:
            choice = next(m for m in compromise if m not in seen)
        path.append(choice); seen.add(choice)
    if len(path) < budget:
        for mask in continuation(orders,n,cap,budget+len(family),'shell',0):
            if mask not in seen:
                path.append(mask); seen.add(mask)
                if len(path) == budget:
                    break
    return path, family

class Recombination:
    def __init__(self, search):
        self.search = search
        self.cache = {}

    def exclusive_scores(self, loser, other, cols):
        """Score an exclusive gene using mistakes corrected by its donor parent."""
        s = self.search
        correct = s.book.oof[other] == s.y
        contrast = []
        for _, _, groups in s.errors(loser):
            for rows, hits, misses in groups:
                keep = correct[rows]
                if keep.any():
                    r,h,m = rows[keep], hits[keep], misses[keep]
                    cur = s.rank_x[np.ix_(r,cols)]
                    distances = abs(cur - s.rank_x[np.ix_(m,cols)]) - abs(cur - s.rank_x[np.ix_(h,cols)])
                    contrast.append((int(s.y[r[0]]),len(r),distances.sum(0)))
        if not contrast:
            return np.zeros(len(cols))
        per_class = []
        for k in sorted({v[0] for v in contrast}):
            items = [v for v in contrast if v[0]==k]
            per_class.append(sum(v[2] for v in items)/sum(v[1] for v in items))
        return np.round(np.mean(per_class,axis=0),12)

    def child(self, a, b, wide, guided):
        s = self.search
        key = (min(a,b),max(a,b),wide,guided)
        if key not in self.cache:
            x,y = (set(s.book.masks[i]) for i in (a,b))
            left,right = sorted(x-y),sorted(y-x)
            both = x & y
            slots = (max(len(x),len(y)) if wide else (len(x)+len(y))//2)-len(both)
            child = None
            if left and right and slots >= 2:
                values = {j:0. for j in left+right}
                if guided:
                    values.update(zip(left,self.exclusive_scores(b,a,left)))
                    values.update(zip(right,self.exclusive_scores(a,b,right)))
                def order(j):return -values[j],s.tie[j],j
                anchors = {min(left,key=order),min(right,key=order)}
                take = sorted((x^y)-anchors,key=order)[:slots-2]
                child = tuple(sorted(both|anchors|set(take)))
            self.cache[key] = child
        return self.cache[key]

    def propose(self, parents, seen, wide, guided):
        for a in parents:
            size=len(self.search.book.masks[a])
            for b in sorted((i for i in parents if i!=a),key=lambda j:abs(len(self.search.book.masks[j])-size)):
                mask=self.child(a,b,wide,guided)
                if mask is not None and mask not in seen:
                    return mask,(a,b),False
        return None

def evolve(s, policy, menu, family, cross, provider='numeric'):
    if policy not in POLICIES:
        raise ValueError(policy)
    founder=Founders(s)
    missing=None if family is None else set(family)
    seen,age,depth,population=set(),{},{},{}
    path,wins,ops,generations,parents=[],[],[],[],[]
    cursor=turn=attempts=0
    transition=None
    champion=None
    def fit(j):
        return -s.book.scores[j]['fold'],len(s.book.masks[j]),age[j]
    for i in range(len(menu)):
        past = missing is not None and not missing
        if past and transition is None:
            transition=i+1
        opportunity=(i>0 and isqrt(i)**2==i) or (past and policy.startswith('handoff_'))
        offer=None
        if policy!='shell' and opportunity:
            if founder.required<=seen:
                niches=rotated(sorted(population),turn)
                pool=[j for r in range(4) for g in niches for j in population[g][r:r+1]]
                if policy.endswith('_fixed'):
                    pool=path[:1]
                elif len(niches)==1:
                    pool=rotated(pool,turn)
                alternating=policy.rsplit('_',1)[-1] in ('full','half','width','disagree','both')
                crossing=alternating and turn%2==1
                if not crossing:
                    attempts+=1; op='M'; offer=s.mutate(pool,seen,provider)
                elif not policy.endswith('_half'):
                    attempts+=1; op='X'
                    if policy.endswith('_full'):
                        offer=s.cross(pool,seen)
                    else:
                        offer=cross.propose(pool,seen,policy.endswith(('_width','_both')),policy.endswith(('_disagree','_both')))
            turn+=1
        if offer is None:
            while cursor<len(menu) and menu[cursor] in seen:
                cursor+=1
            if cursor==len(menu):
                break
            mask,ancestry,op=menu[cursor],(),'I'
            cursor+=1
        else:
            mask,ancestry,_=offer
        assert mask not in seen and 1<=len(mask)<=s.cap
        if op=='X':
            x,y=(set(s.book.masks[j]) for j in ancestry)
            assert x&y<=set(mask)<=x|y and set(mask)&(x-y) and set(mask)&(y-x)
        seen.add(mask)
        if missing is not None:
            missing.discard(mask)
        j=s.book.buy(mask); age[j]=i
        depth[j]=max((depth[h]+1 for h in ancestry),default=0)
        path.append(j);ops.append(op);generations.append(depth[j]);parents.append([age[h] for h in ancestry])
        champion=j if champion is None else min((champion,j),key=fit)
        wins.append(champion)
        if policy!='shell' and not policy.endswith('_fixed'):
            g=len(mask).bit_length()-1;keep=[];behaviors=set()
            for h in sorted(population.get(g,[])+[j],key=fit):
                behavior=s.book.oof[h].tobytes()
                if behavior not in behaviors:
                    behaviors.add(behavior);keep.append(h)
                    if len(keep)==4:
                        break
            population[g]=keep
    return dict(path=path,win=wins,op=''.join(ops),depth=generations,parents=parents,attempts=attempts,handoff=transition)

def select(x,y,*,budget=128,cap=8,seed=7319,policy='shell'):
    """Select original columns using training data only. The budget counts distinct wrappers."""
    lab()
    x=np.asarray(x,float);y=np.asarray(y)
    if x.ndim!=2 or len(x)<10 or y.shape!=(len(x),) or not np.isfinite(x).all():
        raise ValueError('Expected a finite matrix, at least ten rows and matching labels')
    if any(isinstance(v,bool) or not isinstance(v,Integral) or v<1 for v in (cap,budget)) or cap>x.shape[1]:
        raise ValueError('Invalid budget or feature limit')
    s=Search(x,y,seed,budget,limit=cap)
    menu,family=schedule(s)
    state=evolve(s,policy,menu,family,Recombination(s))
    j=state['win'][-1]
    return dict(columns=list(s.book.masks[j]),paid=len(state['path']),score=s.book.scores[j]['fold'],
                handoff=state['handoff'],mutations=state['op'].count('M'),crossovers=state['op'].count('X'))
