"""Small-label, class-paired external assessment of semantic initialization."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from math import isqrt
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from handoff import Search,Recombination,lab,evolve,schedule,HERE
from selector import tree_rank,cdf_rank
from trial import seed,save,read,signature,putz,getz
from clinical_metadata import DIR,SPEC,schema,digest

RES=HERE/'results/clinical'
REPEATS=20
ARMS=('numeric','numthird','lex','manual','gemini','wrong','perm','static_gemini')
CODES={'f':0,'m':1,'y':1,'n':0,'s':.5,'yes':1,'no':0,'normal':0,'abnormal':1,
       'present':1,'notpresent':0,'good':0,'poor':1}
MISSING={'','?','nan','nann','na','none','\t?'}
def rows(t):
    d=pd.read_csv(DIR/'sources'/(t+'.csv'),dtype=str,keep_default_na=False)
    fields=[a['name'] for a in schema(t)['fields']]
    def numeric(value):
        v=str(value).strip().lower()
        if v in MISSING:return np.nan
        if v in CODES:return CODES[v]
        return float(v)
    x=np.array([[numeric(value) for value in row] for row in d[fields].to_numpy()])
    if t=='pbc' or t=='failure':
        days=pd.to_numeric(d['N_Days' if t=='pbc' else 'time']).to_numpy()
        event=(d.Status=='D').to_numpy() if t=='pbc' else pd.to_numeric(d.death_event).to_numpy()==1
        cut=730 if t=='pbc' else 90
        keep=event | (days>=cut);y=(event & (days<=cut)).astype(int)
    elif t=='hcv':
        labels=d.Category.str.strip();keep=~labels.str.startswith('0s').to_numpy()
        assert labels.str.match(r'^(?:0s?|[123])=').all()
        y=(~labels.str.startswith('0')).astype(int).to_numpy()
    elif t=='ckd':
        v=d['class'].str.strip();assert set(v)=={'ckd','notckd'}
        y=(v=='ckd').astype(int).to_numpy();keep=np.ones(len(y),bool)
    elif t=='hepatitis':
        v=pd.to_numeric(d['Class']).to_numpy();assert set(v)=={1,2}
        y=(v==1).astype(int);keep=np.ones(len(y),bool)
    else:
        y=(pd.to_numeric(d.num).to_numpy()>0).astype(int);keep=np.ones(len(y),bool)
    bags={}
    for i in np.flatnonzero(keep):
        bags.setdefault(sha256(x[i].tobytes()).digest(),[]).append(i)
    ids=np.array([v[0] for v in bags.values() if len(set(y[v]))==1])
    rng=np.random.RandomState(seed('clinical-holdout-20261123',t))
    split=[rng.permutation(ids[y[ids]==c]) for c in (0,1)]
    test=np.sort(np.concatenate([a[:len(a)//2] for a in split]))
    train=np.sort(np.concatenate([a[len(a)//2:] for a in split]))
    assert set(train).isdisjoint(test)
    return x,y,train,test

def sample(t,r,n):
    x,y,train,test=rows(t)
    rng=np.random.RandomState(seed('clinical-donor-20261123',t,r))
    groups=[rng.permutation(train[y[train]==c])[:n] for c in (0,1)]
    ids=np.asarray(groups).T.ravel()
    assert len(set(ids))==2*n and not set(ids)&set(test)
    return x,y,ids,test

def empirical(train,query):
    out=np.full(query.shape,.5)
    for j in range(query.shape[1]):
        a=np.sort(train[np.isfinite(train[:,j]),j]);ok=np.isfinite(query[:,j])
        if len(a):
            col=query[ok,j];out[ok,j]=(np.searchsorted(a,col,'left')+np.searchsorted(a,col,'right'))/(2*len(a))
    return out

def margins(raw,labels,query):
    x=empirical(raw,raw);q=empirical(raw,query)
    a,b=x[labels==0].mean(0),x[labels==1].mean(0)
    return (q-a)**2-(q-b)**2

class SmallBook:
    def __init__(self,x,y):
        self.x=empirical(x,x);self.y=y;self.classes=np.array([0,1])
        self.assignment=np.arange(len(y))//2
        self.margin=np.zeros(x.shape)
        for f in np.unique(self.assignment):
            te=self.assignment==f
            self.margin[te]=margins(x[~te],y[~te],x[te])
        self.masks,self.scores,self.oof=[],[],[];self.ids={}
    def buy(self,mask):
        if mask not in self.ids:
            pred=(self.margin[:,mask].sum(1)>0).astype(np.int32);i=len(self.masks);self.ids[mask]=i
            self.masks.append(mask);self.oof.append(pred);self.scores.append({'fold':float((pred==self.y).mean())})
        return self.ids[mask]

def third_menu(base,family,order,cap):
    if order is None:return base,family,set()
    widths=list(dict.fromkeys([cap]+[len(m) for m in base]+list(range(1,cap+1))))
    extra=[tuple(sorted(map(int,order[:k]))) for k in widths]
    existing=set();path=[];ordinary,prior=iter(base),iter(extra)
    for i in range(len(base)):
        m=next((z for z in prior if z not in existing),None) if i>0 and isqrt(i)**2==i else None
        if m is None:m=next((z for z in ordinary if z not in existing),None)
        if m is None:break
        path.append(m);existing.add(m)
    return path,None if family is None else family|set(extra),set(extra)

def train(t):
    s=schema(t);names=[i['name'] for i in s['fields']]
    order={}
    for k,typ in [('gemini','correct'),('wrong','wrong')]:
        f=DIR/'cards'/t/typ/'card.json';card=read(f);assert card['schema']==digest(s) and card['model']=='gemini-3.8-flash'
        order[k]=np.array(card['order'])
    rng=np.random.RandomState(seed('clinical-perm-20261123',t))
    order['perm']=rng.permutation(len(names))[order['gemini']]
    texts=[j['name'].replace('_',' ')+' '+j['description'] for j in s['fields']]
    vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5));features=vec.fit_transform(texts)
    lexical=np.asarray((vec.transform([s['task']])@features.T).toarray()[0])
    manual=[names.index(z) for z in s['manual']]
    order['manual']=np.array(manual+[i for i in range(len(names)) if i not in manual])
    records=[]
    for r in range(REPEATS):
        for n in (2,3,5):
            x,y,ids,_=sample(t,r,n);book=SmallBook(x[ids],y[ids]);rs=seed('clinical-EA-20261123',t,r,n)
            ranks=tree_rank(book.x,book.y,rs),cdf_rank(book.x,book.y)
            for cap in (4,8):
                obj=Search(book.x,book.y,rs,32,limit=cap,book=book,ranks=ranks)
                base,family=schedule(obj);recombine=Recombination(obj)
                third=dict(order,numeric=None,numthird=np.argsort(obj.tie,kind='stable'),
                           lex=np.lexsort((np.arange(len(names)),obj.tie,-np.round(lexical,12))))
                arms={}
                for k in ARMS:
                    if k=='static_gemini':
                        mask=tuple(sorted(map(int,order['gemini'][:cap])));j=book.buy(mask)
                        st={'path':[j],'win':[j],'op':'I','depth':[0],'parents':[[]]};semset={mask}
                    else:
                        menu,fam,semset=third_menu(base,family,third[k],cap)
                        st=evolve(obj,'handoff_mut',menu,fam,recombine)
                    assert len(set(st['path']))==len(st['path'])
                    owned=[book.masks[j] in semset for j in st['path']]
                    propagated=[]
                    for i,pp in enumerate(st['parents']):
                        propagated.append(owned[i] or any(propagated[h] for h in pp))
                    ages={j:i for i,j in enumerate(st['path'])}
                    arms[k]=dict(path=st['path'],win=st['win'],op=st['op'],
                         semantic_parent_children=sum(bool(pp) and any(propagated[h] for h in pp) for pp in st['parents']),
                         winner_prior=[int(propagated[ages[j]]) for j in st['win']])
                records.append(dict(repeat=r,n=n,cap=cap,ids=ids.tolist(),masks=list(book.masks),arms=arms))
    putz(RES/'train'/(t+'.json.gz'),records)
    return t

def outer(t):
    inputs=read(RES/'seal.json');f=RES/'train'/(t+'.json.gz')
    assert signature(f)==inputs['actions'][t]
    result=[]
    for rec in getz(f):
        r,n,cap=rec['repeat'],rec['n'],rec['cap']
        x,y,ids,test=sample(t,r,n)
        assert ids.tolist()==rec['ids']
        delta=margins(x[ids],y[ids],x[test]);v=y[test]
        scores=[]
        for m in rec['masks']:
            pred=(delta[:,m].sum(1)>0).astype(int)
            scores.append(sum((pred[v==c]==c).mean() for c in (0,1))/2.)
        scores=np.array(scores)
        for k,a in rec['arms'].items():
            length=len(a['win'])
            for band,lo,hi in [('main',2,16),('b1',1,1),('b2',2,2),('b8',8,8),('b16',16,16),('b32',32,32)]:
                b=np.arange(lo,hi+1);pos=np.minimum(b,length)-1;w=np.log1p(1/b)
                ba=np.average(scores[np.array(a['win'])[pos]],weights=w)
                oracle=np.average(np.maximum.accumulate(scores[a['path']])[pos],weights=w)
                actual=np.average(np.array([len(rec['masks'][j]) for j in a['win']])[pos],weights=w)
                result.append(dict(t=t,r=r,n=n,cap=cap,arm=k,band=band,ba=ba,oracle=oracle,width=actual,
                               mutations=a['op'].count('M'),semantic_parent_children=a['semantic_parent_children'],paid=length,
                               delivered_prior=np.average(np.array(a['winner_prior'])[pos],weights=w)))
    pd.DataFrame(result).to_csv(RES/(t+'.tsv.gz'),sep='\t',index=False)
    return t

def report():
    from paired_report import infer
    d=pd.concat([pd.read_csv(RES/(t+'.tsv.gz'),sep='\t') for t in SPEC])
    co=['n','cap','band','arm'];measure=['ba','width','oracle','mutations','semantic_parent_children','paid','delivered_prior']
    sources=d.groupby(co+['t'])[measure].mean().reset_index();sources.to_csv(RES/'sources.tsv',sep='\t',index=False)
    means=sources.groupby(co)[measure].mean().reset_index();means.to_csv(RES/'means.tsv',sep='\t',index=False)
    diffs=[]
    for (n,cap,band),part in sources.groupby(co[:3]):
        piv=part.pivot(index='t',columns='arm',values='ba')
        for a,b in [('gemini',b) for b in ARMS if b!='gemini']+[('wrong','numeric'),('perm','numeric')]:
            diffs.append(dict(n=n,cap=cap,band=band,a=a,b=b,**infer(piv[a]-piv[b])))
    pd.DataFrame(diffs).to_csv(RES/'contrasts.tsv',sep='\t',index=False)
    print(means[(means.n==3)&(means.cap==4)&(means.band=='main')].to_string(index=False),flush=True)
if __name__=='__main__':
    lab();RES.mkdir(exist_ok=True,parents=True)
    phase=sys.argv[1]
    if phase=='report':report()
    else:
        with ProcessPoolExecutor(6) as pool:print(list(pool.map(train if phase=='train' else outer,list(SPEC))),flush=True)
        if phase=='train':
            save(RES/'seal.json',dict(actions={t:signature(RES/'train'/(t+'.json.gz')) for t in SPEC},files={str(p.relative_to(HERE)):signature(p) for p in [HERE/'CLINICAL.md',HERE/'clinical_trial.py',HERE/'clinical_cards.py',HERE/'clinical_metadata.py',HERE/'handoff.py',*DIR.rglob('card.json'),*DIR.glob('sources/*')]}))
        else:
            save(RES/'outer.json',dict(seal=signature(RES/'seal.json'),scores={t:signature(RES/(t+'.tsv.gz')) for t in SPEC}))
