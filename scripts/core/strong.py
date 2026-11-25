"""Test the two support expansions after exactly the original MEL-style founder purchases."""
from concurrent.futures import ProcessPoolExecutor
import sys
import numpy as np

from engine import (ROOT, Delivery, ExactBook, ProposalSearch, buy_archived, projected, read, require_lab,
                    run_seed, save, signature)
from local_study import runner


class FixedProvider(ProposalSearch):
    provider = "numeric"

    def support(self, pair, mask, provider, old):
        assert provider == "numeric"
        return super().support(pair, mask, self.provider, old)


def train(key):
    a=runner.archive_for(key)
    c=a["case"]
    x,y=projected.arrays(c)
    rs=run_seed(c)
    cap=min(8,max(1,int(.15*x.shape[1])))
    book=ExactBook(x,y,rs)
    s=FixedProvider(x,y,rs,128,limit=cap,book=book)
    delivery=Delivery(book)
    arms={n:buy_archived(book,a,n,delivery) for n in ("iid","mel","half")}
    narch=len(book.masks)
    for name in ("numeric","mw_equal","tree_union"):
        s.provider=name
        fields=s.proxy_fields
        out=projected.purchase(s,rs,"half")
        result=delivery.trace(out["path"],out["op"])
        result.update(warm=out["initial"],proxy_fields=s.proxy_fields-fields)
        assert result["raw"]==out["win"]
        assert result["path"][:out["initial"]]==arms["iid"]["path"][:out["initial"]]
        if name=="numeric":
            assert result["path"]==arms["half"]["path"] and result["raw"]==arms["half"]["raw"]
        arms[name]=result
    del arms["half"]
    d=ROOT/"strong/train"
    d.mkdir(exist_ok=True,parents=True)
    blob=d/(key+".npz")
    runner.mask_data(book,blob)
    save(d/(key+".json.gz"),dict(case=c,group=a["group"],cap=cap,n=len(y),p=x.shape[1],
         class_counts=np.bincount(book.y).tolist(),arms=arms,blob=signature(blob),physical=len(book.masks),
         reconstructed=narch,notes={}))
    return key,len(book.masks)


def test(key):
    seal=read(ROOT/"strong/training-seal.json")
    assert seal["extra"]==frozen()
    return runner.score(("strong",key))


def frozen():
    return {n:signature(ROOT/n) for n in ("STRONG.md","strong.py","local_study.py")}


if __name__=="__main__":
    require_lab()
    stage=sys.argv[1]
    assert stage in ("train","score")
    keys=runner.cases("replay")
    with ProcessPoolExecutor(7) as pool:
        for row in pool.map(train if stage=="train" else test,keys):
            print(*row,flush=True)
    d=ROOT/"strong"
    if stage=="train":
        save(d/"training-seal.json",dict(code=runner.provenance(),extra=frozen(),actions={
            k:signature(d/"train"/(k+".json.gz")) for k in keys}))
    else:
        save(d/"outer-seal.json",dict(training=signature(d/"training-seal.json"),scores={
            k:signature(d/"outer"/(k+".npz")) for k in keys}))
