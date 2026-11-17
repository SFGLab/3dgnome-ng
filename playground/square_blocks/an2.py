import json, numpy as np
from scipy.stats import spearmanr
d=json.load(open("raw.json")); H=np.array(d["upper"]); M=np.array(d["lower"])
bins=np.array(d["bins"]); n=len(bins); res=25000
loops=json.load(open("loopslist.json"))
beads=[];kind=[]
for l in open("chr1_20000000_25000000_s1.cif"):
    if l.startswith("ATOM"):
        f=l.split(); beads.append((int(f[-3])+int(f[-2]))//2); kind.append(f[-1])
beads=np.array(beads); kind=np.array(kind)
bi=(beads-bins[0])//res
nb=np.bincount(bi,minlength=n)[:n]; na=np.bincount(bi[kind=="anchor"],minlength=n)[:n]
ends=np.zeros(n)
for l in loops:
    for e in ("a","b"):
        i=int((np.mean(l[e])-bins[0])//res)
        if 0<=i<n: ends[i]+=1
cover=np.array([sum(1 for l in loops if l["a"][1]<p<l["b"][0]) for p in bins+res])
for k in (1,2,4):
    m=np.diagonal(M,k); h=np.diagonal(H,k); i=np.arange(n-k)
    feats={"beads i*j":nb[i]*nb[i+k],"anchors i+j":na[i]+na[i+k],"loop ends i+j":ends[i]+ends[i+k],"loops spanning":cover[i:i+k].mean() if False else np.array([cover[a:a+k].mean() for a in i])}
    print(f"sep {k*25}kb: " + "  ".join(f"{nm}: model {spearmanr(v,m)[0]:+.2f} hic {spearmanr(v,h)[0]:+.2f}" for nm,v in feats.items()))
# strength of loops: mean pets by factor
for fct in ("CTCF","RNAPII"):
    p=[l["pets"] for l in loops if l["factor"]==fct]; s=[l["span"] for l in loops if l["factor"]==fct]
    print(fct,len(p),"pets median",np.median(p),"span median kb",np.median(s)/1e3, "max kb", max(s)/1e3)
