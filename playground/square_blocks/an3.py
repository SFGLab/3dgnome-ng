import json, glob, numpy as np
from scipy.stats import spearmanr
d=json.load(open("raw.json")); H=np.array(d["upper"])
C=np.array(json.load(open("cnt.json"))["lower"])
bins=np.array(d["bins"]); n=len(bins); res=25000
S=[]
for f in sorted(glob.glob("*.cif")):
    P=[];mid=[];kd=[]
    for l in open(f):
        if l.startswith("ATOM"):
            x=l.split(); P.append([float(x[10]),float(x[11]),float(x[12])]); mid.append((int(x[-3])+int(x[-2]))//2); kd.append(x[-1])
    S.append(np.array(P))
mid=np.array(mid); kd=np.array(kd); bi=(mid-bins[0])//res
na=np.bincount(bi[kd=="anchor"],minlength=n)[:n]; nb=np.bincount(bi,minlength=n)[:n]
bond=np.median(np.concatenate([np.linalg.norm(np.diff(p,axis=0),axis=1) for p in S]))
# per bin spread: rg of the bin's beads, in bonds, mean over structures
rg=np.zeros(n)
for i in range(n):
    sel=bi==i
    if sel.sum()>1: rg[i]=np.mean([np.sqrt(((p[sel]-p[sel].mean(0))**2).sum(1).mean()) for p in S])/bond
# normalised to genomic size: rg per bead count^0.5
ok=nb>1
print("bond",round(bond,3))
print("bin Rg (bonds) vs anchors/bin: spearman %+.2f; vs beads/bin %+.2f"%(spearmanr(rg[ok],na[ok])[0],spearmanr(rg[ok],nb[ok])[0]))
print("Rg in anchor-poor (<=2) vs anchor-rich (>=8) bins: %.2f vs %.2f"%(rg[ok&(na<=2)].mean(),rg[ok&(na>=8)].mean()))
# counted per-pair probability: normalise counts by beads i*j
for k in (1,2,4):
    i=np.arange(n-k); c=np.diagonal(C,k); h=np.diagonal(H,k); bb=nb[i]*nb[i+k]; good=bb>0
    p=np.where(good,c/np.maximum(bb,1),0)
    print(f"sep {k*25}kb counted: anchors spearman {spearmanr(na[i]+na[i+k],c)[0]:+.2f}; per-pair prob vs anchors {spearmanr((na[i]+na[i+k])[good],p[good])[0]:+.2f}; r(count,hic) {np.corrcoef(c,h)[0,1]:+.2f} r(prob,hic) {np.corrcoef(p[good],h[good])[0,1]:+.2f}")
