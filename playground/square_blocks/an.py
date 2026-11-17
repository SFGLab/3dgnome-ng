import json, sys, numpy as np
from scipy.stats import pearsonr, spearmanr
d=json.load(open("raw.json")); H=np.array(d["upper"]); M=np.array(d["lower"])
C=np.array(json.load(open("cnt.json"))["lower"])
bins=np.array(d["bins"]); n=len(bins); res=25000
loops=json.load(open("loopslist.json"))
def oe(A):
    O=np.zeros_like(A)
    for k in range(-n+1,n):
        dg=np.diagonal(A,k); m=dg.mean()
        idx=np.arange(max(0,-k),min(n,n-k))
        O[idx,idx+k]=dg/m if m>0 else 0
    return O
def ins(A,w=8):
    out=np.full(n,np.nan)
    for i in range(w,n-w): out[i]=np.log(A[i-w:i,i+1:i+1+w].mean()+1e-12)
    return out
# per-diagonal r
print("per-diagonal Pearson r (model vs Hi-C), separation kb:")
for k in [1,2,4,8,16,40,80]:
    print(f"  {k*25:5d} kb  multimm r={pearsonr(np.diagonal(M,k),np.diagonal(H,k))[0]:+.2f}  counted r={pearsonr(np.diagonal(C,k),np.diagonal(H,k))[0]:+.2f}")
# loop coverage per bin edge: loops spanning position
mid=bins+res/2
cover=np.array([sum(1 for l in loops if l["a"][1]<p<l["b"][0]) for p in bins+res])
iH,iM,iC=ins(oe(H)),ins(oe(M)),ins(oe(C))
ok=np.isfinite(iH)
print("insulation Pearson: model vs HiC %.2f, counted vs HiC %.2f"%(pearsonr(iM[ok],iH[ok])[0],pearsonr(iC[ok],iH[ok])[0]))
print("insulation vs log(1+loops spanning): HiC %.2f model %.2f"%(spearmanr(iH[ok],cover[ok])[0],spearmanr(iM[ok],cover[ok])[0]))
# contrast: OE within-vs-between spread
for name,A in [("HiC",H),("model",M),("counted",C)]:
    O=oe(A); v=np.concatenate([np.diagonal(O,k) for k in range(4,41)]); v=v[v>0]
    print(f"{name:8s} O/E spread 100kb-1Mb: sd(log2)={np.log2(v).std():.2f}  p5={np.percentile(v,5):.2f} p95={np.percentile(v,95):.2f}")
# bead density per bin
beads=[]
for l in open("chr1_20000000_25000000_s1.cif"):
    if l.startswith("ATOM"):
        f=l.split(); beads.append((int(f[-3])+int(f[-2]))//2)
beads=np.array(beads); cnt=np.bincount((beads-bins[0])//res,minlength=n)[:n]
cov=H.sum(1)
print("beads/bin: min %d median %d max %d ; zero bins %d"%(cnt.min(),np.median(cnt),cnt.max(),(cnt==0).sum()))
print("Spearman beads/bin vs HiC row coverage: %.2f"%spearmanr(cnt,cov)[0])
print("loops spanning each 25kb edge: min %d median %d max %d, edges with 0: %d"%(cover.min(),np.median(cover),cover.max(),(cover==0).sum()))
np.save("../cover.npy",cover); np.save("../ins.npy",np.vstack([iH,iM,iC]))
