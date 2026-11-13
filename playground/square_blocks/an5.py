import json,sys,glob,numpy as np
sys.path.insert(0,"/Users/nk2/Desktop/3dgnome-torch/web")
from gnome3d_web import contacts as cm
d=json.load(open("raw.json")); H=np.array(d["upper"]); res=25000
S=[];mid=[];kd=[]
for f in sorted(glob.glob("*.cif")):
    P=[];mid=[];kd=[]
    for l in open(f):
        if l.startswith("ATOM"):
            x=l.split(); P.append([float(x[10]),float(x[11]),float(x[12])]); mid.append((int(x[-3])+int(x[-2]))//2); kd.append(x[-1])
    S.append(np.array(P))
mid=np.array(mid); kd=np.array(kd)
n=len(H); s=np.abs(np.subtract.outer(np.arange(n),np.arange(n)))+1.0
na=np.bincount((mid[kd=="anchor"]-20000000)//res,minlength=n)[:n].astype(float)
for name,sel in [("all beads",np.ones(len(mid),bool)),("anchors only",kd=="anchor"),("subanchors only",kd=="subanchor")]:
    M=cm.multimm_map([p[sel] for p in S],mid[sel],20000000,25000000,res)
    print(f"{name:16s} SCC raw {cm.scc(H,M,res):+.3f}  ICE {cm.scc(cm.ice(H),cm.ice(M),res):+.3f}")
# per-diagonal for anchors-only vs all
Ma=cm.multimm_map([p[kd=="anchor"] for p in S],mid[kd=="anchor"],20000000,25000000,res)
Mall=cm.multimm_map(S,mid,20000000,25000000,res)
for k in (1,2,4,8,16,40):
    print(f"  {k*25:5d} kb r all {np.corrcoef(np.diagonal(Mall,k),np.diagonal(H,k))[0,1]:+.2f} anchors {np.corrcoef(np.diagonal(Ma,k),np.diagonal(H,k))[0,1]:+.2f}")
print("--- same bins (bins holding an anchor), each map ICE balanced on that set")
keep=na>0
def sub(A): return A[np.ix_(keep,keep)]
Hk,Ak,Lk=cm.ice(sub(H)),cm.ice(sub(Ma)),cm.ice(sub(Mall))
print(f"bins kept {keep.sum()} of {n}; SCC anchors {cm.scc(Hk,Ak,res):+.3f} all beads {cm.scc(Hk,Lk,res):+.3f}")
for k in (1,2,4,8,16,40):
    print(f"  diag {k:3d} r all {np.corrcoef(np.diagonal(Lk,k),np.diagonal(Hk,k))[0,1]:+.2f} anchors {np.corrcoef(np.diagonal(Ak,k),np.diagonal(Hk,k))[0,1]:+.2f}")
# O/E contrast
def oe_sd(A):
    v=np.concatenate([np.diagonal(A,k)/np.diagonal(A,k).mean() for k in range(4,41)]); v=v[v>0]; return np.log2(v).std()
print(f"O/E sd log2 100kb-1Mb: HiC {oe_sd(Hk):.2f} anchors {oe_sd(Ak):.2f} all {oe_sd(Lk):.2f}")
print("--- O/E by bin class, both maps ICE balanced on all bins")
Hi,Mi=cm.ice(H),cm.ice(Mall)
def oe(A):
    O=np.zeros_like(A)
    for k in range(1,n):
        dg=np.diagonal(A,k); mu=dg[dg>0].mean() if (dg>0).any() else 0
        i=np.arange(n-k); O[i,i+k]=dg/mu if mu>0 else 0; O[i+k,i]=O[i,i+k]
    return O
Ho,Mo=oe(Hi),oe(Mall)
an=na>0
for lo,hi in ((1,2),(2,4),(4,8),(8,16),(16,40)):
    row=[]
    for nm,ci,cj in (("A-A",an,an),("0-0",~an,~an),("A-0",an,~an)):
        sel=np.zeros((n,n),bool)
        for k in range(lo,hi):
            i=np.arange(n-k); m=(ci[i]&cj[i+k])|(cj[i]&ci[i+k]); sel[i[m],i[m]+k]=True
        row.append(f"{nm} HiC {np.median(Ho[sel]):.2f} model {np.median(Mo[sel]):.2f}")
    print(f"  {lo*25}-{hi*25} kb  "+" | ".join(row))
# run lengths of anchorless stretches
runs=[];c=0
for v in ~an:
    if v: c+=1
    elif c: runs.append(c); c=0
if c: runs.append(c)
print("anchorless stretches (bins of 25 kb):",sorted(runs,reverse=True)[:12], "total", sum(runs))
