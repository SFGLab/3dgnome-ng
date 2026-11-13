import glob,numpy as np
S=[];
for f in sorted(glob.glob("*.cif")):
    P=[];mid=[];kd=[]
    for l in open(f):
        if l.startswith("ATOM"):
            x=l.split(); P.append([float(x[10]),float(x[11]),float(x[12])]); mid.append((int(x[-3])+int(x[-2]))//2); kd.append(x[-1])
    S.append(np.array(P))
mid=np.array(mid); kd=np.array(kd); nu=0.344; s0=1000
gap=np.cumsum(kd=="anchor")  # gap id: subanchors after anchor k
order=np.argsort(mid)
# gap length in bp for each subanchor's gap
anc=np.where(kd=="anchor")[0]
glen={}
for a,b in zip(anc[:-1],anc[1:]): glen[gap[a]]=mid[b]-mid[a]
sub=np.where(kd=="subanchor")[0]
rng=np.random.default_rng(0)
rows=[]
for lo,hi in ((25e3,50e3),(50e3,100e3),(100e3,200e3)):
    for gl_lo,gl_hi,name in ((0,100e3,"gap<100kb"),(100e3,1e9,"gap>=100kb")):
        pass
print("median realised / law, pairs by separation; same-gap subanchor pairs vs pairs with an anchor between")
I,J=np.triu_indices(len(mid),1)
sep=np.abs(mid[J]-mid[I]); keep=(sep>=25e3)&(sep<200e3); I,J,sep=I[keep],J[keep],sep[keep]
D=np.mean([np.linalg.norm(p[I]-p[J],axis=1) for p in S],axis=0)
law=np.maximum(1,(sep/s0)**nu); r=D/law
same=(gap[I]==gap[J])&(kd[I]=="subanchor")&(kd[J]=="subanchor")
between=gap[J]-gap[I]
for lo,hi in ((25e3,50e3),(50e3,100e3),(100e3,200e3)):
    b=(sep>=lo)&(sep<hi)
    print(f"  {int(lo/1e3)}-{int(hi/1e3)} kb: same gap {np.median(r[b&same]):.2f} (n={int((b&same).sum())}); 1-2 anchors between {np.median(r[b&~same&(between<=2)]):.2f}; >=5 anchors between {np.median(r[b&(between>=5)]):.2f} (n={int((b&(between>=5)).sum())})")
