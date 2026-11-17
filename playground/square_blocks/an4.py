import json,sys,numpy as np
sys.path.insert(0,"/Users/nk2/Desktop/3dgnome-torch/web")
from gnome3d_web import contacts as cm
d=json.load(open("raw.json")); H=np.array(d["upper"]); M=np.array(d["lower"])
bins=np.array(d["bins"]); n=len(bins); res=25000
mid=[];kd=[]
for l in open("chr1_20000000_25000000_s1.cif"):
    if l.startswith("ATOM"): x=l.split(); mid.append((int(x[-3])+int(x[-2]))//2); kd.append(x[-1])
bi=(np.array(mid)-bins[0])//res; kd=np.array(kd)
nb=np.bincount(bi,minlength=n)[:n].astype(float); na=np.bincount(bi[kd=="anchor"],minlength=n)[:n].astype(float)
s=np.abs(np.subtract.outer(np.arange(n),np.arange(n)))+1.0
P=s**-1.0
print("SCC raw  model      %.3f"%cm.scc(H,M,res))
print("SCC ICE  model      %.3f"%cm.scc(cm.ice(H),cm.ice(M),res))
print("SCC raw  null beads %.3f"%cm.scc(H,np.outer(nb,nb)*P,res))
print("SCC raw  null anchors %.3f"%cm.scc(H,np.outer(na+1,na+1)*P,res))
print("SCC raw  null HiC coverage %.3f"%cm.scc(H,np.outer(H.sum(1),H.sum(1))*P,res))
print("SCC ICE  null anchors %.3f"%cm.scc(cm.ice(H),np.outer(na+1,na+1)*P,res))
