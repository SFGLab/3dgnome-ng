import numpy as np, pickle, logging
import gnome3d.mc.jax as mc_jax
caps=[]
def wsm(probs, s):
    for p in probs:
        caps.append({'pos':np.asarray(p['pos']).copy(),'fixed':np.asarray(p['fixed']).copy(),'dtn':np.asarray(p['dtn']).copy(),
                     'heat':(np.asarray(p['heat_dist']).copy() if p.get('heat_dist') is not None else None)})
    return [(0.0, np.asarray(p['pos'])) for p in probs]  # no-op (we only want inputs)
mc_jax.mc_smooth_jax_batch=wsm
logging.getLogger('gnome3d.mc.numba').setLevel(logging.WARNING)
from gnome3d.simulate import run_genome
run_genome('data/GM12878/config_dryrun.ini','chr1',1,data_dir='data/GM12878')
print(f'captured {len(caps)} smooth IBs', flush=True)
wh=[c for c in caps if c['heat'] is not None]
print(f'{len(wh)} have heat')
print(f'{"B":>7} {"fixed%":>7} {"heat":>5} {"heatdens%":>10} {"avg_contacts/bead":>18}')
for c in sorted(caps, key=lambda c: c['pos'].shape[0])[-8:]:
    B=c['pos'].shape[0]; nf=int(c['fixed'].sum()); h=c['heat']
    if h is not None:
        hm=np.asarray(h)>1e-6; hd=hm.sum()/(B*B)*100; cpb=hm.sum(1).mean()
    else: hd=0.0; cpb=0.0
    print(f'{B:>7} {100*nf/B:>6.1f}% {("yes" if h is not None else "no"):>5} {hd:>10.3f} {cpb:>18.1f}')
pickle.dump(caps, open('/tmp/smooth_ibs.pkl','wb'))
