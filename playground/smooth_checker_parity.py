import numpy as np, pickle, time
import gnome3d.mc.jax.smooth_checker as sc
import gnome3d.mc.numba as mc_numba
from gnome3d.settings import Settings
from gnome3d.util import seed_rng
caps=pickle.load(open('/tmp/smooth_ibs.pkl','rb'))
wh=[c for c in caps if c['heat'] is not None]; wh.sort(key=lambda c:c['pos'].shape[0])
tests=[c for c in wh if 200<=c['pos'].shape[0]<=700][:4]
s=Settings(); s.load_ini('data/GM12878/config.ini'); s.mc_executor_jax_bucket_shapes=True
print("PRODUCTION smooth checker vs numba sequential (EV+heat+confine, NO orientation, same seed):")
print(f'{"B":>5} {"checker_E":>11} {"seq_E":>11} {"chk/seq":>8}')
for c in tests:
    pos0=np.asarray(c['pos'],np.float64); dtn=np.asarray(c['dtn'],np.float64); fixed=np.asarray(c['fixed'],np.bool_); heat=np.asarray(c['heat'],np.float64)
    B=pos0.shape[0]; step=float(np.median(dtn[dtn>1e-6])) if (dtn>1e-6).any() else 1.0
    probs=[{'pos':pos0.astype(np.float32),'dtn':dtn.astype(np.float32),'fixed':fixed,'heat_dist':heat.astype(np.float32),'step_size':step}]
    (sc_ck,_),=sc.mc_smooth_checker_jax_batch(probs, s)
    # numba sequential (no orientation): pass char_orientations=None so use_orn=False
    p=pos0.copy(); seed_rng(0); mc_numba.seed_numba(0)
    sc_seq=mc_numba.mc_smooth_numba(p, dtn, fixed, step, s, None, None, None, heat)
    print(f'{B:>5} {sc_ck:>11.2f} {sc_seq:>11.2f} {sc_ck/max(sc_seq,1e-9):>8.3f}', flush=True)
print("PASS if chk/seq ~ 1.0 (production smooth checker matches sequential, no-orn)")
