import numpy as np, pickle, time
import gnome3d.mc.jax.arcs_checker as ck
import gnome3d.mc.numba as mc_numba
from gnome3d.settings import Settings
from gnome3d.util import seed_rng
ibs=pickle.load(open('/tmp/arcs_conv_ibs.pkl','rb')); byn={t[1].shape[0]:t for t in ibs}
s=Settings(); s.load_ini('data/GM12878/config.ini')
s.exclusion_apply_to_arcs=True            # mirror the user's real config (EV on for arcs)
s.mc_executor_jax_bucket_shapes=True
print("PRODUCTION CHECKER vs numba sequential (both EV-on, same seed, to convergence):")
print(f'{"N":>5} {"bucket":>7} {"checker_E":>11} {"seq_E":>11} {"chk/seq":>8} {"ck_s":>7} {"seq_s":>7}')
for N in (103, 110, 462, 664):
    pos0, exp, step = byn[N]
    pos0=np.asarray(pos0,np.float64); exp=np.asarray(exp,np.float64); step=float(step)
    # checker (production batch entry, single IB)
    probs=[{'pos':pos0.astype(np.float32),'exp_dist':exp.astype(np.float32),'step_size':step}]
    t=time.perf_counter(); (sc_ck,_),=ck.mc_arcs_checker_jax_batch(probs, s); tck=time.perf_counter()-t
    # numba sequential reference, same start
    p=pos0.copy(); seed_rng(0); mc_numba.seed_numba(0)
    t=time.perf_counter(); sc_seq=mc_numba.mc_arcs_numba(p, exp, step, s); tseq=time.perf_counter()-t
    b=ck.jax_bucket_for(N)
    print(f'{N:>5} {b:>7} {sc_ck:>11.1f} {sc_seq:>11.1f} {sc_ck/max(sc_seq,1e-9):>8.3f} {tck:>7.0f} {tseq:>7.0f}', flush=True)
print("PASS if chk/seq ~ 1.0 (production checker matches sequential energy, EV-on)")
