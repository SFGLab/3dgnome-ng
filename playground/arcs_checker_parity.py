import numpy as np, pickle, time
import gnome3d.mc.jax.arcs_checker as ck
import gnome3d.mc.numba as mc_numba
from gnome3d.settings import Settings
from gnome3d.util import seed_rng
ibs=pickle.load(open('/tmp/arcs_conv_ibs.pkl','rb')); byn={t[1].shape[0]:t for t in ibs}
s=Settings(); s.load_ini('data/GM12878/config.ini')
s.exclusion_apply_to_arcs=True            # mirror the user's real config (EV on for arcs)
s.mc_executor_jax_bucket_shapes=True
def _rg(p):
    c = p.mean(0)
    return float(np.sqrt(((p - c) ** 2).sum(1).mean()))

print("PRODUCTION CHECKER (27-color) vs numba sequential (both EV-on, same seed, to convergence):")
print(f'{"N":>5} {"chk_E":>10} {"seq_E":>10} {"chk/seq":>8} {"chk_Rg":>8} {"seq_Rg":>8} {"Rg ratio":>9} {"ck_s":>6}')
for N in (103, 110, 462, 664):
    pos0, exp, step = byn[N]
    pos0=np.asarray(pos0,np.float64); exp=np.asarray(exp,np.float64); step=float(step)
    probs=[{'pos':pos0.astype(np.float32),'exp_dist':exp.astype(np.float32),'step_size':step}]
    t=time.perf_counter(); (sc_ck,pos_ck),=ck.mc_arcs_checker_jax_batch(probs, s); tck=time.perf_counter()-t
    p=pos0.copy(); seed_rng(0); mc_numba.seed_numba(0)
    sc_seq=mc_numba.mc_arcs_numba(p, exp, step, s)
    rck, rseq = _rg(np.asarray(pos_ck)), _rg(p)
    print(f'{N:>5} {sc_ck:>10.1f} {sc_seq:>10.1f} {sc_ck/max(sc_seq,1e-9):>8.3f} '
          f'{rck:>8.3f} {rseq:>8.3f} {rck/max(rseq,1e-9):>9.3f} {tck:>6.0f}', flush=True)
print("PASS if chk/seq ~ 1.0 (energy) AND Rg ratio ~ 1.0 (27-color removed the compaction bias)")
