"""Does the SMOOTH checker's compaction grow with B?  Rg(checker)/Rg(numba) across bead counts.

The integration FAILs Rg at ~0.85 (B=1024), far past the accepted ~3.5%.  The smooth checker's
Rg was never validated at the real scale - only its ENERGY (equal energy != equal compactness).
The smooth still uses spatial mod-2 (8 colours) x chain-mod-3 = 24; arcs had that 8-colour
stale-repulsion compaction (5.5%) until mod-3 (27).  If chkRg/numRg drops from ~0.97 at B=256 to
~0.85 at B=1024, the smooth needs the same mod-3 spatial fix.
"""

import pickle
from collections import defaultdict

import numpy as np

import gnome3d.mc.jax.smooth_checker as sc
import gnome3d.mc.numba as mc_numba
from gnome3d.mc.jax.util import jax_bucket_for
from gnome3d.settings import Settings
from gnome3d.util import seed_rng

caps = pickle.load(open("/tmp/smooth_ibs.pkl", "rb"))
by_bucket = defaultdict(list)
for c in caps:
    if c["heat"] is None:
        continue
    B = c["pos"].shape[0]
    if 150 <= B <= 1100:
        by_bucket[jax_bucket_for(B)].append(c)
tests = []
for bk in sorted(by_bucket):
    tests += sorted(by_bucket[bk], key=lambda c: c["pos"].shape[0])[:2]

s = Settings()
s.load_ini("data/GM12878/config.ini")
s.mc_executor_jax_bucket_shapes = True


def gyration(p):
    c = p.mean(0)
    return float(np.sqrt(((p - c) ** 2).sum(1).mean()))


print(f"{len(tests)} IBs across buckets {sorted(by_bucket)}")
print(f"{'B':>5} {'bucket':>7} {'chkRg/numRg':>12} {'chk_E/num_E':>12}")
for c in tests:
    pos0 = np.asarray(c["pos"], np.float64)
    dtn = np.asarray(c["dtn"], np.float64)
    fixed = np.asarray(c["fixed"], np.bool_)
    heat = np.asarray(c["heat"], np.float64)
    B = pos0.shape[0]
    step = float(np.median(dtn[dtn > 1e-6])) if (dtn > 1e-6).any() else 1.0
    probs = [{"pos": pos0.astype(np.float32), "dtn": dtn.astype(np.float32), "fixed": fixed,
              "heat_dist": heat.astype(np.float32), "step_size": step}]
    (eck, pck), = sc.mc_smooth_checker_jax_batch(probs, s)
    p = pos0.copy()
    seed_rng(0)
    mc_numba.seed_numba(0)
    enum = mc_numba.mc_smooth_numba(p, dtn, fixed, step, s, None, None, None, heat)
    rg_ck = gyration(pck[~fixed])
    rg_num = gyration(p[~fixed])
    print(f"{B:>5} {jax_bucket_for(B):>7} {rg_ck / max(rg_num, 1e-9):>12.4f} "
          f"{eck / max(enum, 1e-9):>12.4f}", flush=True)
print("compaction = 1 - chkRg/numRg; if it grows with B -> smooth needs mod-3 spatial (like arcs)")
