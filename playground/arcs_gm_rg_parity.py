"""IB-level Rg parity on the REAL GM12878 from-scratch arcs IBs (captured via ARCS_DUMP).

The full pipeline fails Rg at ~0.85 and the arcs-only toggle pins it on the arcs checker.
But every prior parity used arcs_conv_ibs.pkl (already-CONVERGED IBs), which never exercise
from-scratch convergence.  This runs checker vs the sequential JAX arcs on the SAME captured
from-scratch seeds:

  - ratio ~0.85  -> the arcs MC itself over-compacts from scratch (a fast test bed for fixes)
  - ratio ~1.0   -> the arcs MC is fine per-IB; the 15% is a downstream/assembly effect
"""

import os
import pickle

import numpy as np

import gnome3d.mc.jax.arcs_checker as ac
from gnome3d.mc import jax as mc_jax

d = pickle.load(open("/tmp/gm_arcs.pkl", "rb"))
expanded = d["expanded"]
s = d["settings"]
print(f"{len(expanded)} from-scratch arcs IBs; bead counts {sorted(set(p['pos'].shape[0] for p in expanded))}")


def gyr(p):
    c = p.mean(0)
    return float(np.sqrt(((p - c) ** 2).sum(1).mean()))


_scale = float(os.environ.get("ARCS_SEED_SCALE", "1"))  # pre-expand the checker's collapsed seed
exp_ck = expanded
if _scale != 1.0:
    exp_ck = []
    for p in expanded:
        pos = np.asarray(p["pos"], np.float32)
        c = pos.mean(0)
        exp_ck.append({**p, "pos": (c + (pos - c) * _scale).astype(np.float32)})
res_ck = ac.mc_arcs_checker_jax_batch(exp_ck, s)
res_mc = mc_jax.mc_arcs_jax_batch(expanded, s)
rg_ck = np.array([gyr(np.asarray(p)) for _, p in res_ck])
rg_mc = np.array([gyr(np.asarray(p)) for _, p in res_mc])
e_ck = np.array([float(sc) for sc, _ in res_ck])
e_mc = np.array([float(sc) for sc, _ in res_mc])

print(f"{'IB':>4} {'chkRg':>9} {'seqRg':>9} {'chk/seq':>8} {'chkE/seqE':>10}")
for i in range(len(expanded)):
    print(f"{i:>4} {rg_ck[i]:>9.3f} {rg_mc[i]:>9.3f} {rg_ck[i] / max(rg_mc[i], 1e-9):>8.4f} "
          f"{e_ck[i] / max(e_mc[i], 1e-9):>10.4f}")
print(f"\nMEAN chk/seq Rg = {rg_ck.mean() / max(rg_mc.mean(), 1e-9):.4f}  "
      f"(energy {e_ck.mean() / max(e_mc.mean(), 1e-9):.4f})")
print("~0.85 = arcs MC over-compacts from scratch (test bed valid); ~1.0 = look downstream")
