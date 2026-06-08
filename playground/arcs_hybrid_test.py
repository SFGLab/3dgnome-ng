"""HYBRID arcs: checker (fast, wrong 0.74 min) as initializer + sequential polish.

Proves two things on the real GM12878 from-scratch arcs IBs:
  1. CORRECTNESS: does a sequential re-anneal FROM the checker's output reach the same Rg
     as full sequential from the collapsed seed?  (rg_polish / rg_full ~ 1.0)
  2. SPEED: does the polish converge in FAR FEWER steps than full sequential?  The GPU cost
     is steps x per-step-latency, so the step count is the hardware-independent speed proxy
     (CPU wall time here does NOT reflect the GPU latency-bound win; the STEP RATIO does).

If the polish reaches full Rg in few steps -> hybrid keeps the checker speed + correctness.
If 0.74 is a barrier the polish can't cheaply escape -> hybrid dies, fall back to sequential.
"""

import pickle
import time

import numpy as np

import gnome3d.mc.jax.arcs_checker as ac
from gnome3d.mc import jax as mc_jax

d = pickle.load(open("/tmp/gm_arcs.pkl", "rb"))
expanded = d["expanded"]
s = d["settings"]


def gyr(p):
    c = p.mean(0)
    return float(np.sqrt(((p - c) ** 2).sum(1).mean()))


def rg(res):
    return float(np.mean([gyr(np.asarray(p)) for _, p in res]))


def bnd(res):
    out = []
    for _, p in res:
        p = np.asarray(p)
        out.append(float(np.mean(np.sqrt(((p[1:] - p[:-1]) ** 2).sum(1)))))
    return float(np.mean(out))


print("=== [1] full sequential from the collapsed seed (baseline) ===", flush=True)
t = time.perf_counter()
res_full = mc_jax.mc_arcs_jax_batch(expanded, s)
print(f"    full sequential: Rg={rg(res_full):.3f} bond={bnd(res_full):.3f}  ({time.perf_counter() - t:.1f}s)", flush=True)

print("=== [2] checker from the collapsed seed (fast init) ===", flush=True)
t = time.perf_counter()
res_ck = ac.mc_arcs_checker_jax_batch(expanded, s)
print(f"    checker: Rg={rg(res_ck):.3f}  ({time.perf_counter() - t:.1f}s)", flush=True)

print("=== [3] sequential POLISH from the checker's output ===", flush=True)
polish_in = [
    {"pos": np.asarray(pc, np.float32), "exp_dist": p["exp_dist"], "step_size": p["step_size"]}
    for p, (_, pc) in zip(expanded, res_ck, strict=True)
]
t = time.perf_counter()
res_pol = mc_jax.mc_arcs_jax_batch(polish_in, s)
print(f"    polish: Rg={rg(res_pol):.3f} bond={bnd(res_pol):.3f}  ({time.perf_counter() - t:.1f}s)", flush=True)

rg_full, rg_pol = rg(res_full), rg(res_pol)
print("\n--- VERDICT ---")
print(f"correctness: rg_polish/rg_full = {rg_pol / max(rg_full, 1e-9):.3f}  (want ~1.0)")
print(f"bonds:       bond_polish/bond_full = {bnd(res_pol) / max(bnd(res_full), 1e-9):.3f}  "
      f"(want ~1.0; >1 => polish over-expands arcs bonds; ~1 => the bond-KS is the smooth checker)")
print("speed: compare the 'N steps' in the [1] full vs [3] polish logs above")
print("       (steps x latency = GPU cost; polish << full => hybrid wins)")
