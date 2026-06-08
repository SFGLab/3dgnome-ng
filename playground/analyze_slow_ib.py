"""Analyze a slow-converging arcs IB captured via ARCS_SLOW_DUMP.

Question: is the slow IB a HARD IB (expansion just slow - Rg climbs monotonically
toward the polish target as the round budget grows) or a PATHOLOGY (Rg stuck near
the checker baseline / oscillating)?

Reports size + contact density, then runs the sequential polish FROM the checker
output at rising round caps.  Same input + fixed seed => each cap is the same
trajectory truncated, so Rg-vs-cap traces the single polish trajectory.

Usage:  python playground/analyze_slow_ib.py [pickle=/tmp/slow_ib.pkl] [caps=200,500,...]
Run on the GPU box for speed; the polish is latency-bound and a 20k-round cap is
slow on CPU.
"""

import os
import pickle
import sys
import time

import numpy as np

from gnome3d.mc import jax as mc_jax

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/slow_ib.pkl"
d = pickle.load(open(path, "rb"))
inp = d["input"]
s = d["settings"]
s.arcs_repulsion_cutoff_factor = float(os.environ.get("REP_FACTOR", "0"))  # 0=off; set to test the fix
n = int(d["n"])
exp = np.asarray(inp["exp_dist"], np.float32)


def rg(pos: object) -> float:
    p = np.asarray(pos)
    c = p - p.mean(0)
    return float(np.sqrt((c**2).sum(1).mean()))


# --- structure: size + contact density --------------------------------------
contacts = int((exp > 0).sum())
allc = d["all_conv"]
print(
    f"IB: n={n}  checker conv_round={d['conv_round']}  "
    f"contacts={contacts} ({contacts / (n * n):.1%} of n^2, ~{contacts / max(n, 1):.1f}/bead)"
)
print(f"batch conv spread: median={int(np.median(allc))} max={max(allc)} over {len(allc)} IBs")
spr = np.sort(exp[exp > 0])
rep_e = np.sort(exp[exp < 0])
print(f"exp springs (e>0, n={len(spr)}): {np.round(spr, 2)}")
print(f"exp reps    (e<0, n={len(rep_e)}): {np.round(rep_e, 2)}  <- these get the 1/d repulsion")
print(f"seed Rg={rg(inp['pos']):.3f}   checker_out Rg={rg(d['checker_out']):.3f} (polish starts here)")

# --- polish trajectory from the checker output at rising round caps ----------
polish_in = [
    {
        "pos": np.asarray(d["checker_out"], np.float32),
        "exp_dist": exp,
        "step_size": float(inp["step_size"]),
    }
]
caps = (
    [int(c) for c in sys.argv[2].split(",")]
    if len(sys.argv) > 2
    else [200, 1000, 3000, 6000, 10000]
)
print("\n   cap     Rg      score    time")
prev = None
for cap in caps:
    t = time.perf_counter()
    ((score, pos),) = mc_jax.mc_arcs_jax_batch(polish_in, s, max_iters=cap)
    r = rg(pos)
    print(f"  {cap:>6}  {r:6.1f}  {score:7.4f}  ({time.perf_counter() - t:.1f}s)", flush=True)
    prev = r

# --- parity: full sequential from the SEED (no checker) - is Rg the true min? ---
print("\nfull sequential from the SEED (no checker, to convergence):")
seed_in = [{"pos": np.asarray(inp["pos"], np.float32), "exp_dist": exp, "step_size": float(inp["step_size"])}]
_t = time.perf_counter()
((sscore, spos),) = mc_jax.mc_arcs_jax_batch(seed_in, s)
seq_rg = rg(spos)
print(f"  Rg={seq_rg:.3f}  score={sscore:.4f}  ({time.perf_counter() - _t:.1f}s)")

# --- full hybrid: FRESH threaded checker -> polish (tests the whole pipeline bounds) ---
from gnome3d.mc.jax.arcs_checker import mc_arcs_checker_jax_batch  # noqa: E402

_seed_in = [{"pos": np.asarray(inp["pos"], np.float32), "exp_dist": exp, "step_size": float(inp["step_size"])}]
((_cs, ck_pos),) = mc_arcs_checker_jax_batch(_seed_in, s)
((_hs, hy_pos),) = mc_arcs_jax_batch(
    [{"pos": np.asarray(ck_pos, np.float32), "exp_dist": exp, "step_size": float(inp["step_size"])}], s
)
print(f"\nfull hybrid (fresh checker -> polish):  checker Rg={rg(ck_pos):.3f} -> polish Rg={rg(hy_pos):.3f}"
      f"   (old unbounded: {rg(d['checker_out']):.0f} -> 515)")

print("\nVERDICT:")
print(f"  hybrid Rg={prev:.1f} ~= sequential Rg={seq_rg:.1f}  => faithful min, slow only b/c flat => CAP IT")
print("  hybrid >> sequential                         => checker seeds a blown-up basin => HYBRID BUG")
print("  both huge vs the contact targets above       => under-constrained IB, EV-dominated => upstream")
