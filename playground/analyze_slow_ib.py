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

import pickle
import sys
import time

import numpy as np

from gnome3d.mc import jax as mc_jax

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/slow_ib.pkl"
d = pickle.load(open(path, "rb"))
inp = d["input"]
s = d["settings"]
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
print(f"checker_out Rg={rg(d['checker_out']):.3f}  (the polish starts here)")

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
print("\n   cap      Rg       score      dRg     time")
prev = None
for cap in caps:
    t = time.perf_counter()
    ((score, pos),) = mc_jax.mc_arcs_jax_batch(polish_in, s, max_iters=cap)
    r = rg(pos)
    drg = "  --  " if prev is None else f"{r - prev:+.3f}"
    print(f"  {cap:>6}   {r:.3f}   {score:9.4f}   {drg:>6}  ({time.perf_counter() - t:.1f}s)", flush=True)
    prev = r

print("\nVERDICT: Rg climbing toward the full-polish target across caps = HARD IB (cap it);")
print("         Rg stuck near the checker baseline or oscillating  = PATHOLOGY (fix it).")
