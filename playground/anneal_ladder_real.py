"""Does a temperature ladder buy anything, on real captured arcs blocks.

`delta_temp` is applied per MC step, so at the reference 0.9999 over 50,000 steps a round the
temperature falls 148 times each round: from 5.0 it is 3e-2 after one round and 1e-13 after six,
and runs take seventy to ninety. Every stage anneals for about three rounds and descends greedily
for the rest.

This settles whether that matters, which decides whether population annealing or parallel
tempering have anything to offer: both exist to traverse a rough landscape with a temperature
ladder, and neither can help if the ladder does not.

One chain, a fixed step budget so the work is matched, only the cooling rate changing. Blocks
come from `capture_arcs.py`, which grabs the real expected-distance matrices and starting anchors
as the arcs stage begins. Synthetic matrices have misled here before, so this uses neither.

    python playground/anneal_ladder_real.py <arcs_real.pkl> [steps]
"""
import os, sys, math, pickle
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np
from gnome3d.mc.numba import seed_numba
from gnome3d.mc.numba.arcs import mc_arcs_numba
from gnome3d.settings import Settings

blocks = pickle.load(open(sys.argv[1], "rb"))
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000
SEEDS = (5, 6, 7)
T0 = 5.0

def dt_for(t_end, steps): return math.exp(math.log(t_end / T0) / steps)
def rg(p): return float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean()))

for pos0, exp, step in blocks[:3]:
    n = pos0.shape[0]
    arcs = int((exp > 1e-6).sum() // 2)
    print(f"\nREAL block N={n}, {arcs} arc pairs "
          f"({arcs / (n * (n - 1) / 2):.2%} of pairs), one round of {STEPS:,} steps")
    print(f"  {'ladder':>24s} {'delta_temp':>12s} {'energy':>12s} {'vs ref':>7s} {'Rg':>8s}")
    base = None
    for label, dt in (("reference 0.9999", 0.9999),
                      ("to 0.01 over a tenth", dt_for(0.01, STEPS // 10)),
                      ("to 0.01 over the run", dt_for(0.01, STEPS)),
                      ("to 0.1 over the run", dt_for(0.1, STEPS)),
                      ("no temperature at all", 0.0)):
        E, G = [], []
        for seed in SEEDS:
            s = Settings()
            s.arcs_repulsion_cutoff_factor = 3.0
            s.use_confinement = True; s.confinement_apply_to_arcs = True
            s.max_temp = T0 if dt > 0 else 0.0
            s.dt_temp = dt if dt > 0 else 1.0
            s.jump_scale = 50.0; s.jump_coef = 20.0
            s.mc_stop_steps = STEPS
            s.mc_stop_improvement = 0.0; s.mc_stop_successes = 10**9
            p = pos0.copy(); seed_numba(seed); np.random.seed(seed)
            E.append(mc_arcs_numba(p, exp, step, s)); G.append(rg(p))
        e = float(np.mean(E))
        if base is None: base = e
        print(f"  {label:>24s} {dt:>12.7f} {e:>12.1f} {e/base:>6.3f}x {np.mean(G):>8.3f}", flush=True)
