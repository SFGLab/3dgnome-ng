"""A/B for routing DISTANCE ESTIMATION through the checker smooth kernel.

Estimation builds the dense distance TARGET (avg_dist) the final smooth then chases.
The worry: the checker's ~3.5% compaction shrinks that target, and the final smooth
(also checker) chases the shrunk target AND adds its own compaction -> the two compound.

This measures the FIRST half directly: how much smaller is the estimated target when the
dry-smooth trials run on the checker vs the current old-JAX kernel.  Same expand+accumulate
as estimate_dist._batch_run, just swapping the kernel.  Ratio ~1.0 => no compounding worry.
"""

import pickle
import time

import numpy as np

import gnome3d.mc.jax.smooth_checker as sc
from gnome3d.mc import jax as mc_jax
from gnome3d.pipeline.ib.estimate_dist import _accumulate_pairwise_dist
from gnome3d.settings import Settings
from gnome3d.util import add_movable_noise_inplace, seed_rng

caps = pickle.load(open("/tmp/smooth_ibs.pkl", "rb"))
wh = [c for c in caps if 120 <= c["pos"].shape[0] <= 1300]
wh.sort(key=lambda c: c["pos"].shape[0])
tests = wh[:12]
print(f"{len(tests)} IBs in [120,1300] (of {len(caps)} captured)")

s = Settings()
s.load_ini("data/GM12878/config.ini")
s.mc_executor_jax_bucket_shapes = True
n_reps = int(s.subanchor_estimate_replicates)
n_steps = int(s.subanchor_estimate_steps)
per_ib = n_reps * n_steps


def gyration(p):
    c = p.mean(axis=0)
    return float(np.sqrt(((p - c) ** 2).sum(axis=1).mean()))


def run_estimate(kernel_fn, pos, fixed, dtn, step, seed):
    """Replicates estimate_dist._batch_run for ONE IB: expand per_ib dry trials,
    run them, keep the best per replicate, accumulate its pairwise distances."""
    seed_rng(seed)
    expanded = []
    for _ in range(per_ib):
        start = pos.copy().astype(np.float32)
        add_movable_noise_inplace(start, fixed, step)
        expanded.append(
            {"pos": start, "dtn": dtn.astype(np.float32), "fixed": fixed, "step_size": step,
             "heat_dist": None, "char_orientations": None,
             "anchor_neighbors": None, "anchor_neighbor_weights": None}
        )
    results = kernel_fn(expanded, s)
    n = len(pos)
    avg = np.zeros((n, n), np.float32)
    best_rgs = []
    for rep in range(n_reps):
        sl = results[rep * n_steps : (rep + 1) * n_steps]
        best = sl[int(np.argmin([r[0] for r in sl]))][1]
        best_rgs.append(gyration(best))
        _accumulate_pairwise_dist(avg, best)
    avg /= n_reps
    return avg, float(np.mean(best_rgs))


print(f"estimation avg_dist: CHECKER vs old-JAX  (per_ib={per_ib} dry smooths)")
print(f'{"B":>5} {"meanDist chk/old":>17} {"bestRg chk/old":>15} {"chk_s":>7}')
d_ratios = []
for c in tests:
    pos = np.asarray(c["pos"], np.float32)
    dtn = np.asarray(c["dtn"], np.float32)
    fixed = np.asarray(c["fixed"], np.bool_)
    B = pos.shape[0]
    step = float(np.median(dtn[dtn > 1e-6])) if (dtn > 1e-6).any() else 1.0
    avg_old, rg_old = run_estimate(mc_jax.mc_smooth_jax_batch, pos, fixed, dtn, step, 1)
    t0 = time.perf_counter()
    avg_chk, rg_chk = run_estimate(sc.mc_smooth_checker_jax_batch, pos, fixed, dtn, step, 1)
    tchk = time.perf_counter() - t0
    m = ~np.eye(B, dtype=bool)
    d_ratio = avg_chk[m].mean() / max(avg_old[m].mean(), 1e-9)
    d_ratios.append(d_ratio)
    print(f"{B:>5} {d_ratio:>17.4f} {rg_chk / max(rg_old, 1e-9):>15.4f} {tchk:>7.1f}", flush=True)
arr = np.array(d_ratios)
print(f"\nmeanDist chk/old: mean={arr.mean():.4f} std={arr.std():.4f} "
      f"min={arr.min():.4f} max={arr.max():.4f}  (n={len(arr)})")
print("SAFE if mean ~ 1.0 (no SYSTEMATIC target shrink; per-IB spread is estimation noise)")
