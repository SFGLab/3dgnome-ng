"""Validate the batched IB heat-estimate path against the sequential one.

The opt-in `subanchor_batch_trials` fast path (solver.py::_build_heat_dist_subanchor)
runs all n_reps*n_steps anneals as ONE vmapped mc_smooth_jax call instead of a
sequential python loop. It diverges intentionally (shared best-of-K convergence
stop, different per-chain RNG), so the bar is STATISTICAL equivalence of the
resulting avg_dist matrix — not bit-identity.

This reproduces both code paths at the avg_dist level (the exact computation the
solver does) on a synthetic region and compares:
  * max / mean absolute difference of avg_dist
  * mean relative difference
  * KS distance on the off-diagonal distance distribution

PASS heuristic: KS distance small (<~0.05) and mean relative diff small (<~2%).
These are stochastic estimates averaged over reps, so expect small nonzero diffs.

Run on the CUDA box:
    pip install "jax[cuda12]"
    python playground/validate_batched_ib.py            # N=800 default
    python playground/validate_batched_ib.py 2000 4 6   # N, n_reps, n_steps
"""

from __future__ import annotations

import sys

import numpy as np
from scipy.stats import ks_2samp

sys.path.insert(0, ".")
from gnome3d.mc_jax import mc_smooth_jax  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402
from gnome3d.util import random_vector_np  # noqa: E402


def make_region(n, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.cumsum(rng.normal(0, 1, size=(n, 3)), axis=0).astype(np.float32)
    dtn = np.linalg.norm(np.diff(pos, axis=0), axis=1).astype(np.float32)
    dtn = np.append(dtn, dtn[-1]).astype(np.float32)
    fixed = np.zeros(n, dtype=bool)
    fixed[::12] = True
    return pos, dtn, fixed


def avg_dist_sequential(pos, dtn, fixed, step_size, s, n_reps, n_steps, run_tag=""):
    n = len(pos)
    s.mc_smooth_chains = 1
    avg = np.zeros((n, n), dtype=np.float64)
    for rep in range(n_reps):
        best_score = -1.0
        best_pos = pos.copy()
        for step in range(n_steps):
            pt = pos.copy()
            for i in range(n):
                if not fixed[i]:
                    pt[i] += random_vector_np(step_size)
            # Distinct label per trial => distinct seed_offset => distinct
            # internal MC RNG, matching production (mc_smooth passes a unique
            # "rep/step" label) AND the batched path's per-chain RNG diversity.
            # run_tag makes run A vs run B independent in internal RNG too.
            label = f"{run_tag} r{rep} s{step}"
            score = mc_smooth_jax(pt, dtn, fixed, step_size, s, label=label)  # mutates pt
            if score < best_score or best_score < 0.0:
                best_score = score
                best_pos = pt.copy()
        diff = best_pos[:, None, :] - best_pos[None, :, :]
        avg += np.sqrt((diff * diff).sum(axis=2))
    return avg / n_reps


def avg_dist_batched(pos, dtn, fixed, step_size, s, n_reps, n_steps):
    n = len(pos)
    n_trials = n_reps * n_steps
    starts = np.empty((n_trials, n, 3), dtype=np.float32)
    b = 0
    for _rep in range(n_reps):
        for _step in range(n_steps):
            pt = pos.copy()
            for i in range(n):
                if not fixed[i]:
                    pt[i] += random_vector_np(step_size)
            starts[b] = pt
            b += 1
    scores, finals = mc_smooth_jax(pos, dtn, fixed, step_size, s, pos_batch=starts, return_all=True)
    scores = np.asarray(scores).reshape(n_reps, n_steps)
    finals = np.asarray(finals).reshape(n_reps, n_steps, n, 3)
    avg = np.zeros((n, n), dtype=np.float64)
    for rep in range(n_reps):
        bt = int(np.argmin(scores[rep]))
        best = finals[rep, bt]
        diff = best[:, None, :] - best[None, :, :]
        avg += np.sqrt((diff * diff).sum(axis=2))
    return avg / n_reps


def main(n=800, n_reps=4, n_steps=4):
    pos, dtn, fixed = make_region(n)
    step_size = 0.5

    def fresh_settings():
        s = Settings()
        s.mc_backend = "jax"
        s.mc_backend_apply_to_smooth = True
        s.use_excluded_volume = True
        s.exclusion_apply_to_smooth = True
        s.mc_stop_steps_smooth = 2000
        return s

    print(f"N={n}  n_reps={n_reps}  n_steps={n_steps}  (trials={n_reps * n_steps})")
    iu = np.triu_indices(n, k=1)

    def compare(x, y):
        a, b = x[iu], y[iu]
        abs_d = np.abs(a - b)
        rel = abs_d / np.maximum(np.abs(a), 1e-9)
        return {
            "ks": float(ks_2samp(a, b).statistic),
            "max_abs": float(abs_d.max()),
            "mean_abs": float(abs_d.mean()),
            "mean_rel": float(rel.mean()),
            "mean_shift": float(b.mean() - a.mean()),
        }

    # CONTROL: the sequential path against ITSELF with two different RNG seeds.
    # This is the intrinsic noise floor of the (stochastic, n_reps-averaged)
    # estimate — the irreducible run-to-run scatter that has nothing to do with
    # batching.  The batched-vs-sequential difference is only meaningful relative
    # to this floor.
    np.random.seed(1234)
    seqA = avg_dist_sequential(
        pos, dtn, fixed, step_size, fresh_settings(), n_reps, n_steps, run_tag="A"
    )
    np.random.seed(9999)
    seqB = avg_dist_sequential(
        pos, dtn, fixed, step_size, fresh_settings(), n_reps, n_steps, run_tag="B"
    )
    # TEST: batched, seeded identically to seqA so the noised starts match;
    # only the batched kernel + shared-convergence stop differ.
    np.random.seed(1234)
    bat = avg_dist_batched(pos, dtn, fixed, step_size, fresh_settings(), n_reps, n_steps)

    floor = compare(seqA, seqB)
    test = compare(seqA, bat)

    print(f"\navg_dist off-diagonal ({len(iu[0]):,} pairs)")
    print(f"{'metric':<14}{'seq-vs-seq (floor)':>20}{'seq-vs-batch (test)':>22}")
    for k, lbl in [
        ("ks", "KS distance"),
        ("max_abs", "max abs diff"),
        ("mean_abs", "mean abs diff"),
        ("mean_rel", "mean rel diff"),
        ("mean_shift", "mean shift"),
    ]:
        f, t = floor[k], test[k]
        fs = f"{f * 100:.2f}%" if "rel" in k else f"{f:.4f}"
        ts = f"{t * 100:.2f}%" if "rel" in k else f"{t:.4f}"
        print(f"{lbl:<14}{fs:>20}{ts:>22}")

    # PASS iff batching adds no more divergence than a fresh sequential seed
    # (within 1.5x slack on the noisy per-pair metrics).
    ok = (
        test["ks"] <= max(0.02, floor["ks"] * 1.5)
        and test["mean_rel"] <= floor["mean_rel"] * 1.5 + 0.005
        and abs(test["mean_shift"]) <= abs(floor["mean_shift"]) + 0.1 * seqA[iu].mean()
    )
    print(
        f"\n{'PASS' if ok else 'REVIEW'}: batched divergence "
        f"{'within the intrinsic seq-vs-seq noise floor' if ok else 'EXCEEDS the noise floor — inspect'}"
    )


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:]]
    main(*args) if args else main()
