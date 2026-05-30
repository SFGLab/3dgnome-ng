"""Validate region-batched smooth-MC == sequential single-problem.

`mc_smooth_jax_batch` anneals K DIFFERENT IBs in one vmapped kernel (per-chain
convergence) instead of the serial IB loop's K separate `mc_smooth_jax` calls.
Because the two paths seed RNG differently (batched: one base key split K ways;
sequential: a per-IB scope seed), the real-bead trajectories diverge under
strict-acceptance f32 chaos — so the bar is STATISTICAL equivalence of the
output ensemble, not bit-identity (same discipline as the batched-IB estimate).

What this MUST catch (the reason it exists): shape / in_axes / per-chain
convergence bugs in the multi-problem kernel.  If batched runs clean, returns
finite per-IB scores at the right shapes, and lands in the same ballpark as
sequential, the plumbing is correct.

Runs on CPU-JAX (slow but valid) or GPU:
    python playground/bench/validate_region_batch.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, ".")
from gnome3d import log  # noqa: E402
from gnome3d.mc_jax import mc_smooth_jax, mc_smooth_jax_batch  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402


def make_ib(n, seed=0):
    """One IB's smooth inputs, mirroring solver._reconstruct_cluster_smooth:
    pos/dtn/fixed + dense subanchor heat + CTCF orientation (bead-indexed)."""
    rng = np.random.default_rng(seed)
    pos = np.cumsum(rng.normal(0, 1, size=(n, 3)), axis=0).astype(np.float32)
    dtn = np.linalg.norm(np.diff(pos, axis=0), axis=1).astype(np.float32)
    dtn = np.append(dtn, dtn[-1]).astype(np.float32)
    fixed = np.zeros(n, dtype=bool)
    fixed[::8] = True  # ~12% anchors
    sep = np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(np.float32)
    heat = 1.0 + np.sqrt(sep)
    heat[rng.random((n, n)) < 0.3] = 0.0
    heat = ((heat + heat.T) / 2).astype(np.float32)
    np.fill_diagonal(heat, 0.0)
    # orientation: bead-indexed chars (length n), CSR neighbours per anchor
    anchor_beads = np.where(fixed)[0]
    n_anchors = len(anchor_beads)
    chars = ["N"] * n
    for ai, bi in enumerate(anchor_beads):
        chars[int(bi)] = "L" if ai % 2 == 0 else "R"
    nbrs, wts = {}, {}
    hw = 3
    for k in range(n_anchors):
        nb = [j for j in range(max(0, k - hw), min(n_anchors, k + hw + 1)) if j != k]
        nbrs[k] = nb
        wts[k] = [1.0] * len(nb)
    step_size = float(dtn.mean()) * 5.0
    return {
        "pos": pos,
        "dtn": dtn,
        "fixed": fixed,
        "heat_dist": heat,
        "char_orientations": chars,
        "anchor_neighbors": nbrs,
        "anchor_neighbor_weights": wts,
        "step_size": step_size,
    }


def settings():
    s = Settings()
    s.mc_backend = "jax"
    s.mc_backend_apply_to_smooth = True
    s.use_excluded_volume = True
    s.exclusion_apply_to_smooth = True
    s.use_confinement = True
    s.confinement_apply_to_smooth = True
    s.use_subanchor_heatmap = True
    s.use_ctcf_motif = True
    s.jax_bucket_shapes = True
    s.jax_precompile_buckets = False  # skip eager precompile for the test
    s.mc_smooth_chains = 1
    s.mc_stop_steps_smooth = 500  # small batches -> fast convergence on CPU
    if not hasattr(s, "motif_weight"):
        s.motif_weight = 50.0
    return s


def seq_one(p, s):
    """Sequential single-problem call (today's path), under a per-IB scope so
    the seed differs per IB exactly like production."""
    pos = p["pos"].copy()
    with log.scope(f"ib n={p['pos'].shape[0]}"):
        score = mc_smooth_jax(
            pos,
            p["dtn"],
            p["fixed"],
            p["step_size"],
            s,
            char_orientations=p["char_orientations"],
            anchor_neighbors=p["anchor_neighbors"],
            anchor_neighbor_weights=p["anchor_neighbor_weights"],
            heat_dist=p["heat_dist"],
        )
    return score, pos


def main():
    log.setup(1)
    s = settings()
    # A few IBs of different sizes that share one shape bucket (B=512 here).
    sizes = [40, 48, 56, 64]
    problems = [make_ib(n, seed=i) for i, n in enumerate(sizes)]

    print("sequential single-problem (one mc_smooth_jax call per IB)...")
    seq = [seq_one(p, s) for p in problems]

    print("region-batched (one mc_smooth_jax_batch call for all IBs)...")
    bat = mc_smooth_jax_batch([dict(p) for p in problems], s)

    print(f"\n{'N':>5} {'seq score':>12} {'batch score':>12} {'rel diff':>10} {'shape ok':>9}")
    print("-" * 52)
    ok_all = True
    for n, (s_sc, _s_pos), (b_sc, b_pos) in zip(sizes, seq, bat, strict=True):
        rel = abs(s_sc - b_sc) / max(abs(s_sc), 1e-9)
        shape_ok = b_pos.shape == (n, 3) and np.all(np.isfinite(b_pos)) and np.isfinite(b_sc)
        ok_all = ok_all and shape_ok
        print(f"{n:>5} {s_sc:>12.4f} {b_sc:>12.4f} {rel:>10.2%} {'ok' if shape_ok else 'BUG':>9}")

    seq_scores = np.array([x[0] for x in seq])
    bat_scores = np.array([x[0] for x in bat])
    print(
        f"\nseq mean={seq_scores.mean():.3f}  batch mean={bat_scores.mean():.3f}  "
        f"ensemble rel diff={abs(seq_scores.mean() - bat_scores.mean()) / max(abs(seq_scores.mean()), 1e-9):.2%}"
    )
    print()
    if not ok_all:
        print("SHAPE/FINITE BUG -> the multi-problem kernel is wired wrong; inspect.")
        sys.exit(1)
    # statistical bar: ensemble means within ~20% (strict-acceptance chaos +
    # per-chain-vs-best convergence differences); the point of this script is
    # the shape/plumbing gate above.
    ens = abs(seq_scores.mean() - bat_scores.mean()) / max(abs(seq_scores.mean()), 1e-9)
    if ens < 0.25:
        print("PASS: batched runs clean, finite, right shapes, ensemble within noise.")
        sys.exit(0)
    print(f"WARN: ensemble means differ by {ens:.1%} (>25%) — inspect convergence semantics.")
    sys.exit(2)


def bench(n=1607, k=16, stop_steps=2000):
    """Time the real win on GPU: K sequential single-problem calls (today's IB
    loop) vs ONE region-batched call, at production N with the full term set.
    This is the go/no-go before the solver restructure."""
    log.setup(0)
    s = settings()
    s.mc_stop_steps_smooth = stop_steps
    problems = [make_ib(n, seed=i) for i in range(k)]

    # warm compile (both paths) so timing excludes one-time XLA compile
    seq_one(problems[0], s)
    mc_smooth_jax_batch([dict(problems[0])], s)

    t0 = time.perf_counter()
    for p in problems:
        seq_one(p, s)
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    mc_smooth_jax_batch([dict(p) for p in problems], s)
    t_bat = time.perf_counter() - t0

    print(
        f"\nREAL-N region-batch  N={n}  K={k}  steps/batch={stop_steps}  terms=chain+EV+heat+orient+conf"
    )
    print(f"  sequential ({k} calls): {t_seq:8.2f}s   ({t_seq / k * 1000:.1f} ms/IB)")
    print(f"  batched    (1 call)   : {t_bat:8.2f}s   ({t_bat / k * 1000:.1f} ms/IB)")
    print(f"  speedup               : {t_seq / max(t_bat, 1e-9):6.1f}x")


if __name__ == "__main__":
    if "--bench" in sys.argv:
        rest = [a for a in sys.argv[1:] if not a.startswith("--")]
        n = int(rest[0]) if rest else 1607
        k = int(rest[1]) if len(rest) > 1 else 16
        bench(n=n, k=k)
    else:
        main()
