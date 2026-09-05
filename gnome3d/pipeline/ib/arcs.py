"""
ARCS stage: position anchors via arc-spring MC.

Pure port of `Solver._reconstruct_cluster_arcs`: the same restart loop (noise the
anchor seeds, run `mc_arcs`, keep the best of `steps_arcs`), but reading the
`Seeded` state instead of the cluster graph and seeding the RNG deterministically
from `Seeded.seed` (Python's `random` for the initial noise + numba's RNG for the
kernel) so the result is reproducible and order-independent - the property the
batched/ensemble paths need.

Serial runner = the numba backend (`mc_arcs_numba`).  The batched JAX runner
(`mc_arcs_jax_batch`) is registered separately later.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gnome3d.pipeline.ib.buckets import batch_bucket
from gnome3d.pipeline.ib.floor import bond_scale, genomic_floor_matrix
from gnome3d.pipeline.stage import Problem, Result, StageKind
from gnome3d.pipeline.state import Arced, Seeded, State
from gnome3d.util import add_movable_noise_inplace, seed_rng

if TYPE_CHECKING:
    from gnome3d.settings import Settings
    from gnome3d.types import F32Array, F64Array


def _run(problem: Problem) -> Result:
    """Serial runner: anneal one IB's anchors.  Returns `(best_score, best_pos)`.

    Mirrors `Solver._reconstruct_cluster_arcs` exactly (initial per-anchor noise,
    `steps_arcs` restarts, best-of), with deterministic seeding added.
    """
    from gnome3d.mc import numba as mc_numba

    pos0: F32Array = problem["anchor_pos"]
    exp_dist = problem["exp_dist"]
    step = float(problem["step_size"])
    s = problem["settings"]
    seed = int(problem["seed"])

    # Deterministic per-IB RNG: Python `random` (initial noise) + numba (kernel).
    seed_rng(seed)
    mc_numba.seed_numba(seed)

    best_score = -1.0
    best: F32Array = pos0.copy()
    for _run_i in range(max(1, int(s.steps_arcs))):
        pos: F32Array = pos0.copy()
        add_movable_noise_inplace(pos, None, step)  # arcs noises ALL anchors
        score = mc_numba.mc_arcs_numba(pos, exp_dist, step, s)  # mutates pos in place
        if score < best_score or best_score < 0.0:
            best_score = score
            best = pos.copy()

    if s.use_genomic_floor:
        # Second pass with the genomic floor, calibrated on the first pass. See floor.py.
        floor = floor_for(problem, best, s)
        if floor is not None:
            pos = best.copy()
            best_score = mc_numba.mc_arcs_numba(
                pos, exp_dist, step, s, floor_mat=floor, temp=polish_temp(s)
            )
            best = pos
    return best_score, np.asarray(best, dtype=np.float32)


def polish_temp(s: Settings) -> float:
    """Starting temperature of the floor pass, a fraction of `max_temp`."""
    return float(s.max_temp) * float(s.genomic_floor_polish_temp)


def floor_for(problem: Problem, anchors: F32Array, s: Settings) -> F64Array | None:
    """The genomic floor matrix for one IB, scaled on the anchors a first anneal produced.
    None when the block has no consecutive anchor pair to calibrate on and no explicit
    scale."""
    scale = float(s.genomic_floor_scale)
    if scale <= 0.0:
        d_bond = bond_scale(anchors)
        if d_bond <= 0.0:
            return None
        scale = float(s.genomic_floor_factor) * d_bond
    return genomic_floor_matrix(
        problem["anchor_genomic"], problem["exp_dist"], scale, float(s.genomic_floor_exponent)
    )


def _batch_run(problems: list[Problem]) -> list[Result]:
    """Batched (JAX) runner: anneal a whole bucket of IBs' arcs in one vmapped
    kernel.  Each IB is fanned out to `steps_arcs` noised restarts (best kept),
    mirroring the serial loop.  Returns one `(score, pos)` per input problem.
    Lazy `mc_jax` import so the numba path never requires JAX."""
    from gnome3d.mc import jax as mc_jax

    s = problems[0]["settings"]
    n_restarts = max(1, int(s.steps_arcs))

    expanded: list[Problem] = []
    owner: list[int] = []
    for gi, prob in enumerate(problems):
        seed_rng(int(prob["seed"]))  # deterministic restart noise for this IB
        pos = prob["anchor_pos"]
        step = float(prob["step_size"])
        for _ in range(n_restarts):
            start = pos.copy()
            add_movable_noise_inplace(start, None, step)  # arcs noises ALL anchors
            expanded.append({"pos": start, "exp_dist": prob["exp_dist"], "step_size": step})
            owner.append(gi)

    results = mc_jax.mc_arcs_jax_batch(expanded, s)

    best: dict[int, Result] = {}
    for (score, final_pos), gi in zip(results, owner, strict=True):
        if gi not in best or score < best[gi][0]:
            best[gi] = (score, final_pos)
    out = [best[gi] for gi in range(len(problems))]

    if s.use_genomic_floor:
        # Second pass with the genomic floor, one launch, calibrated per IB on the first.
        polish: list[Problem] = []
        which: list[int] = []
        for gi, prob in enumerate(problems):
            floor = floor_for(prob, np.asarray(out[gi][1], dtype=np.float32), s)
            if floor is None:
                continue
            polish.append(
                {
                    "pos": np.asarray(out[gi][1], dtype=np.float32),
                    "exp_dist": prob["exp_dist"],
                    "step_size": float(prob["step_size"]),
                    "floor_mat": floor,
                    "seed": prob.get("seed"),
                }
            )
            which.append(gi)
        if polish:
            results = mc_jax.mc_arcs_jax_batch(polish, s, start_temp=polish_temp(s))
            for gi, res in zip(which, results, strict=True):
                out[gi] = res
    return out


class ArcsStage:
    """`Seeded -> Arced`."""

    kind = StageKind.ARCS

    def bucket(self, inputs: tuple[State, ...]) -> int:
        return int(inputs[0].anchor_seed_pos.shape[0])  # type: ignore[attr-defined]

    def batch_key(self, inputs: tuple[State, ...]) -> tuple[object, ...]:
        """Arcs has uniform energy terms, so the batch key is just the anchor
        shape-ladder bucket - all IBs in a bucket share one compiled kernel."""
        st = inputs[0]
        assert isinstance(st, Seeded)
        return (batch_bucket(int(st.anchor_seed_pos.shape[0]), st.settings),)

    def to_problem(self, inputs: tuple[State, ...]) -> Problem:
        st = inputs[0]
        assert isinstance(st, Seeded)
        return {
            "anchor_pos": st.anchor_seed_pos,
            "exp_dist": st.exp_dist,
            "step_size": st.step_size_arcs,
            "settings": st.settings,
            "seed": st.seed,
            "anchor_genomic": st.anchor_genomic,
        }

    def apply(self, inputs: tuple[State, ...], result: Result) -> State:
        st = inputs[0]
        assert isinstance(st, Seeded)
        _score, best_pos = result
        return Arced(**vars(st), anchor_pos=best_pos)
