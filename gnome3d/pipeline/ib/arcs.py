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
from gnome3d.pipeline.stage import Problem, Result, StageKind
from gnome3d.pipeline.state import Arced, Seeded, State
from gnome3d.util import add_movable_noise_inplace, seed_rng

if TYPE_CHECKING:
    from gnome3d.settings import Settings
    from gnome3d.types import F32Array


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

    # Anneal or solve. The solver minimises the same energy directly and reaches the same
    # minimum far faster, because the landscape is a funnel. Restarts, noise and best-of are
    # the same either way, so an ensemble still comes from the perturbed starts.
    solver = arcs_solver(s)

    best_score = -1.0
    best: F32Array = pos0.copy()
    for _run_i in range(max(1, int(s.steps_arcs))):
        pos: F32Array = pos0.copy()
        add_movable_noise_inplace(pos, None, step)  # arcs noises ALL anchors
        if solver == "lbfgs":
            from gnome3d.mc.numba.arcs_solver import solve_arcs  # noqa: PLC0415

            score, pos = solve_arcs(pos, exp_dist, s)
        else:
            score = mc_numba.mc_arcs_numba(pos, exp_dist, step, s)  # mutates pos in place
        if score < best_score or best_score < 0.0:
            best_score = score
            best = pos.copy()

    return best_score, np.asarray(best, dtype=np.float32)


def arcs_solver(s: Settings) -> str:
    """The stage's solver name, validated. An unrecognised name is refused rather than falling
    through to the annealer, which would run the wrong stage and report nothing."""
    name = str(s.arcs_solver).strip().lower()
    if name not in ("mc", "lbfgs"):
        raise ValueError(f"[simulation_arcs] solver must be mc or lbfgs, got {s.arcs_solver!r}")
    return name


def _batch_run(problems: list[Problem]) -> list[Result]:
    """Batched (JAX) runner: anneal a whole bucket of IBs' arcs in one vmapped
    kernel.  Each IB is fanned out to `steps_arcs` noised restarts (best kept),
    mirroring the serial loop.  Returns one `(score, pos)` per input problem.
    Lazy `mc_jax` import so the numba path never requires JAX.

    There is no solver here, only the JAX annealer, so a run that asked for one is refused. It
    would otherwise anneal and look like it had solved."""
    s = problems[0]["settings"]
    if arcs_solver(s) != "mc":
        raise NotImplementedError(
            f"[simulation_arcs] solver = {s.arcs_solver} needs "
            "[simulation_backend] mc_executor_arcs = serial or threaded; "
            "the batch executor has no solver"
        )

    from gnome3d.mc import jax as mc_jax  # noqa: PLC0415

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
