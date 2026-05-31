"""
HEAT_DIST stage: estimate the subanchor contact-distance target matrix.

Pure port of `Solver._build_heat_dist_subanchor` = `_estimate_avg_dist` (dry
smooth-MC passes to get average pairwise distances) + `_heat_dist_from_avg`
(scale high-contact pairs down by `influence`).  Reads the `Densified` state,
seeds deterministically from `Seeded.seed`.  Only present in a chain when the
skeleton's sparse-signal early-out kept it (`IBSeed.wants_heat`).

Serial runner = numba dry smooth (`mc_smooth_numba` with no heat/orientation).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

from gnome3d.mc import numba as mc_numba
from gnome3d.pipeline.stage import Problem, Result, StageKind
from gnome3d.pipeline.state import Densified, HeatReady, State
from gnome3d.util import random_vector_np

if TYPE_CHECKING:
    from gnome3d.settings import Settings
    from gnome3d.types import BoolArray, F32Array, F64Array

_SEED_SALT = 1  # distinct from ARCS(0)/SMOOTH(2) so the noise streams don't correlate


def _estimate_avg_dist(
    pos: F32Array, fixed: BoolArray, dtn: F32Array, step: float, s: Settings, seed: int
) -> F64Array:
    """Average pairwise bead distances over dry smooth-MC passes.  Port of
    `Solver._estimate_avg_dist` (reference sequential path): per replicate, run
    `n_steps` dry passes from pos+noise, keep the best, accumulate its pairwise
    distances."""
    n = len(pos)
    n_reps = int(s.subanchor_estimate_replicates)
    n_steps = int(s.subanchor_estimate_steps)
    random.seed(seed)
    mc_numba.seed_numba(seed)

    avg_dist: F64Array = np.zeros((n, n), dtype=np.float64)
    for _rep in range(n_reps):
        rep_best_score = -1.0
        rep_best_pos: F32Array = pos.copy()
        for _step in range(n_steps):
            pos_trial: F32Array = pos.copy()
            for i in range(n):
                if not fixed[i]:
                    pos_trial[i] += random_vector_np(step)
            score = mc_numba.mc_smooth_numba(pos_trial, dtn, fixed, step, s)  # dry: no heat/orn
            if score < rep_best_score or rep_best_score < 0.0:
                rep_best_score = score
                rep_best_pos = pos_trial.copy()
        diff = rep_best_pos[:, None, :] - rep_best_pos[None, :, :]
        avg_dist += np.sqrt((diff * diff).sum(axis=2))
    avg_dist /= n_reps
    return avg_dist


def _heat_dist_from_avg(avg_dist: F64Array, subanchor_heat_raw: F64Array, s: Settings) -> F64Array | None:
    """Turn avg distances + raw subanchor heat into the expected-distance target
    (Reference createExpectedDistSubanchorHeatmap): high-contact pairs scaled
    down by `influence`.  None when the heatmap is empty.  Port of
    `Solver._heat_dist_from_avg` (sans the INFO logging/counters)."""
    n = avg_dist.shape[0]
    avg_heat = float(subanchor_heat_raw.mean())
    if avg_heat < 1e-6:
        return None
    influence = float(s.subanchor_heatmap_influence)
    heat_dist: F64Array = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            s_val = (subanchor_heat_raw[i, j] / avg_heat) * influence
            if s_val > 1.0:
                s_val = 1.0
            target = avg_dist[i, j] * (1.0 - s_val)
            heat_dist[i, j] = target
            heat_dist[j, i] = target
    return heat_dist


def _run(problem: Problem) -> Result:
    """Serial runner: estimate avg_dist (dry smooth) then build the target matrix."""
    avg = _estimate_avg_dist(
        problem["pos"], problem["fixed"], problem["dtn"], problem["step_size"],
        problem["settings"], problem["seed"],
    )
    return _heat_dist_from_avg(avg, problem["subanchor_heat_raw"], problem["settings"])


def _batch_run(problems: list[Problem]) -> list[Result]:
    """Batched (JAX) runner: run every IB's dry-smooth trials in one vmapped
    kernel (reusing `mc_smooth_jax_batch` — no heat/orientation), then build each
    IB's target matrix.  Mirrors `JaxSolver._batched_heat_dist`.  Returns one
    heat_dist (or None) per input problem, in order.  Lazy `mc_jax` import."""
    from gnome3d.mc import jax as mc_jax

    s = problems[0]["settings"]
    n_reps = int(s.subanchor_estimate_replicates)
    n_steps = int(s.subanchor_estimate_steps)
    per_ib = n_reps * n_steps

    expanded: list[Problem] = []
    spans: list[int] = []
    for prob in problems:
        spans.append(len(expanded))
        random.seed(int(prob["seed"]))
        pos = prob["pos"]
        fixed = prob["fixed"]
        step = float(prob["step_size"])
        n = len(pos)
        for _ in range(per_ib):
            start = pos.copy()
            for i in range(n):
                if not fixed[i]:
                    start[i] += random_vector_np(step)
            expanded.append(
                {
                    "pos": start,
                    "dtn": prob["dtn"],
                    "fixed": fixed,
                    "step_size": step,
                    "heat_dist": None,  # dry: chain+EV+conf only
                    "char_orientations": None,
                    "anchor_neighbors": None,
                    "anchor_neighbor_weights": None,
                }
            )

    results = mc_jax.mc_smooth_jax_batch(expanded, s)

    out: list[Result] = []
    for gi, prob in enumerate(problems):
        n = len(prob["pos"])
        base = spans[gi]
        avg_dist: F64Array = np.zeros((n, n), dtype=np.float64)
        for rep in range(n_reps):
            rep_slice = results[base + rep * n_steps : base + (rep + 1) * n_steps]
            scores = [r[0] for r in rep_slice]
            best_pos = rep_slice[int(np.argmin(scores))][1]
            diff = best_pos[:, None, :] - best_pos[None, :, :]
            avg_dist += np.sqrt((diff * diff).sum(axis=2))
        avg_dist /= n_reps
        out.append(_heat_dist_from_avg(avg_dist, prob["subanchor_heat_raw"], s))
    return out


class HeatDistStage:
    """`Densified -> HeatReady`."""

    kind = StageKind.HEAT_DIST

    def bucket(self, inputs: tuple[State, ...]) -> int:
        return int(inputs[0].pos.shape[0])  # type: ignore[attr-defined]

    def to_problem(self, inputs: tuple[State, ...]) -> Problem:
        st = inputs[0]
        assert isinstance(st, Densified)
        return {
            "pos": st.pos,
            "fixed": st.fixed,
            "dtn": st.dtn,
            "step_size": st.step_size_smooth,
            "subanchor_heat_raw": st.subanchor_heat_raw,
            "settings": st.settings,
            "seed": (st.seed + _SEED_SALT) & 0x7FFFFFFF,
        }

    def apply(self, inputs: tuple[State, ...], result: Result) -> State:
        st = inputs[0]
        assert isinstance(st, Densified)
        return HeatReady(**vars(st), heat_dist=result)
