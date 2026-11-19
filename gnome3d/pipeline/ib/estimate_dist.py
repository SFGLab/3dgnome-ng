"""The ESTIMATE_DIST stage estimates the subanchor distance target matrix.

Dry smooth passes give the average pair distances and pairs with many contacts are scaled
down by `influence`. The stage is in a chain only when `IBSeed.wants_heat` is set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange  # type: ignore[reportMissingTypeStubs]

from gnome3d import log
from gnome3d.pipeline.ib.buckets import batch_bucket
from gnome3d.pipeline.stage import Problem, Result, StageKind
from gnome3d.pipeline.state import Densified, DistEstimated, State
from gnome3d.util import add_movable_noise_inplace, seed_rng

if TYPE_CHECKING:
    from gnome3d.settings import Settings
    from gnome3d.types import BoolArray, F32Array

_SEED_SALT = 1  # differs from ARCS (0) and SMOOTH (2)
_LOG = log.get("mc.jax")


@njit(cache=True, parallel=True, nogil=True, fastmath=True)
def _accumulate_pairwise_dist(acc: F32Array, pos: F32Array) -> None:
    """Adds the (n, n) distance matrix of `pos` into `acc`, in place and in float32."""
    n = pos.shape[0]
    for i in prange(n):
        xi = pos[i, 0]
        yi = pos[i, 1]
        zi = pos[i, 2]
        for j in range(n):
            dx = xi - pos[j, 0]
            dy = yi - pos[j, 1]
            dz = zi - pos[j, 2]
            acc[i, j] += np.sqrt(dx * dx + dy * dy + dz * dz)


@njit(cache=True, parallel=True, nogil=True)  # no fastmath, the result must stay bit identical
def _fill_heat_dist_inplace(
    avg_dist: F32Array, heat_raw: F32Array, avg_heat: float, influence: float
) -> None:
    """Overwrites `avg_dist` in place with the target
    `avg_dist * (1 - min(heat / avg_heat * influence, 1))`. `heat_raw` and `avg_dist` must be symmetric.
    """
    n = avg_dist.shape[0]
    for i in prange(n):
        for j in range(n):
            s_val = (heat_raw[i, j] / avg_heat) * influence
            if s_val > 1.0:
                s_val = 1.0
            avg_dist[i, j] = avg_dist[i, j] * (1.0 - s_val)


def _estimate_avg_dist(
    pos: F32Array, fixed: BoolArray, dtn: F32Array, step: float, s: Settings, seed: int
) -> F32Array:
    """Returns the average pair distances over dry smooth passes.

    Each replicate runs `n_steps` dry passes from the noised positions and adds the pair distances
    of its best.
    """
    from gnome3d.mc import numba as mc_numba

    n = len(pos)
    n_reps = int(s.subanchor_estimate_replicates)
    n_steps = int(s.subanchor_estimate_steps)
    seed_rng(seed)
    mc_numba.seed_numba(seed)

    avg_dist: F32Array = np.zeros((n, n), dtype=np.float32)
    for _rep in range(n_reps):
        rep_best_score = -1.0
        rep_best_pos: F32Array = pos.copy()
        for _step in range(n_steps):
            pos_trial: F32Array = pos.copy()
            add_movable_noise_inplace(pos_trial, fixed, step)
            # The dry pass has no heat, no orientation and no affinity. Affinity is attractive and
            # would shrink the distances that become the heat target.
            score = mc_numba.mc_smooth_numba(pos_trial, dtn, fixed, step, s)
            if score < rep_best_score or rep_best_score < 0.0:
                rep_best_score = score
                rep_best_pos = pos_trial.copy()
        _accumulate_pairwise_dist(avg_dist, rep_best_pos)
    avg_dist /= n_reps
    return avg_dist


def _heat_dist_from_avg(
    avg_dist: F32Array, subanchor_heat_raw: F32Array, s: Settings
) -> F32Array | None:
    """Returns the distance target from the average distances and the raw subanchor heat, None
    for an empty heatmap.
    """
    avg_heat = float(subanchor_heat_raw.mean(dtype=np.float64))  # summed in float64
    if avg_heat < 1e-6:
        return None
    influence = float(s.subanchor_heatmap_influence)
    # `avg_dist` is overwritten, which saves a second (n, n) matrix.
    _fill_heat_dist_inplace(avg_dist, subanchor_heat_raw, avg_heat, influence)
    return avg_dist


def _run(problem: Problem) -> Result:
    """Estimates the average distances and builds the target matrix of one IB."""
    avg = _estimate_avg_dist(
        problem["pos"],
        problem["fixed"],
        problem["dtn"],
        problem["step_size"],
        problem["settings"],
        problem["seed"],
    )

    return _heat_dist_from_avg(avg, problem["subanchor_heat_raw"], problem["settings"])


# Spreads a parent seed and a replicate number. Blocks with adjacent seeds do not share
# replicate streams.
_REPLICATE_STRIDE = 0x9E3779B1


def _expand_replicates(problems: list[Problem], per_ib: int) -> tuple[list[Problem], list[int]]:
    """Returns one dry smooth problem per replicate and the index where the replicates of each
    block start.

    Each replicate carries a seed set by its block and its number. The batched kernel seeds a
    chain from that seed. The grouping does not decide its stream.
    """
    expanded: list[Problem] = []
    spans: list[int] = []
    for prob in problems:
        spans.append(len(expanded))
        seed_rng(int(prob["seed"]))
        pos = prob["pos"]
        fixed = prob["fixed"]
        step = float(prob["step_size"])
        for rep in range(per_ib):
            start = pos.copy()
            add_movable_noise_inplace(start, fixed, step)
            expanded.append(
                {
                    "pos": start,
                    "dtn": prob["dtn"],
                    "fixed": fixed,
                    "step_size": step,
                    "seed": (int(prob["seed"]) + rep * _REPLICATE_STRIDE) & 0x7FFFFFFF,
                    "heat_dist": None,  # dry
                    "char_orientations": None,
                    "anchor_neighbors": None,
                    "anchor_neighbor_weights": None,
                }
            )
    return expanded, spans


def _batch_run(problems: list[Problem]) -> list[Result]:
    """Runs the dry smooth trials of every IB in one batched kernel and returns one heat_dist
    or None per problem.
    """
    from gnome3d.pipeline.ib.smooth import run_smooth_batch

    s = problems[0]["settings"]
    n_reps = int(s.subanchor_estimate_replicates)
    n_steps = int(s.subanchor_estimate_steps)
    per_ib = n_reps * n_steps

    expanded, spans = _expand_replicates(problems, per_ib)

    results = run_smooth_batch(expanded, s)

    out: list[Result] = []
    for gi, prob in enumerate(problems):
        n = len(prob["pos"])
        base = spans[gi]
        avg_dist: F32Array = np.zeros((n, n), dtype=np.float32)
        for rep in range(n_reps):
            rep_slice = results[base + rep * n_steps : base + (rep + 1) * n_steps]
            scores = [r[0] for r in rep_slice]
            best_pos = rep_slice[int(np.argmin(scores))][1]
            _accumulate_pairwise_dist(avg_dist, best_pos)
        avg_dist /= n_reps
        out.append(_heat_dist_from_avg(avg_dist, prob["subanchor_heat_raw"], s))
    return out


class EstimateDistStage:
    """`Densified -> DistEstimated`."""

    kind = StageKind.ESTIMATE_DIST

    def bucket(self, inputs: tuple[State, ...]) -> int:
        return int(inputs[0].pos.shape[0])  # type: ignore[attr-defined]

    def batch_key(self, inputs: tuple[State, ...]) -> tuple[object, ...]:
        """The batch key has no bead bucket when launches are merged. The dry pass has no heat
        target, the only input that grows with the square of the padded extent.
        """
        st = inputs[0]
        assert isinstance(st, Densified)
        if bool(st.settings.merge_smooth_launches):
            return (0,)
        return (batch_bucket(int(st.pos.shape[0]), st.settings),)

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
        return DistEstimated(**vars(st), heat_dist=result)
