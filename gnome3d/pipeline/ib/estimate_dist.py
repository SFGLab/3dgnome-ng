"""
ESTIMATE_DIST stage: estimate the subanchor contact-distance target matrix.

Pure port of `Solver._build_heat_dist_subanchor` = `_estimate_avg_dist` (dry
smooth-MC passes to get average pairwise distances) + `_heat_dist_from_avg`
(scale high-contact pairs down by `influence`).  Reads the `Densified` state,
seeds deterministically from `Seeded.seed`.  Only present in a chain when the
skeleton's sparse-signal early-out kept it (`IBSeed.wants_heat`).

Serial runner = numba dry smooth (`mc_smooth_numba` with no heat/orientation).
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

_SEED_SALT = 1  # distinct from ARCS(0)/SMOOTH(2) so the noise streams don't correlate
_LOG = log.get("mc.jax")


@njit(cache=True, parallel=True, nogil=True, fastmath=True)
def _accumulate_pairwise_dist(acc: F32Array, pos: F32Array) -> None:
    """Add the full (n,n) Euclidean distance matrix of `pos` into `acc`, in place.

    Distances are computed directly (subtract then sqrt) exactly like the
    reference `calcTrueDistancesHeatmapForRegion` (LooperSolver.cpp):
    `d = (pos_i - pos_j).length()`.  Replaces the `pos[:,None,:] - pos[None,:,:]`
    broadcast (which materialised an (n,n,3) temporary) *and* the separate (n,n)
    result it accumulated into - folding the add into the kernel avoids a
    per-rep temp at the B=32768 bucket.  `acc` is float32 (matching the reference's
    float Heatmap, which also accumulates in single precision).  prange-parallel."""
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


@njit(cache=True, parallel=True, nogil=True)  # no fastmath: keep bit-identical to the old loop
def _fill_heat_dist_inplace(
    avg_dist: F32Array, heat_raw: F32Array, avg_heat: float, influence: float
) -> None:
    """Overwrite `avg_dist` in place with the expected-distance target, mirroring
    the reference `createExpectedDistSubanchorHeatmap`: high-contact pairs scaled
    down by `influence`.  Same arithmetic as the old pure-Python double loop
    (`target = avg_dist * (1 - min(heat/avg_heat*influence, 1))`), but
    prange-parallel and with no extra (n,n) allocation - the old loop both stalled
    (~5e8 Python iterations at n=32768) and allocated a second 8.6 GB matrix.
    Relies on `heat_raw`/`avg_dist` being symmetric (contact heatmaps are); the
    diagonal stays 0 since `avg_dist[i,i]` is 0."""
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
    """Average pairwise bead distances over dry smooth-MC passes.  Port of
    `Solver._estimate_avg_dist` (reference sequential path): per replicate, run
    `n_steps` dry passes from pos+noise, keep the best, accumulate its pairwise
    distances."""
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
            # Dry pass: no heat, no orientation, and no affinity either.  The
            # affinity terms are attractive, so including them here would shrink
            # the estimated pairwise distances that become the heat target, and
            # the real smooth pass would then compact against an already-compacted
            # target.  EV and confinement stay because they push outward.
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
    """Turn avg distances + raw subanchor heat into the expected-distance target
    (Reference createExpectedDistSubanchorHeatmap): high-contact pairs scaled
    down by `influence`.  None when the heatmap is empty.  Port of
    `Solver._heat_dist_from_avg` (sans the INFO logging/counters)."""
    avg_heat = float(subanchor_heat_raw.mean(dtype=np.float64))  # f64 accum on the f32 matrix
    if avg_heat < 1e-6:
        return None
    influence = float(s.subanchor_heatmap_influence)
    # Overwrite avg_dist in place into the target matrix - avg_dist is not needed
    # afterwards, so this reuses its (n,n) buffer instead of allocating a second.
    _fill_heat_dist_inplace(avg_dist, subanchor_heat_raw, avg_heat, influence)
    return avg_dist


def _run(problem: Problem) -> Result:
    """Serial runner: estimate avg_dist (dry smooth) then build the target matrix."""
    avg = _estimate_avg_dist(
        problem["pos"],
        problem["fixed"],
        problem["dtn"],
        problem["step_size"],
        problem["settings"],
        problem["seed"],
    )

    return _heat_dist_from_avg(avg, problem["subanchor_heat_raw"], problem["settings"])


# Spreads a parent seed and a replicate number across the output range before they are mixed,
# so two blocks whose seeds differ by one do not hand their replicates overlapping streams.
_REPLICATE_STRIDE = 0x9E3779B1


def _expand_replicates(problems: list[Problem], per_ib: int) -> tuple[list[Problem], list[int]]:
    """One dry smooth problem per replicate, and where each input block's run starts.

    Each replicate carries a seed fixed by its parent block and its replicate number. The
    batched kernel seeds a chain from the seed its problem carries, so a replicate without one
    would fall back to its slot in the launch and the grouping would decide its stream.
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
                    "heat_dist": None,  # dry: chain+EV+conf only
                    "char_orientations": None,
                    "anchor_neighbors": None,
                    "anchor_neighbor_weights": None,
                }
            )
    return expanded, spans


def _batch_run(problems: list[Problem]) -> list[Result]:
    """Batched (JAX) runner: run every IB's dry-smooth trials in one vmapped
    kernel (no heat/orientation), then build each IB's target matrix.  Mirrors
    `JaxSolver._batched_heat_dist`.  Returns one heat_dist (or None) per input
    problem, in order."""
    from gnome3d.pipeline.ib.smooth import run_smooth_batch

    s = problems[0]["settings"]
    n_reps = int(s.subanchor_estimate_replicates)
    n_steps = int(s.subanchor_estimate_steps)
    per_ib = n_reps * n_steps

    expanded, spans = _expand_replicates(problems, per_ib)

    # Plain "checker" double-compacts here: estimation's output is the dense distance TARGET
    # the final smooth chases, and the checker's stale-EV compaction shrinks it (measured Rg
    # 0.965 -> 0.890 at B=1024).  Only "hybrid" (checker init + sequential polish) yields a
    # CORRECT target, so estimation upgrades checker->hybrid and is otherwise sequential -
    # never plain checker.  See docs/arcs-gpu-acceleration.md.
    est_setting = str(getattr(s, "mc_executor_jax_estimate_kernel", "auto")).strip().lower()
    if est_setting in ("mc", "hybrid"):
        est_kernel = est_setting  # explicit override (never plain checker - it compounds)
    else:  # "auto": follow the final-smooth kernel (hybrid -> hybrid), else sequential
        smooth_k = str(getattr(s, "mc_executor_jax_smooth_kernel", "mc")).strip().lower()
        est_kernel = "hybrid" if smooth_k == "hybrid" else "mc"
    # Make the fan-out explicit: a batch of N estimate nodes expands to N x (reps*steps) dry-smooth
    # IBs, which the kernel then chunks - so the smooth[checker]/[mc] line count is NOT the node count.
    log.status(
        _LOG,
        "    estimate: %d nodes x %d reps = %d dry-smooth IBs",
        len(problems),
        per_ib,
        len(expanded),
    )
    results = run_smooth_batch(expanded, s, est_kernel)

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
        """Heat-dist is the dry smooth, with no heat or orientation terms, so nothing but the
        bead extent could separate two blocks.

        Merging drops that too. The dry pass carries no heat target, which is the only input
        that grows with the square of the padded extent, so a merged launch of every block's
        replicates costs tens of megabytes and the packing in `mc_smooth_jax_batch` has no
        reason to split it."""
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
