"""The ARCS stage places the anchors of an IB by arc springs.

The RNGs are seeded from `Seeded.seed`. A result does not depend on the order of execution.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from gnome3d.pipeline.ib.buckets import batch_bucket
from gnome3d.pipeline.stage import Problem, Result, StageKind
from gnome3d.pipeline.state import Arced, Seeded, State
from gnome3d.polymer import PolymerLaw
from gnome3d.util import add_movable_noise_inplace, positioning_rng, seed_rng

if TYPE_CHECKING:
    from gnome3d.settings import Settings
    from gnome3d.types import F32Array, F64Array, I64Array


def _run(problem: Problem) -> Result:
    """Runs one problem on the CPU. Returns `(best_score, best_pos)`."""
    return _run_one(problem, "numba")


def _run_one(problem: Problem, backend: str) -> Result:
    """Anneals or solves the anchors of one IB with the energy on `backend`.

    Noises the anchor seeds, runs `steps_arcs` restarts and keeps the best.
    """
    from gnome3d.mc import numba as mc_numba

    pos0: F32Array = problem["anchor_pos"]
    exp_dist = problem["exp_dist"]
    arc_w = problem.get("arc_w")
    activity = problem.get("activity")
    step = float(problem["step_size"])
    s = settings_for_block(problem["settings"], problem["anchor_genomic"])
    seed = int(problem["seed"])

    seed_rng(seed)
    mc_numba.seed_numba(seed)

    solver = arcs_solver(s)
    if s.arcs_scope == "chromosome":
        # At chromosome scope the anchors were solved together when the seed was built.
        return 0.0, np.asarray(pos0, dtype=np.float32)
    if s.arcs_start == "walk":
        pos0 = walk_start(pos0, problem["anchor_genomic"], s.polymer_law())
    elif s.arcs_start == "hilbert":
        pos0 = hilbert_start(pos0, problem["anchor_genomic"], s.polymer_law())
    elif s.arcs_start != "centroid":
        raise ValueError(
            f"arcs start {s.arcs_start!r} at block scope, expected centroid, walk or hilbert;"
            " maxent is drawn at chromosome scope only"
        )

    best_score = -1.0
    best: F32Array = pos0.copy()
    for _run_i in range(max(1, int(s.steps_arcs))):
        pos: F32Array = pos0.copy()
        add_movable_noise_inplace(pos, None, step)  # every anchor moves
        if solver == "lbfgs":
            from gnome3d.mc.numba.arcs_solver import solve_arcs  # noqa: PLC0415

            g = np.asarray(problem["anchor_genomic"], dtype=np.int64)
            score, pos = solve_arcs(
                pos,
                exp_dist,
                s,
                backend=backend,
                arc_w=arc_w,
                activity=activity,
                mids=g[:, 2] if g.ndim == 2 else g,
                reach=problem.get("reach"),
            )
        else:
            if problem.get("reach") is not None:
                raise NotImplementedError(
                    "the reach term of the uniform chain is in the solver's energy, not the"
                    " annealer's; set [simulation_arcs] solver = lbfgs"
                )
            if float(s.gyration_weight) > 0.0:
                raise NotImplementedError(
                    "the gyration term is in the solver's energy, not the annealer's; set "
                    "[simulation_arcs] solver = lbfgs or [springs] gyration_weight = 0"
                )
            if activity is not None:
                raise NotImplementedError(
                    "the factory term is in the solver's energy, not the annealer's; set "
                    "[simulation_arcs] solver = lbfgs or [factories] weight = 0"
                )
            score = mc_numba.mc_arcs_numba(pos, exp_dist, step, s, arc_w=arc_w)  # in place
        if score < best_score or best_score < 0.0:
            best_score = score
            best = pos.copy()

    return best_score, np.asarray(best, dtype=np.float32)


def walk_start(pos0: F32Array, anchor_genomic: object, law: PolymerLaw) -> F32Array:
    """Returns anchor starts on a random walk whose steps are the law's distance of each
    consecutive gap, centred on the centroid of the block. Directions come from the seeded
    Python RNG.
    """
    g = np.asarray(anchor_genomic, dtype=np.int64)
    mids = g[:, 2] if g.ndim == 2 else g
    n = mids.shape[0]
    out = np.zeros((n, 3), dtype=np.float64)
    rng = positioning_rng()
    for i in range(1, n):
        v = np.array([rng.gauss(0.0, 1.0) for _ in range(3)])
        v /= max(float(np.linalg.norm(v)), 1e-12)
        out[i] = out[i - 1] + v * law.background(int(mids[i] - mids[i - 1]))
    out += np.asarray(pos0, dtype=np.float64).mean(axis=0) - out.mean(axis=0)
    return np.ascontiguousarray(out, dtype=np.float32)


def _hilbert_points(index: I64Array, bits: int) -> F64Array:
    """Returns the coordinates on the 3D Hilbert curve of `bits` bits per axis for each curve
    index, by Skilling's transpose to axes.
    """
    n = 3
    x = np.zeros((index.shape[0], n), dtype=np.int64)
    for level in range(bits):
        for axis in range(n):
            src = n * bits - 1 - (level * n + axis)
            x[:, axis] |= ((index >> src) & 1) << (bits - 1 - level)
    t = x[:, n - 1] >> 1
    for i in range(n - 1, 0, -1):
        x[:, i] ^= x[:, i - 1]
    x[:, 0] ^= t
    q = 2
    top = 1 << bits
    while q != top:
        p = q - 1
        for i in range(n - 1, -1, -1):
            hit = (x[:, i] & q) != 0
            x[hit, 0] ^= p
            t = (x[:, 0] ^ x[:, i]) & p
            t[hit] = 0
            x[:, 0] ^= t
            x[:, i] ^= t
        q <<= 1
    return x.astype(np.float64)


def hilbert_start(pos0: F32Array, anchor_genomic: object, law: PolymerLaw) -> F32Array:
    """Returns anchor starts along a 3D Hilbert curve, each anchor at the curve point of its
    genomic fraction.

    The curve is scaled to the law's bond between consecutive anchors on average and centred
    on the centroid of the block. Uses no RNG.
    """
    g = np.asarray(anchor_genomic, dtype=np.int64)
    mids = g[:, 2] if g.ndim == 2 else g
    n = mids.shape[0]
    bits = 1
    while (1 << (3 * bits)) < 4 * n:
        bits += 1
    span = max(int(mids[-1] - mids[0]), 1)
    frac = (mids - mids[0]).astype(np.float64) / span
    index = np.rint(frac * ((1 << (3 * bits)) - 1)).astype(np.int64)
    pts = _hilbert_points(index, bits)
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    want = np.mean([law.background(int(b - a)) for a, b in zip(mids[:-1], mids[1:], strict=True)])
    scale = want / max(float(steps.mean()), 1e-12) if n > 1 else 1.0
    out = pts * scale
    out += np.asarray(pos0, dtype=np.float64).mean(axis=0) - out.mean(axis=0)
    return np.ascontiguousarray(out, dtype=np.float32)


def arcs_backend(s: Settings) -> str:
    """Returns the place where the arcs solver evaluates its energy. The batch executor is the
    JAX device, serial and threaded are the CPU.
    """
    from gnome3d.pipeline.executor import ExecutorStrategy  # noqa: PLC0415
    from gnome3d.pipeline.stage import StageKind  # noqa: PLC0415
    from gnome3d.reconstruct import stage_strategy  # noqa: PLC0415

    return "jax" if stage_strategy(s, StageKind.ARCS) == ExecutorStrategy.BATCH else "numba"


def run_arcs_problem(problem: Problem) -> Result:
    """Solves or anneals one arcs problem on the calling thread, on the backend of
    `mc_executor_arcs`.
    """
    if arcs_backend(problem["settings"]) == "jax":
        return _run_one(problem, "jax")
    return _run(problem)


def settings_for_block(
    s: Settings, anchor_genomic: Sequence[int] | Sequence[tuple[int, int, int]] | I64Array
) -> Settings:
    """Returns the settings that the kernels of one block run with.

    With `confinement_packing_factor_arcs` at zero the confinement radius is the sphere of the law
    for the genomic span of the block, set as `confinement_radius_arcs` on a shallow copy. The
    input is returned when the factor is positive, when confinement is off at this level or when
    a radius is set.

    Parameters
    ----------
    s
        The settings of the run.
    anchor_genomic
        The anchors of the block as (start, end, midpoint) triples or a flat sequence of
        positions. The span is from the first start to the last end.
    """
    derive = (
        bool(s.use_confinement)
        and bool(s.confinement_apply_to_arcs)
        and float(s.confinement_radius_arcs) <= 0.0
        and float(s.confinement_packing_factor_arcs) <= 0.0
    )
    w_arcs = float(s.confinement_weight_arcs)
    if not derive and w_arcs <= 0.0:
        return s
    if not derive:
        out = copy.copy(s)
        out.confinement_weight = w_arcs
        return out
    g = np.asarray(anchor_genomic, dtype=np.int64)
    if g.size == 0:
        span = 0
    elif g.ndim == 2:
        span = int(g[:, 1].max() - g[:, 0].min())
    else:
        span = int(g.max() - g.min())
    out = copy.copy(s)
    out.confinement_radius_arcs = s.polymer_law().confinement_radius(span)
    if w_arcs > 0.0:
        out.confinement_weight = w_arcs
    return out


def arcs_solver(s: Settings) -> str:
    """Returns the validated solver name of the stage. An unknown name raises."""
    name = str(s.arcs_solver).strip().lower()
    if name not in ("mc", "lbfgs"):
        raise ValueError(f"[simulation_arcs] solver must be mc or lbfgs, got {s.arcs_solver!r}")
    return name


def _batch_run(problems: list[Problem]) -> list[Result]:
    """Runs a group of problems on JAX and returns one `(score, pos)` per problem.

    With the solver each block is solved in turn with its energy on the device. With the annealer
    the IBs of a bucket anneal in one vmapped kernel, each with `steps_arcs` noised restarts.
    """
    s = problems[0]["settings"]
    if s.arcs_scope == "chromosome":
        return [(0.0, np.asarray(p["anchor_pos"], dtype=np.float32)) for p in problems]
    if arcs_solver(s) == "lbfgs":
        return [_run_one(p, "jax") for p in problems]
    if s.arcs_start != "centroid":
        raise NotImplementedError("the batched arcs runner starts at the centroid only")
    if settings_for_block(s, problems[0]["anchor_genomic"]) is not s:
        raise ValueError(
            "the batch executor runs a launch on one settings and cannot give each block the "
            "confinement radius the law derives for its span; set mc_executor_arcs to serial "
            "or threaded, or give confinement_packing_factor_arcs a positive value"
        )
    from gnome3d.mc import jax as mc_jax  # noqa: PLC0415

    n_restarts = max(1, int(s.steps_arcs))

    expanded: list[Problem] = []
    owner: list[int] = []
    for gi, prob in enumerate(problems):
        if prob.get("arc_w") is not None:
            raise NotImplementedError(
                "loop weights are not in the JAX annealer; the solver carries them on the "
                "device, and the numba annealer on the CPU"
            )
        if prob.get("activity") is not None:
            raise NotImplementedError(
                "the factory term is not in the JAX annealer; the solver carries it on the device"
            )
        seed_rng(int(prob["seed"]))
        pos = prob["anchor_pos"]
        step = float(prob["step_size"])
        for _ in range(n_restarts):
            start = pos.copy()
            add_movable_noise_inplace(start, None, step)  # every anchor moves
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
        """The batch key is the anchor shape bucket. Every arcs problem has the same energy
        terms.
        """
        st = inputs[0]
        assert isinstance(st, Seeded)
        return (batch_bucket(int(st.anchor_seed_pos.shape[0]), st.settings),)

    def to_problem(self, inputs: tuple[State, ...]) -> Problem:
        st = inputs[0]
        assert isinstance(st, Seeded)
        return {
            "anchor_pos": st.anchor_seed_pos,
            "exp_dist": st.exp_dist,
            "arc_w": st.arc_w,
            "activity": st.activity,
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
