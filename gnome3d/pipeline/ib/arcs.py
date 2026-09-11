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
    """Serial runner: anneal one IB's anchors.  Returns `(best_score, best_pos)`.

    Mirrors `Solver._reconstruct_cluster_arcs` exactly (initial per-anchor noise,
    `steps_arcs` restarts, best-of), with deterministic seeding added.
    """
    from gnome3d.mc import numba as mc_numba

    pos0: F32Array = problem["anchor_pos"]
    exp_dist = problem["exp_dist"]
    step = float(problem["step_size"])
    s = settings_for_block(problem["settings"], problem["anchor_genomic"])
    seed = int(problem["seed"])

    # Deterministic per-IB RNG: Python `random` (initial noise) + numba (kernel).
    seed_rng(seed)
    mc_numba.seed_numba(seed)

    # Anneal or solve. The solver minimises the same energy directly and reaches the same
    # minimum far faster, because the landscape is a funnel. Restarts, noise and best-of are
    # the same either way, so an ensemble still comes from the perturbed starts.
    solver = arcs_solver(s)
    if s.arcs_scope == "chromosome":
        # The chromosome's anchors were solved together at seed time; the block's are final.
        return 0.0, np.asarray(pos0, dtype=np.float32)
    if s.arcs_start == "walk":
        pos0 = walk_start(pos0, problem["anchor_genomic"], s.polymer_law())
    elif s.arcs_start == "hilbert":
        pos0 = hilbert_start(pos0, problem["anchor_genomic"], s.polymer_law())
    elif s.arcs_start != "centroid":
        raise ValueError(f"unknown arcs start {s.arcs_start!r}, expected centroid, walk or hilbert")

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


def walk_start(pos0: F32Array, anchor_genomic: object, law: PolymerLaw) -> F32Array:
    """Anchor starting positions on a random walk whose steps are the law's distance for each
    consecutive gap, centred where the block's centroid was.

    Every anchor otherwise starts at the block's centroid and the solver descends from that
    collapsed point to a compact minimum, bounded only by the block's size. On the walk a pair
    no term acts on begins near the law and the solver leaves it there. Directions come from
    the seeded Python RNG, so a block's walk is reproducible.
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
    """Coordinates on the 3D Hilbert curve of `bits` bits per axis for each curve index.

    Skilling's transpose to axes, vectorised. The index's bits are dealt round robin to the
    three axes, most significant first, then Gray decoded and unwound axis by axis.
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
    """Anchor starting positions along a 3D Hilbert curve, each anchor at the curve point of
    its genomic fraction, scaled so consecutive anchors sit at the law's bond on average and
    centred where the block's centroid was.

    A space filling curve keeps genomic neighbours spatial neighbours at every scale and its
    size grows as the cube root of the count, the exponent the maps show beyond a megabase.
    A walk has the wrong exponent and a collapsed start none. No RNG; the start is a
    function of the genomic positions alone.
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


def run_arcs_problem(problem: Problem) -> Result:
    """Solve or anneal one arcs problem on the calling thread. The joint chromosome solve in
    the skeleton uses it directly, outside the executor."""
    return _run(problem)


def settings_for_block(
    s: Settings, anchor_genomic: Sequence[int] | Sequence[tuple[int, int, int]] | I64Array
) -> Settings:
    """The settings one block's kernels run with.

    With `confinement_packing_factor_arcs` at zero the block's confinement radius is the sphere
    the law says a chain of its genomic span fills, and it reaches the kernels as an explicit
    `confinement_radius_arcs` on a shallow copy, so the numba annealer, the solver and the JAX
    kernel all read one value. The input is returned as is when the factor is positive, when
    confinement is off at this level, or when a radius is set by hand.

    Parameters
    ----------
    s
        The run's settings.
    anchor_genomic
        The block's anchors as the state carries them, (start, end, midpoint) triples, or a
        flat sequence of positions. The span runs from the first start to the last end.
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
    if s.arcs_scope == "chromosome":
        return [(0.0, np.asarray(p["anchor_pos"], dtype=np.float32)) for p in problems]
    if s.arcs_start != "centroid":
        raise NotImplementedError("the batched arcs runner starts at the centroid only")
    if settings_for_block(s, problems[0]["anchor_genomic"]) is not s:
        raise ValueError(
            "the batch executor runs a launch on one settings and cannot give each block the "
            "confinement radius the law derives for its span; set mc_executor_arcs to serial "
            "or threaded, or give confinement_packing_factor_arcs a positive value"
        )
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
