"""
SMOOTH stage: the final smooth-MC, producing positioned beads.

Pure port of `Solver._run_smooth_serial` (steps_smooth restarts from the running
best, keep best) + `_apply_smooth_problem` (assemble BeadOut), reading
`Densified`/`DistEstimated` and seeding deterministically from `Seeded.seed`.

Two boundary conversions live here:
  * orientation: the compact per-anchor `int8` `Orientation` codes are expanded
    to the per-bead `"<U1"` array the numba kernel expects, via `anchor_map`.
  * heat: a chain without a ESTIMATE_DIST stage arrives as `Densified` (no heat_dist)
    rather than `DistEstimated`; both are accepted (`getattr(..., None)`).

The cluster write-back the old `_apply_smooth_problem` did is gone - IBs are
terminal here, the beads ARE the output.  Serial runner = numba `mc_smooth_numba`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gnome3d.pipeline.ib.buckets import batch_bucket
from gnome3d.pipeline.stage import Problem, Result, StageKind
from gnome3d.pipeline.state import AnchorMapEntry, Densified, Orientation, Smoothed, State
from gnome3d.types import BeadOut
from gnome3d.util import add_movable_noise_inplace, seed_rng

if TYPE_CHECKING:
    from gnome3d.settings import Settings
    from gnome3d.types import BoolArray, F32Array, I8Array, StrArray

_SEED_SALT = 2  # distinct from ARCS(0)/EST_DIST(1)
_CODE_TO_CHAR = {int(Orientation.NONE): "N", int(Orientation.LEFT): "L", int(Orientation.RIGHT): "R"}


def _expand_orientations(orientations: I8Array, anchor_map: list[AnchorMapEntry], n: int) -> StrArray:
    """Per-anchor int8 codes -> per-bead `"<U1"` array ('N' for subanchors),
    placing each anchor's orientation at its bead via `anchor_map`.  Mirrors the
    char_orn construction in `Solver._build_smooth_problem`."""
    char: StrArray = np.array(["N"] * n, dtype="<U1")
    for bead_idx, anchor_k in anchor_map:
        char[bead_idx] = _CODE_TO_CHAR[int(orientations[anchor_k])]
    return char


def _assemble_beads(
    final_pos: F32Array, fixed: BoolArray, starts: list[int], ends: list[int], s: Settings
) -> list[BeadOut]:
    """Build the BeadOut list.  Port of `Solver._apply_smooth_problem` (sans the
    cluster write-back): optionally drop zero-length subanchors from the visible
    output (the MC chain still contained them)."""
    drop_zero = s.drop_zero_length_subanchors
    return [
        BeadOut(
            start=starts[i],
            end=ends[i],
            x=float(final_pos[i, 0]),
            y=float(final_pos[i, 1]),
            z=float(final_pos[i, 2]),
            kind="anchor" if bool(fixed[i]) else "subanchor",
        )
        for i in range(len(final_pos))
        if not (drop_zero and not bool(fixed[i]) and starts[i] == ends[i])
    ]


def _batch_run(problems: list[Problem]) -> list[Result]:
    """Batched (JAX) runner: anneal a whole bucket of IBs' smooths in one vmapped
    kernel.  Each IB is fanned out to `steps_smooth` noised restarts (best kept),
    mirroring `JaxSolver._batched_final_smooth`.  Returns one `(score, pos)`` per
    input problem, in order.  `mc_jax` is imported lazily so the numba path never
    requires JAX."""
    from gnome3d.mc import jax as mc_jax

    s = problems[0]["settings"]
    n_restarts = max(1, int(s.steps_smooth))

    expanded: list[Problem] = []
    owner: list[int] = []
    for gi, prob in enumerate(problems):
        seed_rng(int(prob["seed"]))  # deterministic restart noise for this IB
        pos = prob["pos"]
        fixed = prob["fixed"]
        step = float(prob["step_size"])
        for _ in range(n_restarts):
            start = pos.copy()
            add_movable_noise_inplace(start, fixed, step)
            expanded.append({**prob, "pos": start})
            owner.append(gi)

    results = mc_jax.mc_smooth_jax_batch(expanded, s)

    best: dict[int, Result] = {}
    for (score, final_pos), gi in zip(results, owner, strict=True):
        if gi not in best or score < best[gi][0]:
            best[gi] = (score, final_pos)
    return [best[gi] for gi in range(len(problems))]


def _run(problem: Problem) -> Result:
    """Serial runner: steps_smooth restarts from the running best; returns
    ``(best_score, best_pos)``.  Mirrors `Solver._run_smooth_serial`."""
    from gnome3d.mc import numba as mc_numba

    pos: F32Array = problem["pos"]
    dtn = problem["dtn"]
    fixed = problem["fixed"]
    step = float(problem["step_size"])
    s = problem["settings"]
    seed = int(problem["seed"])
    char_orn = problem["char_orientations"]
    nbrs = problem["anchor_neighbors"]
    nbr_w = problem["anchor_neighbor_weights"]
    heat = problem["heat_dist"]

    seed_rng(seed)
    mc_numba.seed_numba(seed)

    best_score = -1.0
    best_pos: F32Array = pos.copy()
    for _run_i in range(max(1, int(s.steps_smooth))):
        pos_run: F32Array = best_pos.copy()
        add_movable_noise_inplace(pos_run, fixed, step)
        score = mc_numba.mc_smooth_numba(pos_run, dtn, fixed, step, s, char_orn, nbrs, nbr_w, heat)
        if score < best_score or best_score < 0.0:
            best_score = score
            best_pos = pos_run.copy()
    return best_score, best_pos


class SmoothStage:
    """`Densified | DistEstimated -> Smoothed`."""

    kind = StageKind.SMOOTH

    def bucket(self, inputs: tuple[State, ...]) -> int:
        return int(inputs[0].pos.shape[0])  # type: ignore[attr-defined]

    def batch_key(self, inputs: tuple[State, ...]) -> tuple[object, ...]:
        """``(heat?, orn?, bead-bucket)`` - the exact signature
        `mc_smooth_jax_batch` reads from ``problems[0]`` to pick its kernel.  A
        batch MUST be uniform in these flags, so they are part of the key (else a
        no-heat IB could land in a heat batch and get the wrong kernel)."""
        st = inputs[0]
        assert isinstance(st, Densified)
        heat = getattr(st, "heat_dist", None) is not None
        orn = (
            st.orientations is not None
            and st.anchor_neighbors is not None
            and st.anchor_neighbor_weights is not None
        )
        return heat, orn, batch_bucket(int(st.pos.shape[0]), st.settings)

    def to_problem(self, inputs: tuple[State, ...]) -> Problem:
        st = inputs[0]
        assert isinstance(st, Densified)  # DistEstimated is a Densified subclass
        char_orn = (
            _expand_orientations(st.orientations, st.anchor_map, st.pos.shape[0])
            if st.orientations is not None
            else None
        )
        return {
            "pos": st.pos,
            "dtn": st.dtn,
            "fixed": st.fixed,
            "step_size": st.step_size_smooth,
            "settings": st.settings,
            "seed": (st.seed + _SEED_SALT) & 0x7FFFFFFF,
            "char_orientations": char_orn,
            "anchor_neighbors": st.anchor_neighbors,
            "anchor_neighbor_weights": st.anchor_neighbor_weights,
            "heat_dist": getattr(st, "heat_dist", None),
        }

    def apply(self, inputs: tuple[State, ...], result: Result) -> State:
        st = inputs[0]
        assert isinstance(st, Densified)
        _score, final_pos = result
        beads = _assemble_beads(final_pos, st.fixed, st.bead_starts, st.bead_ends, st.settings)
        base = {**vars(st), "heat_dist": getattr(st, "heat_dist", None)}
        return Smoothed(**base, final_pos=final_pos, beads=beads)
