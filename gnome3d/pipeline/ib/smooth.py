"""The SMOOTH stage runs the final smooth MC and returns the beads.

The per anchor int8 `Orientation` codes are expanded to the per bead `"<U1"` array of the
numba kernel. A chain without an ESTIMATE_DIST stage arrives as `Densified`, with no
`heat_dist`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from gnome3d.pipeline.ib.buckets import batch_bucket
from gnome3d.pipeline.ib.maxent import maxent_start
from gnome3d.pipeline.ib.start import coil_start
from gnome3d.pipeline.stage import Problem, Result, StageKind
from gnome3d.pipeline.state import AnchorMapEntry, Densified, Orientation, Smoothed, State
from gnome3d.types import BeadOut
from gnome3d.util import add_movable_noise_inplace, seed_rng

if TYPE_CHECKING:
    from gnome3d.settings import Settings
    from gnome3d.types import BoolArray, F32Array, I8Array, StrArray

_SEED_SALT = 2  # differs from ARCS (0) and ESTIMATE_DIST (1)
_CODE_TO_CHAR = {
    int(Orientation.NONE): "N",
    int(Orientation.LEFT): "L",
    int(Orientation.RIGHT): "R",
}


def _expand_orientations(
    orientations: I8Array, anchor_map: list[AnchorMapEntry], n: int
) -> StrArray:
    """Expands per anchor int8 codes to a per bead `"<U1"` array, 'N' for subanchors."""
    char: StrArray = np.array(["N"] * n, dtype="<U1")
    for bead_idx, anchor_k in anchor_map:
        char[bead_idx] = _CODE_TO_CHAR[int(orientations[anchor_k])]
    return char


def _assemble_beads(
    final_pos: F32Array, fixed: BoolArray, starts: list[int], ends: list[int], s: Settings
) -> list[BeadOut]:
    """Builds the BeadOut list. `fixed` marks the beads of kind anchor. With
    `drop_zero_length_subanchors` the subanchors of zero length are left out.
    """
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


def centre_beads(fixed: BoolArray) -> BoolArray:
    """Returns the fixed beads that move under `anchor_centre`, each with the bead before it and
    the bead after it.

    Such a bead is not at an end of the chain, neither of its neighbours is fixed and it is three
    beads or more from the bead of the rule before it. No bead then belongs to two of them.
    """
    n = int(fixed.shape[0])
    out: BoolArray = np.zeros(n, dtype=np.bool_)
    last = -3
    for b in np.flatnonzero(fixed):
        if 1 <= b <= n - 2 and not fixed[b - 1] and not fixed[b + 1] and b - last >= 3:
            out[b] = True
            last = int(b)
    return out


def run_smooth_batch(expanded: list[Problem], s: Settings) -> list[Result]:
    """Runs one batched smooth launch."""
    from gnome3d.mc import jax as mc_jax

    return mc_jax.mc_smooth_jax_batch(expanded, s)


def _batch_run(problems: list[Problem]) -> list[Result]:
    """Runs the smooth of a group of IBs in one vmapped kernel, each IB with `steps_smooth`
    noised restarts. Returns one `(score, pos)` per problem.
    """
    s = problems[0]["settings"]
    if float(s.motif_bond_limit) > 0.0 and problems[0].get("char_orientations") is not None:
        raise ValueError(
            "[motif_orientation] bond_limit is above zero and the smooth stage runs on the batch"
            " executor, whose kernel has no bond limit. Set [simulation_backend]"
            " mc_executor_smooth to threaded or serial, or bond_limit to 0."
        )
    if s.smooth_anchor_centre:
        raise ValueError(
            "[simulation_arcs_smooth] anchor_centre is on and the smooth stage runs on the batch"
            " executor, whose kernel has no move of three beads. Set [simulation_backend]"
            " mc_executor_smooth to threaded or serial."
        )
    n_restarts = max(1, int(s.steps_smooth))

    expanded: list[Problem] = []
    owner: list[int] = []
    for gi, prob in enumerate(problems):
        seed_rng(int(prob["seed"]))
        pos = _start_positions(prob)
        fixed = prob["fixed"]
        step = float(prob["step_size"])
        for _ in range(n_restarts):
            start = pos.copy()
            add_movable_noise_inplace(start, fixed, step)
            expanded.append({**prob, "pos": start})
            owner.append(gi)

    results = run_smooth_batch(expanded, s)

    best: dict[int, Result] = {}
    for (score, final_pos), gi in zip(results, owner, strict=True):
        if gi not in best or score < best[gi][0]:
            best[gi] = (score, final_pos)
    return [best[gi] for gi in range(len(problems))]


def _start_positions(prob: Problem) -> F32Array:
    """Returns the chain that the stage starts from: the densified line, a compact bridge per
    gap under `smooth_start = coil` or a maximum entropy bridge under `maxent`.

    The bridges draw from a generator seeded by the problem and leave the stream of the kernel
    unchanged.
    """
    pos: F32Array = prob["pos"]
    s = prob["settings"]
    start = str(getattr(s, "smooth_start", "line"))
    if start not in ("coil", "maxent"):
        return pos
    dtn = prob["dtn"]
    r = float(s.exclusion_radius_smooth)
    if r <= 0.0:
        r = float(s.exclusion_auto_factor_smooth) * float(np.mean(dtn))
    rng = np.random.default_rng(int(prob["seed"]))
    if start == "coil":
        return coil_start(pos, prob["fixed"], dtn, r, rng)
    return maxent_start(
        pos,
        prob["fixed"],
        prob["bead_starts"],
        prob["bead_ends"],
        s.polymer_law(),
        float(s.heatmap_distance_stretching),
        rng,
        str(s.data_contact_map),
        str(prob["chrom"]),
    )


def _run(problem: Problem) -> Result:
    """Runs `steps_smooth` restarts from the running best and returns `(best_score, best_pos)`."""
    from gnome3d.mc import numba as mc_numba

    pos: F32Array = _start_positions(problem)
    dtn = problem["dtn"]
    fixed = problem["fixed"]
    step = float(problem["step_size"])
    s = problem["settings"]
    seed = int(problem["seed"])
    char_orn = problem["char_orientations"]
    nbrs = problem["anchor_neighbors"]
    nbr_w = problem["anchor_neighbor_weights"]
    heat = problem["heat_dist"]
    comp = problem["compartment"]

    centre = centre_beads(fixed) if s.smooth_anchor_centre else None

    seed_rng(seed)
    mc_numba.seed_numba(seed)

    best_score = -1.0
    best_pos: F32Array = pos.copy()
    for _run_i in range(max(1, int(s.steps_smooth))):
        pos_run: F32Array = best_pos.copy()
        add_movable_noise_inplace(pos_run, fixed, step)
        score = mc_numba.mc_smooth_numba(
            pos_run, dtn, fixed, step, s, char_orn, nbrs, nbr_w, heat, comp, centre=centre
        )
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
        """Returns `(heat, orn, comp, bead bucket)`, the flags `mc_smooth_jax_batch` reads from
        `problems[0]`. A batch must be uniform in them.
        """
        st = inputs[0]
        assert isinstance(st, Densified)
        heat = getattr(st, "heat_dist", None) is not None
        orn = (
            st.orientations is not None
            and st.anchor_neighbors is not None
            and st.anchor_neighbor_weights is not None
        )
        comp = st.bead_compartment is not None
        # With merged launches the bead bucket is not in the key. `mc_smooth_jax_batch` splits a
        # merged group for device memory.
        if bool(st.settings.merge_smooth_launches):
            return heat, orn, comp, 0
        return heat, orn, comp, batch_bucket(int(st.pos.shape[0]), st.settings)

    @staticmethod
    def describe_batch_key(key: tuple[object, ...]) -> str:
        """Returns the batch key in words, for logs."""
        heat, orn, comp, bucket = key
        return (
            f"heat={'yes' if heat else 'no'} orn={'yes' if orn else 'no'} "
            f"comp={'yes' if comp else 'no'} "
            + (f"{bucket}-bead bucket" if bucket else "all bead sizes")
        )

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
            "compartment": st.bead_compartment,
            "bead_starts": st.bead_starts,
            "bead_ends": st.bead_ends,
            "chrom": st.chrom,
        }

    def apply(self, inputs: tuple[State, ...], result: Result) -> State:
        st = inputs[0]
        assert isinstance(st, Densified)
        _score, final_pos = result
        # On a uniform chain a bead under an anchor has the kind anchor in every conformation.
        kind = st.fixed if st.covered is None else st.covered
        beads = _assemble_beads(final_pos, kind, st.bead_starts, st.bead_ends, st.settings)
        base: dict[str, Any] = {**vars(st), "heat_dist": getattr(st, "heat_dist", None)}
        return Smoothed(**base, final_pos=final_pos, beads=beads)
