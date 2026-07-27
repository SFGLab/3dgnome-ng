"""
DENSIFY stage: insert subanchor beads between anchors.

Pure port of `Solver._densify_active_region` / `_subanchor_counts_per_arc`:
same arithmetic, same order, but parameterized on the `Arced` state (anchor
positions + genomic spans) instead of reaching into the cluster graph.  No RNG,
so this stage is byte-exact-validatable against the solver on its own.

`anchor_map` here is local - `(bead_index, anchor_index_in_ib)` - never a
global cluster index, which is what lets the densified IB stay self-contained.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gnome3d.pipeline.stage import Problem, Result, StageKind
from gnome3d.pipeline.state import AnchorMapEntry, Arced, Densified, State
from gnome3d.tracks import bin_compartments, bin_signal

if TYPE_CHECKING:
    from gnome3d.settings import Settings
    from gnome3d.types import (
        BoolArray,
        CompartmentInterval,
        F32Array,
        I8Array,
        SignalInterval,
    )

# (pos, fixed, starts, ends, dtn, anchor_map, step_size_smooth, compartment, accessibility)
DensifyResult = tuple[
    "F32Array",
    "BoolArray",
    list[int],
    list[int],
    "F32Array",
    list[AnchorMapEntry],
    float,
    "I8Array | None",
    "F32Array | None",
]


def _subanchor_counts(anchor_genomic: list[tuple[int, int, int]], s: Settings) -> list[int]:
    """Subanchors to insert per consecutive anchor pair.  Mirrors
    `Solver._subanchor_counts_per_arc` (uniform `loop_density`, or dynamic by
    in-between span when `use_dynamic_loop_density`)."""
    n_arcs = max(len(anchor_genomic) - 1, 0)
    if not s.use_dynamic_loop_density:
        return [s.loop_density] * n_arcs
    target = max(s.target_bp_per_subanchor, 1)
    mn = max(s.min_subanchors_per_arc, 0)
    mx = max(s.max_subanchors_per_arc, mn)
    counts: list[int] = []
    for i in range(n_arcs):
        _start_a, end_a, _mid_a = anchor_genomic[i]
        start_b, _end_b, _mid_b = anchor_genomic[i + 1]
        span = abs(start_b - end_a)
        n = round(span / target) - 1
        counts.append(max(mn, min(mx, n)))
    return counts


def densify(
    anchor_pos: F32Array,
    anchor_genomic: list[tuple[int, int, int]],
    s: Settings,
    track_compartments: list[CompartmentInterval] | None = None,
    track_accessibility: list[SignalInterval] | None = None,
) -> DensifyResult:
    """Insert subanchor beads between each consecutive anchor pair.  Byte-exact
    port of `Solver._densify_active_region` (see its docstring for the overlap /
    span / midpoint conventions).

    Epigenomic tracks are binned onto the bead ranges produced here, so a
    subanchor reads the track over its own genomic slot rather than inheriting
    from a neighbouring anchor.  When `use_fibre_compaction` is on, accessibility
    also shortens the chain bond targets: closed chromatin is locally compact,
    which is what HiP-HoP's extra i,i+2 springs do."""
    counts = _subanchor_counts(anchor_genomic, s)
    bead_starts: list[int] = []
    bead_ends: list[int] = []
    bead_pos: list[F32Array] = []
    bead_gpos: list[int] = []
    bead_fixed: list[bool] = []
    anchor_map: list[AnchorMapEntry] = []

    for i in range(len(anchor_genomic) - 1):
        sa, ea, ma = anchor_genomic[i]
        sb, _eb, _mb = anchor_genomic[i + 1]
        ld = counts[i]
        ca_pos = anchor_pos[i]
        cb_pos = anchor_pos[i + 1]

        k = len(bead_pos)
        bead_starts.append(sa)
        bead_ends.append(ea)
        bead_pos.append(ca_pos.copy())
        bead_gpos.append(ma)
        bead_fixed.append(True)
        anchor_map.append((k, i))

        if s.overlap_anchor_strict:
            boundary_lo = ea
            span = max(sb - ea, 0)
        else:
            boundary_lo = min(ea, sb)
            boundary_hi = max(ea, sb)
            span = boundary_hi - boundary_lo
        d_bp = span // (ld + 1)
        half_lo = d_bp // 2
        half_hi = d_bp - half_lo
        for j in range(ld):
            midpoint = boundary_lo + (j + 1) * d_bp
            s_bp = midpoint - half_lo
            e_bp = midpoint + half_hi
            t = (j + 1.0) / (ld + 1)
            sub_pos: F32Array = ((1.0 - t) * ca_pos + t * cb_pos).astype(np.float32)
            bead_starts.append(s_bp)
            bead_ends.append(e_bp)
            bead_pos.append(sub_pos)
            bead_gpos.append(midpoint)
            bead_fixed.append(False)

    sl, el, ml = anchor_genomic[-1]
    k = len(bead_pos)
    bead_starts.append(sl)
    bead_ends.append(el)
    bead_pos.append(anchor_pos[-1].copy())
    bead_gpos.append(ml)
    bead_fixed.append(True)
    anchor_map.append((k, len(anchor_genomic) - 1))

    n = len(bead_pos)
    pos_arr: F32Array = np.array(bead_pos, dtype=np.float32)
    fixed_arr: BoolArray = np.array(bead_fixed, dtype=np.bool_)
    dtn: F32Array = np.zeros(n - 1, dtype=np.float32)
    for i in range(n - 1):
        gap = max(bead_gpos[i + 1] - bead_gpos[i], 0)
        dtn[i] = float(s.genomic_length_to_distance(gap))

    compartment: I8Array | None = None
    if track_compartments:
        compartment, _score = bin_compartments(track_compartments, bead_starts, bead_ends)

    accessibility: F32Array | None = None
    if track_accessibility:
        accessibility = bin_signal(track_accessibility, bead_starts, bead_ends)
        if s.use_fibre_compaction and s.fibre_compaction > 0.0:
            # A bond spans beads i and i+1, so it compacts by their mean openness.
            a_bond = 0.5 * (accessibility[:-1] + accessibility[1:])
            scale = 1.0 - s.fibre_compaction * (1.0 - np.clip(a_bond, 0.0, 1.0))
            # Never let a bond target collapse to zero: a zero target makes the
            # chain spring's relative error undefined.
            dtn *= np.maximum(scale, 1e-3).astype(np.float32)

    step_size_smooth = float(dtn.mean()) * s.noise_smooth
    return (
        pos_arr,
        fixed_arr,
        bead_starts,
        bead_ends,
        dtn,
        anchor_map,
        step_size_smooth,
        compartment,
        accessibility,
    )


def _run(problem: Problem) -> Result:
    """Serial runner: densify one IB.  (No GPU kernel - DENSIFY registers only a
    serial runner; the batched executor maps it.)"""
    return densify(
        problem["anchor_pos"],
        problem["anchor_genomic"],
        problem["settings"],
        problem["track_compartments"],
        problem["track_accessibility"],
    )


class DensifyStage:
    """`Arced -> Densified`."""

    kind = StageKind.DENSIFY

    def bucket(self, inputs: tuple[State, ...]) -> int:
        # Not GPU-batched; bucket only groups the (serial-mapped) densify calls.
        return len(inputs[0].anchor_genomic)  # type: ignore[attr-defined]

    def to_problem(self, inputs: tuple[State, ...]) -> Problem:
        st = inputs[0]
        assert isinstance(st, Arced)
        return {
            "anchor_pos": st.anchor_pos,
            "anchor_genomic": st.anchor_genomic,
            "settings": st.settings,
            "track_compartments": st.track_compartments,
            "track_accessibility": st.track_accessibility,
        }

    def apply(self, inputs: tuple[State, ...], result: Result) -> State:
        st = inputs[0]
        assert isinstance(st, Arced)
        pos, fixed, starts, ends, dtn, anchor_map, step, compartment, accessibility = result
        return Densified(
            **vars(st),
            pos=pos,
            fixed=fixed,
            dtn=dtn,
            anchor_map=anchor_map,
            bead_starts=starts,
            bead_ends=ends,
            step_size_smooth=step,
            bead_compartment=compartment,
            bead_accessibility=accessibility,
        )
