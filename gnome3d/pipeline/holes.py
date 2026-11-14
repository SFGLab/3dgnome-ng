"""Beads for the holes between blocks.

A block runs until a position no loop covers, so the stretch between the last anchor of one block
and the first anchor of the next has no beads. With `[holes] use_hole_beads` each such hole gets
one bead per `target_bp_per_subanchor`, the spacing inside a block, with no cap, drawn from the
maximum entropy bridge between the two anchors that bound it, the draw `start = maxent` makes for
a gap. A hole of more than 500 beads is drawn in two levels, a bridge over every 500th bead and
then a bridge between each two of those, so the cost stays linear in the hole's length. A segment
is conditioned on its two ends alone, which leaves out its correlation with the rest of the hole.
The contact map scales the segments and not the coarse level, where beads 500 kb apart share
almost no counts at 1 kb. The drawn chain then runs through the smooth stage's kernel with its
two ends fixed, the terms a gap inside a block runs through, so a hole's short range matches the
blocks'. Excluded volume acts within a hole and not against the blocks.
The stretches before the first block and after the last are left empty, since a bridge needs two
ends. See `design/gap-fill.md`.
"""

from __future__ import annotations

import zlib

import numpy as np

from gnome3d import log
from gnome3d.pipeline.ib.maxent import maxent_start
from gnome3d.settings import Settings
from gnome3d.types import BeadOut, F32Array
from gnome3d.util import add_movable_noise_inplace, seed_rng

LOG = log.get("holes")


_SEGMENT = 500


def hole_slots(end_a: int, start_b: int, target_bp: int) -> list[tuple[int, int]]:
    """The genomic slots of a hole's beads, one per `target_bp`, laid out as densification lays
    out a gap's."""
    span = start_b - end_a
    n = round(span / max(target_bp, 1)) - 1
    if span <= 0 or n < 1:
        return []
    d_bp = span // (n + 1)
    lo = d_bp // 2
    hi = d_bp - lo
    return [(end_a + (j + 1) * d_bp - lo, end_a + (j + 1) * d_bp + hi) for j in range(n)]


def _draw(
    starts: list[int],
    ends: list[int],
    ends_pos: F32Array,
    s: Settings,
    rng: np.random.Generator,
    chrom: str,
) -> F32Array:
    """Positions of a chain whose first and last beads are fixed at `ends_pos`."""
    law = s.polymer_law()
    stretch = float(s.heatmap_distance_stretching)
    uri = str(s.data_contact_map)
    n = len(starts)
    knots = list(range(0, n - 1, _SEGMENT)) + [n - 1]
    pos = np.zeros((n, 3), dtype=np.float32)
    pos[0], pos[-1] = ends_pos[0], ends_pos[1]
    if len(knots) > 2:
        fixed = np.zeros(len(knots), dtype=np.bool_)
        fixed[[0, -1]] = True
        coarse = maxent_start(
            pos[knots],
            fixed,
            [starts[k] for k in knots],
            [ends[k] for k in knots],
            law,
            stretch,
            rng,
        )
        pos[knots] = coarse
    for lo, hi in zip(knots[:-1], knots[1:], strict=True):
        fixed = np.zeros(hi - lo + 1, dtype=np.bool_)
        fixed[[0, -1]] = True
        pos[lo : hi + 1] = maxent_start(
            pos[lo : hi + 1],
            fixed,
            starts[lo : hi + 1],
            ends[lo : hi + 1],
            law,
            stretch,
            rng,
            uri,
            chrom,
        )
    return pos


def _smooth(pos: F32Array, starts: list[int], ends: list[int], s: Settings, seed: int) -> None:
    """Run the smooth stage's kernel over a hole chain in place, its two ends fixed, with bond
    targets and step set as densification sets a block's."""
    from gnome3d.mc import numba as mc_numba

    mids = (np.asarray(starts, dtype=np.int64) + np.asarray(ends, dtype=np.int64)) // 2
    dtn = np.array(
        [float(s.genomic_length_to_distance(int(g))) for g in np.maximum(np.diff(mids), 0)],
        dtype=np.float32,
    )
    fixed = np.zeros(len(pos), dtype=np.bool_)
    fixed[[0, -1]] = True
    step = float(dtn.mean()) * float(s.noise_smooth)
    seed_rng(seed)
    mc_numba.seed_numba(seed)
    add_movable_noise_inplace(pos, fixed, step)
    mc_numba.mc_smooth_numba(pos, dtn, fixed, step, s)


def fill_holes(
    blocks: list[list[BeadOut]], s: Settings, chrom: str, seed: int
) -> list[list[BeadOut]]:
    """The blocks with one more bead list per hole between two of them.

    Parameters
    ----------
    blocks
        One chromosome's per block bead lists, in any order.
    s
        Settings. Reads `use_hole_beads`, `target_bp_per_subanchor` and `[data] contact_map`.
    chrom
        The chromosome, for the contact map and the draw's seed.
    seed
        The structure's seed offset. Each hole draws from a generator seeded by it, the
        chromosome and the hole's place, so a structure reproduces.
    """
    if not s.use_hole_beads or len(blocks) < 2:
        return blocks
    nonempty = [b for b in blocks if b]
    order = sorted(nonempty, key=lambda b: min(x.start for x in b))
    target = int(s.target_bp_per_subanchor)
    out = list(blocks)
    n_holes = n_beads = 0
    span_bp = 0
    for k in range(len(order) - 1):
        a = max(order[k], key=lambda x: x.end)
        b = min(order[k + 1], key=lambda x: x.start)
        slots = hole_slots(a.end, b.start, target)
        if not slots:
            continue
        starts = [a.start] + [lo for lo, _ in slots] + [b.start]
        ends = [a.end] + [hi for _, hi in slots] + [b.end]
        rng = np.random.default_rng([seed & 0x7FFFFFFF, zlib.crc32(chrom.encode()), k])
        ends_pos = np.array([(a.x, a.y, a.z), (b.x, b.y, b.z)], dtype=np.float32)
        drawn = _draw(starts, ends, ends_pos, s, rng, chrom)
        _smooth(drawn, starts, ends, s, int(rng.integers(0, 0x7FFFFFFF)))
        out.append(
            [
                BeadOut(lo, hi, float(p[0]), float(p[1]), float(p[2]), "subanchor")
                for (lo, hi), p in zip(slots, drawn[1:-1], strict=True)
            ]
        )
        n_holes += 1
        n_beads += len(slots)
        span_bp += b.start - a.end
    log.status(
        LOG,
        "hole beads %s: %d holes, %d beads over %.1f Mb",
        chrom,
        n_holes,
        n_beads,
        span_bp / 1e6,
    )
    return out
