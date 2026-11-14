"""Beads for the holes between blocks.

A block runs until a position no loop covers, so the stretch between the last anchor of one block
and the first anchor of the next has no beads. With `[holes] use_hole_beads` each such hole gets
beads at the count the dynamic loop density gives a gap of that span, drawn from the maximum
entropy bridge between the two anchors that bound it, the draw `start = maxent` makes for a gap.
The stretches before the first block and after the last are left empty, since a bridge needs two
ends. See `design/gap-fill.md`.
"""

from __future__ import annotations

import zlib

import numpy as np

from gnome3d import log
from gnome3d.pipeline.ib.maxent import maxent_start
from gnome3d.settings import Settings
from gnome3d.types import BeadOut

LOG = log.get("holes")


def hole_slots(end_a: int, start_b: int, s: Settings) -> list[tuple[int, int]]:
    """The genomic slots of a hole's beads, laid out as densification lays out a gap's."""
    span = start_b - end_a
    if span <= 0:
        return []
    target = max(int(s.target_bp_per_subanchor), 1)
    mn = max(int(s.min_subanchors_per_arc), 0)
    n = max(mn, min(max(int(s.max_subanchors_per_arc), mn), round(span / target) - 1))
    if n < 1:
        return []
    d_bp = span // (n + 1)
    lo = d_bp // 2
    hi = d_bp - lo
    return [(end_a + (j + 1) * d_bp - lo, end_a + (j + 1) * d_bp + hi) for j in range(n)]


def fill_holes(
    blocks: list[list[BeadOut]], s: Settings, chrom: str, seed: int
) -> list[list[BeadOut]]:
    """The blocks with one more bead list per hole between two of them.

    Parameters
    ----------
    blocks
        One chromosome's per block bead lists, in any order.
    s
        Settings. Reads `use_hole_beads`, the dynamic loop density keys and `[data] contact_map`.
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
    law = s.polymer_law()
    stretch = float(s.heatmap_distance_stretching)
    out = list(blocks)
    n_holes = n_beads = 0
    span_bp = 0
    for k in range(len(order) - 1):
        a = max(order[k], key=lambda x: x.end)
        b = min(order[k + 1], key=lambda x: x.start)
        slots = hole_slots(a.end, b.start, s)
        if not slots:
            continue
        starts = [a.start] + [lo for lo, _ in slots] + [b.start]
        ends = [a.end] + [hi for _, hi in slots] + [b.end]
        pos = np.zeros((len(starts), 3), dtype=np.float32)
        pos[0] = (a.x, a.y, a.z)
        pos[-1] = (b.x, b.y, b.z)
        fixed = np.zeros(len(starts), dtype=np.bool_)
        fixed[[0, -1]] = True
        rng = np.random.default_rng([seed & 0x7FFFFFFF, zlib.crc32(chrom.encode()), k])
        drawn = maxent_start(
            pos, fixed, starts, ends, law, stretch, rng, str(s.data_contact_map), chrom
        )
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
