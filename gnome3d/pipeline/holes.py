"""Beads for the holes between blocks.

The stretch between the last anchor of one block and the first anchor of the next has no
beads. With `[holes] use_hole_beads` each hole gets one bead per `target_bp_per_subanchor`,
drawn from the maximum entropy bridge between its two anchors. A hole of more than 500 beads is
drawn in two levels, a bridge over every 500th bead and then a bridge between each two of
those. The drawn chains then run through the smooth kernel once over the whole chromosome,
with every block bead fixed and without confinement. `[holes] hard_wall = no` leaves the wall
out of that pass. `[holes] flank_beads` conditions the draw on that many beads of each
flanking block. The stretches before the first block and after the last stay empty. See
`design/gap-fill.md`.
"""

from __future__ import annotations

import copy
import zlib

import numpy as np

from gnome3d import log
from gnome3d.pipeline.ib.maxent import chain_targets, conditioned_draw, maxent_start
from gnome3d.settings import Settings
from gnome3d.types import BeadOut, F32Array
from gnome3d.util import add_movable_noise_inplace, seed_rng

LOG = log.get("holes")


_SEGMENT = 500


def hole_slots(end_a: int, start_b: int, target_bp: int) -> list[tuple[int, int]]:
    """Returns the genomic slots of the beads of a hole, one per `target_bp`, as densification
    lays out a gap.
    """
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
    flank: tuple[list[BeadOut], list[BeadOut]] = ([], []),
) -> F32Array:
    """Returns the positions of a chain whose first and last beads are fixed at `ends_pos`.

    `flank` holds block beads before the first bead and after the last, in genomic order. The top
    level of the hole is drawn conditioned on them too.
    """
    law = s.polymer_law()
    stretch = float(s.heatmap_distance_stretching)
    uri = str(s.data_contact_map)
    n = len(starts)
    knots = list(range(0, n - 1, _SEGMENT)) + [n - 1]
    pos = np.zeros((n, 3), dtype=np.float32)
    pos[0], pos[-1] = ends_pos[0], ends_pos[1]
    left, right = flank
    if left or right:
        # The top level is every bead of a short hole or the knots of a long one.
        top = knots if len(knots) > 2 else list(range(n))
        f_st = [x.start for x in left] + [starts[k] for k in top] + [x.start for x in right]
        f_en = [x.end for x in left] + [ends[k] for k in top] + [x.end for x in right]
        f_pos = np.concatenate(
            [
                np.array([[x.x, x.y, x.z] for x in left], dtype=np.float64).reshape(-1, 3),
                pos[top].astype(np.float64),
                np.array([[x.x, x.y, x.z] for x in right], dtype=np.float64).reshape(-1, 3),
            ]
        )
        fixed = np.zeros(len(f_st), dtype=np.bool_)
        fixed[: len(left) + 1] = True
        fixed[len(left) + len(top) - 1 :] = True
        d = chain_targets(f_st, f_en, law, stretch, "" if len(knots) > 2 else uri, chrom)
        drawn = conditioned_draw(f_pos, fixed, d, rng)
        pos[top] = drawn[len(left) : len(left) + len(top)].astype(np.float32)
        if len(knots) <= 2:
            return pos
    elif len(knots) > 2:
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


def _smooth(chain: list[BeadOut], hole: list[bool], s: Settings, seed: int) -> F32Array:
    """Returns the positions of `chain` after the smooth kernel with only the hole beads free.

    Parameters
    ----------
    chain
        The beads of one chromosome in genomic order, blocks and holes.
    hole
        Per bead, True for a hole bead.
    s
        Settings. The terms of the smooth stage are used, without confinement and the
        compartment term.
    seed
        Seed of the restart noise and of the kernel.
    """
    from gnome3d.mc import numba as mc_numba

    pos = np.array([[b.x, b.y, b.z] for b in chain], dtype=np.float32)
    fixed = ~np.asarray(hole, dtype=np.bool_)
    mids = np.array([(b.start + b.end) // 2 for b in chain], dtype=np.int64)
    dtn = np.array(
        [float(s.genomic_length_to_distance(int(g))) for g in np.maximum(np.diff(mids), 0)],
        dtype=np.float32,
    )
    # The target of a bond between two block beads is its own length. The pass does not change
    # the score of a block.
    both = fixed[:-1] & fixed[1:]
    dtn[both] = np.linalg.norm(np.diff(pos, axis=0), axis=1)[both]
    free = ~(fixed[:-1] & fixed[1:])
    mean_hole = float(dtn[free].mean())
    r = copy.copy(s)
    r.use_confinement = False
    r.use_compartments = False
    r.mc_smooth_chains = 1
    r.smooth_hard_wall = bool(s.smooth_hard_wall) and bool(s.holes_hard_wall)
    if float(s.exclusion_radius_smooth) <= 0.0:
        r.exclusion_radius_smooth = float(s.exclusion_auto_factor_smooth) * mean_hole
    step = mean_hole * float(s.noise_smooth)
    seed_rng(seed)
    mc_numba.seed_numba(seed)
    add_movable_noise_inplace(pos, fixed, step)
    mc_numba.mc_smooth_numba(pos, dtn, fixed, step, r)
    return pos


def fill_holes(
    blocks: list[list[BeadOut]], s: Settings, chrom: str, seed: int
) -> list[list[BeadOut]]:
    """Returns the blocks with one more bead list per hole.

    Parameters
    ----------
    blocks
        The bead lists of one chromosome's blocks, in any order.
    s
        Settings. Reads `use_hole_beads`, `flank_beads`, `hard_wall`, `target_bp_per_subanchor`
        and `[data] contact_map`.
    chrom
        The chromosome, for the contact map and the seed of the draw.
    seed
        The seed offset of the structure. Each hole draws from a generator seeded by it, the
        chromosome and the place of the hole.
    """
    if not s.use_hole_beads or len(blocks) < 2:
        return blocks
    nonempty = [b for b in blocks if b]
    order = sorted(nonempty, key=lambda b: min(x.start for x in b))
    target = int(s.target_bp_per_subanchor)
    out = list(blocks)
    drawn_holes: list[list[BeadOut]] = []
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
        nf = int(s.holes_flank_beads)
        flank: tuple[list[BeadOut], list[BeadOut]] = ([], [])
        if nf > 0:
            lb = sorted(order[k], key=lambda x: x.start)
            rb = sorted(order[k + 1], key=lambda x: x.start)
            flank = (
                [x for x in lb if x is not a][-nf:],
                [x for x in rb if x is not b][:nf],
            )
        drawn = _draw(starts, ends, ends_pos, s, rng, chrom, flank)
        drawn_holes.append(
            [
                BeadOut(lo, hi, float(p[0]), float(p[1]), float(p[2]), "subanchor")
                for (lo, hi), p in zip(slots, drawn[1:-1], strict=True)
            ]
        )
        n_holes += 1
        n_beads += len(slots)
        span_bp += b.start - a.end
    if drawn_holes:
        lists = nonempty + drawn_holes
        tag = [False] * len(nonempty) + [True] * len(drawn_holes)
        keyed = sorted(
            ((x, t) for blk, t in zip(lists, tag, strict=True) for x in blk),
            key=lambda xt: xt[0].start,
        )
        chain = [x for x, _ in keyed]
        moved = _smooth(
            chain,
            [t for _, t in keyed],
            s,
            int(
                np.random.default_rng([seed & 0x7FFFFFFF, zlib.crc32(chrom.encode())]).integers(
                    0, 0x7FFFFFFF
                )
            ),
        )
        where = {x.start: p for x, p in zip(chain, moved, strict=True)}
        for h in drawn_holes:
            out.append(
                [
                    BeadOut(
                        x.start,
                        x.end,
                        float(where[x.start][0]),
                        float(where[x.start][1]),
                        float(where[x.start][2]),
                        x.kind,
                    )
                    for x in h
                ]
            )
    log.status(
        LOG,
        "hole beads %s: %d holes, %d beads over %.1f Mb",
        chrom,
        n_holes,
        n_beads,
        span_bp / 1e6,
    )
    return out
