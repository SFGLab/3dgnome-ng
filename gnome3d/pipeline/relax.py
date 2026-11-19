"""Cross block relaxation.

The excluded volume of the smooth stage acts within one block. This pass runs the smooth
kernel once over the whole chromosome with excluded volume on every pair and every anchor
fixed. Only the subanchor coils move. Bond targets are the bonds as they are.
"""

from __future__ import annotations

import copy

import numpy as np
from scipy.spatial import KDTree

from gnome3d import log
from gnome3d.settings import Settings
from gnome3d.types import BeadOut, BoolArray, F32Array, F64Array

LOG = log.get("relax")


def _positions(block: list[BeadOut]) -> F64Array:
    return np.array([[b.x, b.y, b.z] for b in block], dtype=np.float64)


def cross_block_contacts(blocks: list[list[BeadOut]], radius: float) -> tuple[int, int]:
    """Returns the number of bead pairs of different blocks closer than `radius` and the number
    of beads in them.

    Parameters
    ----------
    blocks
        The bead lists of one chromosome's blocks.
    radius
        The distance under which two beads touch, in model units.
    """
    pos = np.concatenate([_positions(b) for b in blocks]) if blocks else np.zeros((0, 3))
    owner = (
        np.concatenate([np.full(len(b), k) for k, b in enumerate(blocks)])
        if blocks
        else np.zeros(0)
    )
    if pos.shape[0] < 2:
        return 0, 0
    cross = _cross_pairs(pos, owner, radius)
    return int(cross.shape[0]), int(np.unique(cross.ravel()).size)


def _cross_pairs(pos: np.ndarray, owner: np.ndarray, radius: float) -> np.ndarray:
    """Returns the index pairs of beads of different blocks closer than `radius`."""
    if pos.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.int64)
    pairs = KDTree(pos).query_pairs(radius, output_type="ndarray")
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return pairs[owner[pairs[:, 0]] != owner[pairs[:, 1]]]


def _within_overlaps(pos: np.ndarray, owner: np.ndarray, radius: float) -> np.ndarray:
    """Returns the index pairs, in chain order, of non neighbour beads of one block closer than
    `radius`.
    """
    if pos.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.int64)
    pairs = KDTree(pos).query_pairs(radius, output_type="ndarray")
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    same = owner[pairs[:, 0]] == owner[pairs[:, 1]]
    far = np.abs(pairs[:, 0] - pairs[:, 1]) > 1
    return pairs[same & far]


def _relax_settings(s: Settings, bond: float) -> Settings:
    """Returns a copy of the settings with only chain bonds and excluded volume on."""
    r = copy.copy(s)
    r.use_excluded_volume = True
    r.exclusion_apply_to_smooth = True
    r.exclusion_weight = float(s.relax_ev_weight)
    # The excluded volume acts at 1.5 bonds by default. No pair is left under one bond. The
    # bonds carry their own weight. At the 0.1 of the smooth stage the excluded volume tears the
    # coil.
    r.exclusion_radius_smooth = float(s.relax_ev_radius) if s.relax_ev_radius > 0.0 else 1.5 * bond
    r.exclusion_skip_neighbors = 1
    r.spring_stretch = float(s.relax_bond_weight)
    r.spring_squeeze = float(s.relax_bond_weight)
    r.use_confinement = False
    r.use_compartments = False
    r.max_temp_smooth = float(s.max_temp_smooth) * float(s.relax_temp)
    # The wall and the anchor cap belong to the smooth stage. The pass fixes anchors and has a wider
    # radius.
    r.smooth_hard_wall = False
    r.smooth_anchor_cap = 0.0
    r.smooth_start = "line"
    r.mc_smooth_chains = 1
    return r


def relax_blocks(blocks: list[list[BeadOut]], s: Settings) -> list[list[BeadOut]]:
    """Returns the blocks with their overlaps removed by moving subanchors. Anchors do not move.

    Returns the input with `use_cross_block_relax` off, or with fewer than two blocks and
    `within_blocks` off.

    Parameters
    ----------
    blocks
        The bead lists of one chromosome's blocks, in any order. The chain is in genomic order.
    s
        Settings. Reads the `relax_*` keys and `mc_executor_smooth`.
    """
    within = bool(getattr(s, "relax_within_blocks", False))
    if not s.use_cross_block_relax or len(blocks) < (1 if within else 2):
        return blocks
    order = sorted(range(len(blocks)), key=lambda k: blocks[k][0].start if blocks[k] else 0)
    chain = [b for k in order for b in blocks[k]]
    if len(chain) < 3:
        return blocks
    pos: F32Array = np.array([[b.x, b.y, b.z] for b in chain], dtype=np.float32)
    fixed: BoolArray = np.array([b.kind == "anchor" for b in chain], dtype=np.bool_)
    bonds = np.linalg.norm(np.diff(pos.astype(np.float64), axis=0), axis=1)
    dtn: F32Array = bonds.astype(np.float32)
    bond = float(np.median(bonds)) if bonds.size else 1.0
    r = _relax_settings(s, bond)
    radius = bond  # contacts are counted at one bond
    before, _ = cross_block_contacts(blocks, radius)

    # With a window, only the beads that touch another block move, with that many chain neighbours
    # on either side.
    window = int(getattr(s, "relax_local_window", -1))
    if window >= 0:
        owner_chain = (
            np.concatenate([np.full(len(blocks[k]), k) for k in order]) if blocks else np.zeros(0)
        )
        touching = np.unique(_cross_pairs(pos, owner_chain, radius).ravel())
        if within:
            # Non neighbour pairs of one block under 0.7 of a bond move as well.
            inner = _within_overlaps(pos, owner_chain, 0.7 * bond)
            LOG.info("cross block relax: %d overlapping pairs inside blocks", inner.shape[0])
            touching = np.unique(np.concatenate([touching, inner.ravel()]))
        movable = np.zeros(len(chain), dtype=np.bool_)
        for off in range(-window, window + 1):
            idx = touching + off
            movable[idx[(idx >= 0) & (idx < len(chain))]] = True
        movable &= ~fixed
        LOG.info(
            "cross block relax: %d of %d beads may move (%d touching, window %d)",
            int(movable.sum()),
            len(chain),
            int(touching.size),
            window,
        )
        if not movable.any():
            return blocks
        fixed = ~movable

    step = float(s.relax_noise) * bond

    use_jax = str(s.mc_executor_smooth).strip().lower() == "batch"
    if use_jax:
        from gnome3d.mc.jax.util import jax_is_available

        use_jax = jax_is_available()
    if use_jax:
        from gnome3d.mc import jax as mc_jax

        out_pos = np.asarray(mc_jax.mc_smooth_jax(pos, dtn, fixed, step, r), dtype=np.float32)
        if out_pos.shape != pos.shape:
            out_pos = pos
    else:
        from gnome3d.mc import numba as mc_numba

        mc_numba.mc_smooth_numba(pos, dtn, fixed, step, r)
        out_pos = pos

    moved = [
        BeadOut(b.start, b.end, float(p[0]), float(p[1]), float(p[2]), b.kind)
        for b, p in zip(chain, out_pos, strict=True)
    ]
    out = list(blocks)
    i = 0
    for k in order:
        n = len(blocks[k])
        out[k] = moved[i : i + n]
        i += n
    after, touched = cross_block_contacts(out, radius)
    LOG.info(
        "cross block relax: %d beads, contacts within %.2f: %d -> %d (%d beads touched)",
        len(chain),
        radius,
        before,
        after,
        touched,
    )
    return out
