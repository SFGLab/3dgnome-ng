"""Cross block relaxation after the stitch.

The smooth stage's excluded volume acts within one block and the stitch guards block centroids
only, so once blocks are stitched together nothing acts between their beads and two coils can
pass through each other. This runs the smooth kernel once over the whole chromosome with
excluded volume on every pair and every anchor held fixed, so the arcs and the stitch are kept
and only the subanchor coils re route around each other. Bond targets are the bonds as they
are, so the pass changes nothing it does not have to.

The gate is `cross_block_contacts`, bead pairs of different blocks inside the excluded volume
radius, which the pass exists to drive to zero.
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
    """Pairs of beads from different blocks closer than `radius`, and how many beads take part.

    Parameters
    ----------
    blocks
        One chromosome's per block bead lists.
    radius
        The distance below which two beads count as touching, in model units.
    """
    pos = np.concatenate([_positions(b) for b in blocks]) if blocks else np.zeros((0, 3))
    owner = (
        np.concatenate([np.full(len(b), k) for k, b in enumerate(blocks)])
        if blocks
        else np.zeros(0)
    )
    if pos.shape[0] < 2:
        return 0, 0
    pairs = KDTree(pos).query_pairs(radius, output_type="ndarray")
    if pairs.size == 0:
        return 0, 0
    cross = pairs[owner[pairs[:, 0]] != owner[pairs[:, 1]]]
    return int(cross.shape[0]), int(np.unique(cross.ravel()).size)


def _relax_settings(s: Settings, bond: float) -> Settings:
    """A copy of the settings with only chain bonds and excluded volume active."""
    r = copy.copy(s)
    r.use_excluded_volume = True
    r.exclusion_apply_to_smooth = True
    r.exclusion_weight = float(s.relax_ev_weight)
    # The excluded volume acts at 1.5 bonds by default so that at equilibrium nothing is left
    # under one bond, which is where contacts are counted. Bonds carry their own weight, since
    # at the smooth stage's 0.1 the excluded volume tears the coil instead of re routing it.
    r.exclusion_radius_smooth = float(s.relax_ev_radius) if s.relax_ev_radius > 0.0 else 1.5 * bond
    r.exclusion_skip_neighbors = 1
    r.spring_stretch = float(s.relax_bond_weight)
    r.spring_squeeze = float(s.relax_bond_weight)
    r.use_confinement = False
    r.use_compartments = False
    r.use_bridging = False
    r.use_fibre_compaction = False
    r.max_temp_smooth = float(s.max_temp_smooth) * float(s.relax_temp)
    r.mc_smooth_chains = 1
    return r


def relax_blocks(blocks: list[list[BeadOut]], s: Settings) -> list[list[BeadOut]]:
    """Return the blocks with their subanchors moved so no two blocks interpenetrate. Anchors do
    not move. With `use_cross_block_relax` off, or fewer than two blocks, the input is returned
    as is.

    Parameters
    ----------
    blocks
        One chromosome's per block bead lists, in any order. The chain is taken in genomic order.
    s
        Settings. Reads the `relax_*` keys and `mc_executor_smooth` to pick the kernel.
    """
    if not s.use_cross_block_relax or len(blocks) < 2:
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
    radius = bond  # contacts are counted at one bond; the excluded volume acts wider
    before, _ = cross_block_contacts(blocks, radius)
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
