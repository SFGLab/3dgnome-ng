"""The smooth MC on JAX: chain bonds and angles, with optional excluded volume, heat,
orientation, confinement and the compartment term.

`mc_smooth_jax` anneals one problem. `mc_smooth_jax_batch` anneals K different IBs in one
vmapped kernel. Both compile through `_build_smooth_kernel`, which caches the kernel.
"""

import logging
import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from gnome3d import log
from gnome3d.mc.jax.memory import max_k_for_bytes, max_k_for_host
from gnome3d.mc.jax.util import (
    ANCHOR_BUCKETS,
    NBR_BUCKETS,
    jax_bucket_for,
    jax_device_budget_bytes,
    jax_is_available,
    log_kernel_done,
    log_kernel_start,
    stable_seed_offset,
)
from gnome3d.mc.numba.common import SMOOTH_STOP_RATIO, orientation_weight, sites_of_moves
from gnome3d.mc.numba.terms import ORIENTATION_STRETCH
from gnome3d.types import F32Array, I32Array, I64Array

# The sites that one move reads at most: those within the stretch of a node and its two neighbours.
MOVE_SITES = 2 * ORIENTATION_STRETCH + 3
# The beads around a move that the vectors of those sites read.
ORN_WINDOW = 4 * ORIENTATION_STRETCH + 3

if TYPE_CHECKING:
    from gnome3d.settings import Settings

LOG = log.get("mc.jax")

_kernel_cache: dict[Any, Any] = {}


GRID_BUCKETS: tuple[int, ...] = (8, 16, 24, 32, 48)


def _grid_static(structs: list[tuple[np.ndarray[Any, Any], float]]) -> tuple[int, int]:
    """Returns the cells per axis and the beads per cell of a launch, from its starting structures.

    The grid is bucketed. The compiled shape depends on the coarse size of the launch.
    Cells are at least `r0` wide and wider when the structure does not fit 48 cells across. The
    capacity is twice the fullest starting cell plus four, rounded up to four. The kernel does not
    detect a cell that fills past it later.
    """
    g_all = 0
    cap = 0
    for pos, r0 in structs:
        if pos.shape[0] == 0:
            continue
        ext = float(np.max(pos.max(axis=0) - pos.min(axis=0)))
        need = int(np.ceil(ext / max(r0, 1e-6))) + 3
        g = next((b for b in GRID_BUCKETS if b >= need), GRID_BUCKETS[-1])
        w = max(r0, ext / (g - 2)) * (1.0 + 1e-5)
        lo = pos.min(axis=0) - w
        c = np.clip(np.floor((pos - lo) / w).astype(np.int64), 0, g - 1)
        lin = (c[:, 0] * g + c[:, 1]) * g + c[:, 2]
        occ = int(np.bincount(lin).max())
        g_all = max(g_all, g)
        cap = max(cap, 2 * occ + 4)
    return g_all, int(np.ceil(cap / 4.0) * 4)


def _build_smooth_kernel(
    n_steps_per_batch: int,
    excl_skip: int,
    use_heat: bool,
    use_orn: bool,
    max_nbrs: int,
    use_aff: bool = False,
    use_wall: bool = False,
    use_cap: bool = False,
    prefetch: int = 1,
    grid_g: int = 0,
    grid_cap: int = 0,
    use_centre: bool = False,
) -> Any:
    """Builds the compiled smooth kernel, or returns the cached one.

    `grid_g` above zero puts the excluded volume and the wall on a cell grid of that many cells
    per axis, each with up to `grid_cap` beads. A proposal then visits the 27 cells around it.
    Both are array shapes and static. `_grid_static` picks them.

    `prefetch` is the number of proposals one step evaluates against the current state. Above 1
    the step applies the first proposal in draw order that passes the Metropolis test and the hard
    rules and discards the rest. Every applied move is a Metropolis move from the state it was
    drawn on. At 1 the body is the serial one.

    `use_wall` and `use_cap` are the hard rules of the smooth stage. The wall rejects a move that
    adds a non neighbour pair under the excluded volume radius or deepens one. The cap rejects a
    move that takes a capped bead further than `cap_r` from `cap_home`. Neither draws a random
    number.

    `use_centre` is the three bead move of the uniform chain. A node keeps the centre of mass of
    itself, the bead before it and the bead after it. `centre_of` gives that node for each of the
    three beads, -1 for any other bead. A proposal on the node moves it by the draw and its two
    neighbours by minus half of it. A proposal on a neighbour moves it by the draw and the other
    neighbour by minus it. The wall judges the whole move.

    `move_sites` gives the orientation sites that the move of a bead reads, a row padded with
    -1: the sites within `ORIENTATION_STRETCH` beads of the bead, or of the three beads of its
    node. Each is scored again with its vector over the stretch of chain around it.

    Every argument is part of the cache key. `use_aff` is static. The affinity term costs a
    pass per step. The compartment energy is added into the excluded volume score `se`. Both are
    pairwise and double counted. The sum is exact and only the breakdown per term is lost.

    Returns the kernels and the init functions, which compute the initial scores on the device.
    """
    cache_key = (
        n_steps_per_batch,
        excl_skip,
        use_heat,
        use_orn,
        max_nbrs,
        use_aff,
        use_wall,
        use_cap,
        int(prefetch),
        int(grid_g),
        int(grid_cap),
        bool(use_centre),
    )
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    import jax
    import jax.numpy as jnp

    def _smooth_len(pa: Any, pb: Any, e: Any, stretch_k: Any, squeeze_k: Any, dist_w: Any) -> Any:
        diff = pa - pb
        d = jnp.sqrt(jnp.sum(diff * diff))
        e_safe = jnp.maximum(e, 1e-6)
        rel = (d - e_safe) / e_safe
        k = jnp.where(rel >= 0, stretch_k, squeeze_k)
        return rel * rel * k * dist_w

    def _smooth_ang(pa: Any, pb: Any, pc: Any, ang_k: Any, ang_w: Any) -> Any:
        v1 = pa - pb
        v2 = pb - pc
        n1 = jnp.sqrt(jnp.sum(v1 * v1))
        n2 = jnp.sqrt(jnp.sum(v2 * v2))
        scale = jnp.where(jnp.logical_or(n1 < 1e-12, n2 < 1e-12), 0.0, 1.0)
        cos_a = jnp.sum(v1 * v2) / jnp.maximum(n1 * n2, 1e-30)
        cos_a = jnp.clip(cos_a, -1.0, 1.0)
        ang = 1.0 - (cos_a + 1.0) * 0.5
        return scale * ang * ang * ang * ang_k * ang_w

    def _local_smooth_at(
        pos: Any,
        p_pos: Any,
        p: Any,
        dtn: Any,
        stretch_k: Any,
        squeeze_k: Any,
        ang_k: Any,
        dist_w: Any,
        ang_w: Any,
        n_active: Any,
    ) -> Any:
        # `n_active` is the real chain length. Bonds and angles that span a pad bead are masked. Pad
        # beads are a contiguous tail.
        n = pos.shape[0]
        a_pm1 = pos[jnp.maximum(p - 1, 0)]
        bond_L_ok = jnp.logical_and(p - 1 >= 0, p - 1 < n_active - 1)
        bond_L = jnp.where(
            bond_L_ok,
            _smooth_len(a_pm1, p_pos, dtn[jnp.maximum(p - 1, 0)], stretch_k, squeeze_k, dist_w),
            0.0,
        )
        a_pp1 = pos[jnp.minimum(p + 1, n - 1)]
        bond_R_ok = jnp.logical_and(p >= 0, p < n_active - 1)
        bond_R = jnp.where(
            bond_R_ok,
            _smooth_len(p_pos, a_pp1, dtn[jnp.minimum(p, n - 2)], stretch_k, squeeze_k, dist_w),
            0.0,
        )

        def angle_at(off: int) -> Any:
            i = p + off
            i0 = jnp.clip(i, 0, n - 1)
            i1 = jnp.clip(i + 1, 0, n - 1)
            i2 = jnp.clip(i + 2, 0, n - 1)
            a0 = pos[i0]
            a1 = pos[i1]
            a2 = pos[i2]
            a0 = jnp.where(i == p, p_pos, a0)
            a1 = jnp.where(i + 1 == p, p_pos, a1)
            a2 = jnp.where(i + 2 == p, p_pos, a2)
            valid = jnp.logical_and(i >= 0, i < n_active - 2)
            return jnp.where(valid, _smooth_ang(a0, a1, a2, ang_k, ang_w), 0.0)

        return bond_L + bond_R + angle_at(-2) + angle_at(-1) + angle_at(0)

    def _local_excl_at(pos: Any, p_pos: Any, p: Any, r0: Any, weight: Any, n_active: Any) -> Any:
        n = pos.shape[0]
        diff = pos - p_pos
        d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
        rel = jnp.maximum(0.0, (r0 - d) / r0)
        contrib = weight * rel * rel
        idx = jnp.arange(n)
        # Pad beads, idx >= n_active, are left out.
        in_range = jnp.logical_and(jnp.abs(idx - p) > excl_skip, idx < n_active)
        return jnp.sum(jnp.where(in_range, contrib, 0.0))

    def _wall_at(pos: Any, p_pos: Any, p: Any, r0: Any, n_active: Any) -> tuple[Any, Any]:
        """Returns the number of non neighbour beads under `r0` from `p_pos` and their total depth."""
        n = pos.shape[0]
        diff = pos - p_pos
        d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
        idx = jnp.arange(n)
        in_range = jnp.logical_and(jnp.abs(idx - p) > excl_skip, idx < n_active)
        under = jnp.logical_and(in_range, d < r0)
        cnt = jnp.sum(under.astype(jnp.int32))
        depth = jnp.sum(jnp.where(under, r0 - d, 0.0))
        return cnt, depth

    # The cell grid.
    #
    # Cells are `w` wide, at least `r0`. The 27 cells around a point hold every bead within `r0`.
    # Beads outside the box are clamped into the edge cells and so is the cell of a query. Row `C`
    # holds the pad beads and is not queried. Row `C + 1` is empty and stands for a neighbour cell
    # out of range.
    #
    # The table is not written between rebuilds. A bead that an accepted move carries away is
    # flagged and listed. A query takes flagged beads from the list at their current position and
    # skips them in the cells. The sum is exact. The table is rebuilt every `grid_period`
    # batches. A write per move would be a batched scatter under the chain vmap, which XLA does out
    # of place.
    n_cells = int(grid_g) ** 3
    grid_period = max(1, 2048 // max(int(prefetch), 1)) if grid_g > 0 else 1
    # A three bead move lists up to three beads a step.
    move_parts = 3 if use_centre else 1
    offs27 = jnp.asarray(
        [[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], dtype=jnp.int32
    )

    def _cell_of(x: Any, lo3: Any, w: Any) -> Any:
        c3 = jnp.clip(jnp.floor((x - lo3) / w).astype(jnp.int32), 0, grid_g - 1)
        return (c3[..., 0] * grid_g + c3[..., 1]) * grid_g + c3[..., 2]

    def _grid_build(pos: Any, n_active: Any, r0: Any) -> tuple[Any, ...]:
        n = pos.shape[0]
        valid = jnp.arange(n) < n_active
        big = jnp.float32(3e38)
        lo3 = jnp.min(jnp.where(valid[:, None], pos, big), axis=0)
        hi3 = jnp.max(jnp.where(valid[:, None], pos, -big), axis=0)
        extent = jnp.max(hi3 - lo3)
        w = jnp.maximum(r0, extent / (grid_g - 2)) * (1.0 + 1e-5)
        lo3 = lo3 - w
        cid = jnp.where(valid, _cell_of(pos, lo3, w), n_cells)
        order = jnp.argsort(cid)
        sc = cid[order]
        first = jnp.searchsorted(sc, sc, side="left")
        rank_c = jnp.minimum(jnp.arange(n) - first, grid_cap - 1)
        cell_beads = jnp.full((n_cells + 2, grid_cap), -1, dtype=jnp.int32)
        cell_beads = cell_beads.at[sc, rank_c].set(order.astype(jnp.int32))
        moved = jnp.full((grid_period * move_parts,), -1, dtype=jnp.int32)
        flag = jnp.zeros((n,), dtype=jnp.bool_)
        return cell_beads, lo3, w, moved, flag

    def _grid_candidates(grid: Any, x: Any) -> tuple[Any, Any]:
        cell_beads, lo3, w, moved, _flag = grid
        c3 = jnp.clip(jnp.floor((x - lo3) / w).astype(jnp.int32), 0, grid_g - 1)
        nb = c3[None, :] + offs27
        inr = jnp.all(jnp.logical_and(nb >= 0, nb < grid_g), axis=1)
        lin = (nb[:, 0] * grid_g + nb[:, 1]) * grid_g + nb[:, 2]
        lin = jnp.where(inr, lin, n_cells + 1)
        return cell_beads[lin].reshape(-1), moved

    def _grid_pairs(pos: Any, p_pos: Any, p: Any, n_active: Any, grid: Any) -> tuple[Any, Any]:
        """Returns the candidate distances from `p_pos` and the mask of those that count."""
        flag = grid[4]
        idx_c, idx_m = _grid_candidates(grid, p_pos)
        idx = jnp.concatenate([idx_c, idx_m])
        idx_i = jnp.maximum(idx, 0)
        listed = jnp.concatenate(
            [jnp.logical_not(flag[jnp.maximum(idx_c, 0)]), jnp.ones(idx_m.shape, dtype=jnp.bool_)]
        )
        ok = jnp.logical_and(
            jnp.logical_and(idx >= 0, listed),
            jnp.logical_and(idx < n_active, jnp.abs(idx - p) > excl_skip),
        )
        diff = pos[idx_i] - p_pos
        return jnp.sqrt(jnp.sum(diff * diff, axis=1)), ok

    def _local_excl_grid(
        pos: Any, p_pos: Any, p: Any, r0: Any, weight: Any, n_active: Any, grid: Any
    ) -> Any:
        d, ok = _grid_pairs(pos, p_pos, p, n_active, grid)
        rel = jnp.maximum(0.0, (r0 - d) / r0)
        return jnp.sum(jnp.where(ok, weight * rel * rel, 0.0))

    def _wall_grid(
        pos: Any, p_pos: Any, p: Any, r0: Any, n_active: Any, grid: Any
    ) -> tuple[Any, Any]:
        d, ok = _grid_pairs(pos, p_pos, p, n_active, grid)
        under = jnp.logical_and(ok, d < r0)
        return jnp.sum(under.astype(jnp.int32)), jnp.sum(jnp.where(under, r0 - d, 0.0))

    def _grid_note(grid: Any, p: Any, ok: Any, slot: Any) -> tuple[Any, ...]:
        """Records that bead `p` moved, in the slot of this period, once per bead per period."""
        cell_beads, lo3, w, moved, flag = grid
        first_time = jnp.logical_and(ok, jnp.logical_not(flag[p]))
        moved = moved.at[slot].set(jnp.where(first_time, p, moved[slot]))
        flag = flag.at[p].set(jnp.logical_or(flag[p], ok))
        return cell_beads, lo3, w, moved, flag

    # Confinement, a soft envelope per bead.
    #
    # E(p) = weight * ((|r_p - c| - R) / R)^2 if |r_p - c| > R, else 0
    #
    # Single counted. Always in the kernel and a weight of 0 turns it off.

    def _local_confine_at(p_pos: Any, cx: Any, cy: Any, cz: Any, R: Any, weight: Any) -> Any:
        dx = p_pos[0] - cx
        dy = p_pos[1] - cy
        dz = p_pos[2] - cz
        r = jnp.sqrt(dx * dx + dy * dy + dz * dz)
        rel = (r - R) / jnp.maximum(R, 1e-30)
        contrib = weight * rel * rel
        return jnp.where(r > R, contrib, 0.0)

    # Affinity, the compartment term, as `local_affinity_nb` computes it.
    #
    # E_pair = w * g * (1 - exp(-d^2 / (2 r0^2))) / (N - 1), with g = Ea or Eb for a pair of one
    # compartment.

    def _local_affinity_at(
        pos: Any,
        p_pos: Any,
        p: Any,
        comp_cls: Any,
        comp_r0: Any,
        comp_w: Any,
        comp_ea: Any,
        comp_eb: Any,
        n_active: Any,
    ) -> Any:
        n = pos.shape[0]
        diff = pos - p_pos
        d2 = jnp.sum(diff * diff, axis=1)
        idx = jnp.arange(n)
        live = jnp.logical_and(idx != p, idx < n_active)
        inv_n = 1.0 / jnp.maximum(n_active - 1, 1).astype(pos.dtype)

        ci = comp_cls[p]
        both_a = jnp.logical_and(ci > 0, comp_cls > 0)
        both_b = jnp.logical_and(ci < 0, comp_cls < 0)
        g_comp = jnp.where(both_a, comp_ea, jnp.where(both_b, comp_eb, 0.0))
        e = comp_w * g_comp * (1.0 - jnp.exp(-d2 / (2.0 * comp_r0 * comp_r0)))
        return jnp.sum(jnp.where(live, e, 0.0)) * inv_n

    def _local_heat_at(pos: Any, p_pos: Any, p: Any, heat_dist: Any, heat_weight: Any) -> Any:
        """Returns the heat score of bead p placed at `p_pos`, the sum over i != p with
        heat_dist[i, p] > 0 of ((d - heat_dist[i, p]) / heat_dist[i, p])^2 * heat_weight.
        """
        n = pos.shape[0]
        diff = pos - p_pos
        d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
        exp_d = heat_dist[:, p]  # (N,)
        idx = jnp.arange(n)
        # i == p and pairs without contact data, heat_dist < 1e-6, are skipped.
        active = jnp.logical_and(idx != p, exp_d >= 1e-6)
        exp_d_safe = jnp.maximum(exp_d, 1e-6)
        rel = (d - exp_d_safe) / exp_d_safe
        contrib = rel * rel
        return heat_weight * jnp.sum(jnp.where(active, contrib, 0.0))

    def _stretch_vectors(pos: Any, ar: Any, is_L: Any, last: Any) -> Any:
        """Returns the unit orientation vectors of the anchors at the beads `ar` of `pos`, each
        from the centroid of the `ORIENTATION_STRETCH` beads before it to the centroid of those
        after it, as `calc_orientation_stretch_nb`. `last` is the last bead of the chain in
        `pos`. A side with fewer beads has the centroid of those there are and a side with none
        has the anchor bead.
        """
        n = pos.shape[0]
        off = np.arange(-ORIENTATION_STRETCH, ORIENTATION_STRETCH + 1)
        idx = ar[:, None] + off[None, :]
        inside = jnp.logical_and(idx >= 0, idx <= jnp.minimum(last, n - 1))
        pts = pos[jnp.clip(idx, 0, n - 1)]
        own = pos[jnp.clip(ar, 0, n - 1)]

        def centroid(side: Any) -> Any:
            mask = jnp.logical_and(inside, side[None, :])
            count = jnp.sum(mask, axis=1)
            total = jnp.sum(jnp.where(mask[:, :, None], pts, 0.0), axis=1)
            mean = total / jnp.maximum(count, 1)[:, None].astype(pos.dtype)
            return jnp.where((count > 0)[:, None], mean, own)

        o = centroid(off > 0) - centroid(off < 0)
        o = jnp.where(is_L[:, None], -o, o)
        nm = jnp.sqrt(jnp.sum(o * o, axis=1, keepdims=True))
        return jnp.where(nm > 1e-12, o / jnp.maximum(nm, 1e-30), jnp.zeros_like(o))

    # One pair of beads, as the sums over every other bead count it. The three bead move reads
    # them for the pairs inside its three beads.

    def _pair_excl(x: Any, y: Any, r0: Any, weight: Any) -> Any:
        diff = y - x
        d = jnp.sqrt(jnp.sum(diff * diff))
        rel = jnp.maximum(0.0, (r0 - d) / r0)
        return weight * rel * rel

    def _pair_wall(x: Any, y: Any, r0: Any) -> tuple[Any, Any]:
        diff = y - x
        d = jnp.sqrt(jnp.sum(diff * diff))
        under = d < r0
        return under.astype(jnp.int32), jnp.where(under, r0 - d, 0.0)

    def _pair_heat(x: Any, y: Any, exp_d: Any, heat_weight: Any) -> Any:
        diff = y - x
        d = jnp.sqrt(jnp.sum(diff * diff))
        exp_d_safe = jnp.maximum(exp_d, 1e-6)
        rel = (d - exp_d_safe) / exp_d_safe
        return jnp.where(exp_d >= 1e-6, heat_weight * rel * rel, 0.0)

    def _pair_affinity(
        x: Any, y: Any, ci: Any, cj: Any, comp_r0: Any, comp_w: Any, comp_ea: Any, comp_eb: Any,
        n_active: Any,
    ) -> Any:  # fmt: skip
        diff = y - x
        d2 = jnp.sum(diff * diff)
        inv_n = 1.0 / jnp.maximum(n_active - 1, 1).astype(x.dtype)
        both_a = jnp.logical_and(ci > 0, cj > 0)
        both_b = jnp.logical_and(ci < 0, cj < 0)
        g_comp = jnp.where(both_a, comp_ea, jnp.where(both_b, comp_eb, 0.0))
        return comp_w * g_comp * (1.0 - jnp.exp(-d2 / (2.0 * comp_r0 * comp_r0))) * inv_n

    def _local_orientation_after(
        anchor_orn: Any,
        k: Any,
        nbr_idx: Any,
        nbr_w: Any,
        nbr_valid: Any,
        motif_weight: Any,
        symmetric: Any,
        a_vec: Any,
        moved_k: Any,
        moved_vec: Any,
        moved_on: Any,
    ) -> Any:
        """Returns the orientation score of anchor k with the vector `a_vec` for it and the
        vectors `moved_vec` for the anchors `moved_k` where `moved_on`. No array is written.
        """
        neighbors_k = nbr_idx[k]
        a = a_vec
        b = anchor_orn[neighbors_k]
        for j in range(moved_k.shape[0]):
            swap = jnp.logical_and(moved_on[j], neighbors_k == moved_k[j])
            b = jnp.where(swap[:, None], moved_vec[j][None, :], b)
        b = jnp.where((neighbors_k == k)[:, None], a[None, :], b)
        b_signed = jnp.where(symmetric, b, -b)
        dot = jnp.sum(a[None, :] * b_signed, axis=1)
        ang = 1.0 - (dot + 1.0) * 0.5
        contrib = jnp.where(nbr_valid[k], ang * ang * nbr_w[k], 0.0)
        return motif_weight * jnp.sum(contrib)

    def _local_orientation_at(
        anchor_orn: Any,
        k: Any,
        nbr_idx: Any,
        nbr_w: Any,
        nbr_valid: Any,
        motif_weight: Any,
        symmetric: Any,
    ) -> Any:
        """Returns the orientation score of anchor k over its padded neighbours."""
        a = anchor_orn[k]
        b = anchor_orn[nbr_idx[k]]
        b_signed = jnp.where(symmetric, b, -b)
        dot = jnp.sum(a[None, :] * b_signed, axis=1)
        ang = 1.0 - (dot + 1.0) * 0.5
        contrib = jnp.where(nbr_valid[k], ang * ang * nbr_w[k], 0.0)
        return motif_weight * jnp.sum(contrib)

    def _sites_delta(
        work: Any,
        w0: Any,
        last: Any,
        sites: Any,
        anchor_orn: Any,
        anchor_ar: Any,
        is_L: Any,
        nbr_idx: Any,
        nbr_w: Any,
        nbr_valid: Any,
        motif_weight: Any,
        symmetric: Any,
    ) -> tuple[Any, Any, Any, Any]:
        """Scores the orientation sites that a move reads, one after another with the sites
        before at their new vectors, as `rescore_sites_nb`.

        `work` is the window of the chain around the move with the moved beads at their new
        places, from bead `w0`. `last` is the last bead of the chain. `sites` is the row of the
        move in `move_sites`. Returns the change of the orientation energy over the pairs, which
        the running score counts twice, the mask of the sites, their indices and their new
        vectors. No array is written.
        """
        has = sites >= 0
        k_all = jnp.maximum(sites, 0)
        ar = anchor_ar[k_all]
        vec = _stretch_vectors(work, ar - w0, is_L[ar], last - w0)
        change = jnp.float32(0.0)
        for m in range(sites.shape[0]):
            done = jnp.logical_and(has, np.arange(sites.shape[0]) < m)
            before = _local_orientation_after(
                anchor_orn, k_all[m], nbr_idx, nbr_w, nbr_valid, motif_weight, symmetric,
                anchor_orn[k_all[m]], k_all, vec, done,
            )  # fmt: skip
            after = _local_orientation_after(
                anchor_orn, k_all[m], nbr_idx, nbr_w, nbr_valid, motif_weight, symmetric,
                vec[m], k_all, vec, done,
            )  # fmt: skip
            change = change + jnp.where(has[m], after - before, 0.0)
        return change, has, k_all, vec

    def _write_sites(anchor_orn: Any, ok: Any, has: Any, k_all: Any, vec: Any) -> Any:
        """Returns the orientation vectors with those of an accepted move written, in one
        scatter. A site that is not written is sent out of range and dropped.
        """
        idx = jnp.where(jnp.logical_and(ok, has), k_all, anchor_orn.shape[0])
        return anchor_orn.at[idx].set(vec, mode="drop")

    def chain_batch(
        # state
        pos0: Any,
        ss0: Any,
        se0: Any,
        sh0: Any,
        so0: Any,
        sc0: Any,
        anchor_orn0: Any,
        T0_: Any,
        # static problem data
        dtn: Any,
        movable: Any,
        heat_dist: Any,
        comp_cls: Any,
        anchor_ar: Any,
        move_sites: Any,
        nbr_idx: Any,
        nbr_w: Any,
        nbr_valid: Any,
        is_L: Any,
        # schedule
        step_size: Any,
        dt: Any,
        js: Any,
        jc: Any,
        stretch_k: Any,
        squeeze_k: Any,
        ang_k: Any,
        dist_w: Any,
        ang_w: Any,
        r0: Any,
        excl_w: Any,
        heat_weight: Any,
        motif_weight: Any,
        symmetric: Any,
        conf_cx: Any,
        conf_cy: Any,
        conf_cz: Any,
        conf_R: Any,
        conf_w: Any,
        comp_r0: Any,
        comp_w: Any,
        comp_ea: Any,
        comp_eb: Any,
        # RNG
        key: Any,
        # the real bead count and the real movable count
        n_active: Any,
        n_movable_active: Any,
        cap_home: Any,
        cap_r: Any,
        centre_of: Any,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
        """Runs one batch of `n_steps_per_batch` MC steps of one chain. Returns (pos_f, ss_f,
        se_f, sh_f, so_f, sc_f, anchor_orn_f, T_f, n_ok).
        """

        # `movable` is padded to the bucket. The sampler draws from the first `n_movable_active`
        # entries.
        def evaluate(
            pos: Any,
            ss: Any,
            se: Any,
            sh: Any,
            so: Any,
            sc: Any,
            anchor_orn: Any,
            T: Any,
            p: Any,
            disp: Any,
            u: Any,
            grid: Any = None,
        ) -> tuple[Any, ...]:
            """Scores one proposal against the current state.

            Returns whether it is accepted, the new position of the bead, the five scores after
            it and for the orientation term the anchor slot it changes and the new vector.
            """
            score = ss + se + sh + so + sc
            old_p = pos[p]
            new_p = old_p + disp

            loc_s_prev = _local_smooth_at(
                pos, old_p, p, dtn, stretch_k, squeeze_k, ang_k, dist_w, ang_w, n_active
            )
            loc_s_curr = _local_smooth_at(
                pos, new_p, p, dtn, stretch_k, squeeze_k, ang_k, dist_w, ang_w, n_active
            )
            ss_new = ss + (loc_s_curr - loc_s_prev)

            if grid_g > 0:
                loc_e_prev = _local_excl_grid(pos, old_p, p, r0, excl_w, n_active, grid)
                loc_e_curr = _local_excl_grid(pos, new_p, p, r0, excl_w, n_active, grid)
            else:
                loc_e_prev = _local_excl_at(pos, old_p, p, r0, excl_w, n_active)
                loc_e_curr = _local_excl_at(pos, new_p, p, r0, excl_w, n_active)
            if use_aff:
                # Added to the excluded volume score, which is counted the same way.
                loc_e_prev = loc_e_prev + _local_affinity_at(
                    pos,
                    old_p,
                    p,
                    comp_cls,
                    comp_r0,
                    comp_w,
                    comp_ea,
                    comp_eb,
                    n_active,
                )
                loc_e_curr = loc_e_curr + _local_affinity_at(
                    pos,
                    new_p,
                    p,
                    comp_cls,
                    comp_r0,
                    comp_w,
                    comp_ea,
                    comp_eb,
                    n_active,
                )
            se_new = se + 2.0 * (loc_e_curr - loc_e_prev)
            delta = (loc_s_curr - loc_s_prev) + 2.0 * (loc_e_curr - loc_e_prev)

            if use_heat:
                loc_h_prev = _local_heat_at(pos, old_p, p, heat_dist, heat_weight)
                loc_h_curr = _local_heat_at(pos, new_p, p, heat_dist, heat_weight)
                sh_new = sh + 2.0 * (loc_h_curr - loc_h_prev)
                delta = delta + 2.0 * (loc_h_curr - loc_h_prev)
            else:
                sh_new = sh

            if use_orn:
                # Every site within the stretch of the bead reads it. Their vectors come from a
                # window of the chain with the bead at its new place.
                n = pos.shape[0]
                width = min(ORN_WINDOW, n)
                w0 = jnp.clip(p - ORN_WINDOW // 2, 0, n - width)
                work = pos[w0 + np.arange(width)].at[p - w0].set(new_p)
                d_o, site_has, site_k, site_vec = _sites_delta(
                    work, w0, n_active - 1, move_sites[p], anchor_orn, anchor_ar, is_L, nbr_idx,
                    nbr_w, nbr_valid, motif_weight, symmetric,
                )  # fmt: skip
                so_new = so + 2.0 * d_o
                delta = delta + 2.0 * d_o
            else:
                so_new = so
                site_has = np.zeros(1, dtype=np.bool_)
                site_k = np.zeros(1, dtype=np.int32)
                site_vec = np.zeros((1, 3), dtype=np.float32)

            # Confinement is per bead and single counted. At weight 0 XLA removes the branch.
            loc_c_prev = _local_confine_at(old_p, conf_cx, conf_cy, conf_cz, conf_R, conf_w)
            loc_c_curr = _local_confine_at(new_p, conf_cx, conf_cy, conf_cz, conf_R, conf_w)
            sc_new = sc + (loc_c_curr - loc_c_prev)
            delta = delta + (loc_c_curr - loc_c_prev)

            # A move is accepted on the summed local delta, not on the difference of two running
            # totals. The totals are float32 and on a large block one unit in the last place exceeds
            # the gain of a small improvement. The totals are still kept for the Metropolis ratio
            # and the plateau test.
            score_new = score + delta
            ok_unc = delta < 0.0  # strict less than
            can_jump = jnp.logical_and(T > 0, score > 0)
            exponent = -jc * (score_new / jnp.maximum(score, 1e-30)) / jnp.maximum(T, 1e-30)
            exponent = jnp.clip(exponent, -80.0, 80.0)
            p_acc = js * jnp.exp(exponent)
            ok = jnp.logical_or(ok_unc, jnp.logical_and(can_jump, u < p_acc))
            # The hard rules are decided on the same draw.
            if use_cap:
                hd = new_p - cap_home[p]
                cap_reject = jnp.logical_and(cap_r[p] > 0.0, jnp.sum(hd * hd) > cap_r[p] * cap_r[p])
                ok = jnp.logical_and(ok, jnp.logical_not(cap_reject))
            if use_wall:
                if grid_g > 0:
                    cnt0, dep0 = _wall_grid(pos, old_p, p, r0, n_active, grid)
                    cnt1, dep1 = _wall_grid(pos, new_p, p, r0, n_active, grid)
                else:
                    cnt0, dep0 = _wall_at(pos, old_p, p, r0, n_active)
                    cnt1, dep1 = _wall_at(pos, new_p, p, r0, n_active)
                worse = jnp.logical_or(
                    cnt1 > cnt0,
                    jnp.logical_and(jnp.logical_and(cnt1 == cnt0, cnt1 > 0), dep1 > dep0),
                )
                ok = jnp.logical_and(ok, jnp.logical_not(worse))

            return ok, new_p, ss_new, se_new, sh_new, so_new, sc_new, site_has, site_k, site_vec

        def apply(
            carry: Any, p: Any, ok: Any, res: tuple[Any, ...], T_next: Any, slot: Any = 0
        ) -> Any:
            (pos, ss, se, sh, so, sc, anchor_orn, _T, n_ok), grid = carry
            _ok, new_p, ss_new, se_new, sh_new, so_new, sc_new, site_has, site_k, site_vec = res
            pos_next = pos.at[p].set(jnp.where(ok, new_p, pos[p]))
            if grid_g > 0:
                grid = _grid_note(grid, p, ok, slot)
            ss_next = jnp.where(ok, ss_new, ss)
            se_next = jnp.where(ok, se_new, se)
            sh_next = jnp.where(ok, sh_new, sh)
            so_next = jnp.where(ok, so_new, so)
            sc_next = jnp.where(ok, sc_new, sc)
            if use_orn:
                anchor_orn_next = _write_sites(anchor_orn, ok, site_has, site_k, site_vec)
            else:
                anchor_orn_next = anchor_orn
            n_ok_next = n_ok + jnp.where(ok, 1, 0)
            return (
                pos_next,
                ss_next,
                se_next,
                sh_next,
                so_next,
                sc_next,
                anchor_orn_next,
                T_next,
                n_ok_next,
            ), grid

        def evaluate_centre(
            pos: Any,
            ss: Any,
            se: Any,
            sh: Any,
            so: Any,
            sc: Any,
            anchor_orn: Any,
            T: Any,
            p: Any,
            disp: Any,
            u: Any,
            grid: Any = None,
        ) -> tuple[Any, ...]:
            """Scores one proposal of the kernel with the three bead move.

            A bead of a node that keeps its centre of mass moves with the two other beads of the
            node. Any other bead moves alone. The change is that of three moves of one bead,
            one after another, and is computed without a copy of the chain: every sum over the
            other beads reads the positions as they are, the pairs inside the three beads are
            corrected one by one, the bonds and angles are scored on a window of beads and the
            orientation sites take their new vectors as arguments. A part with a factor of zero
            has a change of zero. Returns whether the move is accepted, the three beads and their
            new positions, the five scores after it and the orientation sites of the move with
            their new vectors.
            """
            score = ss + se + sh + so + sc
            n = pos.shape[0]
            c = centre_of[p]
            is_c = c >= 0
            cs = jnp.clip(c, 1, n - 2)
            tri = jnp.where(is_c, jnp.stack([cs - 1, cs, cs + 1]), jnp.stack([p, p, p]))
            # The factors of the draw for the three beads: (-0.5, 1, -0.5) for the node,
            # (1, 0, -1) for the bead before it, (-1, 0, 1) for the bead after it and (1, 0, 0)
            # for a bead that moves alone.
            node = p == cs
            left = p == cs - 1
            f = jnp.stack(
                [
                    jnp.where(is_c, jnp.where(node, -0.5, jnp.where(left, 1.0, -1.0)), 1.0),
                    jnp.where(jnp.logical_and(is_c, node), 1.0, 0.0),
                    jnp.where(is_c, jnp.where(node, -0.5, jnp.where(left, -1.0, 1.0)), 0.0),
                ]
            ).astype(pos.dtype)
            old = pos[tri]
            new = old + f[:, None] * disp[None, :]
            zero = jnp.float32(0.0)
            d_e, d_h, d_c, d_dep = zero, zero, zero, zero
            d_cnt = jnp.int32(0)

            # Each of the three beads against every other bead, the two others of the three at
            # the places they had.
            for part in range(3):
                q = tri[part]
                if grid_g > 0:
                    e_prev = _local_excl_grid(pos, old[part], q, r0, excl_w, n_active, grid)
                    e_curr = _local_excl_grid(pos, new[part], q, r0, excl_w, n_active, grid)
                else:
                    e_prev = _local_excl_at(pos, old[part], q, r0, excl_w, n_active)
                    e_curr = _local_excl_at(pos, new[part], q, r0, excl_w, n_active)
                if use_aff:
                    e_prev = e_prev + _local_affinity_at(
                        pos, old[part], q, comp_cls, comp_r0, comp_w, comp_ea, comp_eb, n_active
                    )
                    e_curr = e_curr + _local_affinity_at(
                        pos, new[part], q, comp_cls, comp_r0, comp_w, comp_ea, comp_eb, n_active
                    )
                d_e = d_e + (e_curr - e_prev)
                if use_heat:
                    d_h = d_h + (
                        _local_heat_at(pos, new[part], q, heat_dist, heat_weight)
                        - _local_heat_at(pos, old[part], q, heat_dist, heat_weight)
                    )
                d_c = d_c + (
                    _local_confine_at(new[part], conf_cx, conf_cy, conf_cz, conf_R, conf_w)
                    - _local_confine_at(old[part], conf_cx, conf_cy, conf_cz, conf_R, conf_w)
                )
                if use_wall:
                    if grid_g > 0:
                        cnt0, dep0 = _wall_grid(pos, old[part], q, r0, n_active, grid)
                        cnt1, dep1 = _wall_grid(pos, new[part], q, r0, n_active, grid)
                    else:
                        cnt0, dep0 = _wall_at(pos, old[part], q, r0, n_active)
                        cnt1, dep1 = _wall_at(pos, new[part], q, r0, n_active)
                    d_cnt = d_cnt + (cnt1 - cnt0)
                    d_dep = d_dep + (dep1 - dep0)

            # A pair inside the three beads was read twice above, each time with one bead at its
            # old place. The four terms below leave the pair at its new places against its old.
            for a, b in ((0, 1), (1, 2), (0, 2)):
                oa, ob, na, nb = old[a], old[b], new[a], new[b]
                if b - a > excl_skip:
                    fix_e = (
                        _pair_excl(na, nb, r0, excl_w)
                        - _pair_excl(na, ob, r0, excl_w)
                        - _pair_excl(oa, nb, r0, excl_w)
                        + _pair_excl(oa, ob, r0, excl_w)
                    )
                    d_e = d_e + jnp.where(is_c, fix_e, 0.0)
                    if use_wall:
                        c_nn, p_nn = _pair_wall(na, nb, r0)
                        c_no, p_no = _pair_wall(na, ob, r0)
                        c_on, p_on = _pair_wall(oa, nb, r0)
                        c_oo, p_oo = _pair_wall(oa, ob, r0)
                        d_cnt = d_cnt + jnp.where(is_c, c_nn - c_no - c_on + c_oo, 0)
                        d_dep = d_dep + jnp.where(is_c, p_nn - p_no - p_on + p_oo, 0.0)
                if use_aff:
                    ca, cb = comp_cls[tri[a]], comp_cls[tri[b]]

                    def aff(x: Any, y: Any, ca: Any = ca, cb: Any = cb) -> Any:
                        return _pair_affinity(
                            x, y, ca, cb, comp_r0, comp_w, comp_ea, comp_eb, n_active
                        )

                    fix_a = aff(na, nb) - aff(na, ob) - aff(oa, nb) + aff(oa, ob)
                    d_e = d_e + jnp.where(is_c, fix_a, 0.0)
                if use_heat:
                    h_ba = heat_dist[tri[b], tri[a]]
                    h_ab = heat_dist[tri[a], tri[b]]
                    fix_h = (
                        _pair_heat(na, nb, h_ba, heat_weight)
                        - _pair_heat(na, ob, h_ba, heat_weight)
                        - _pair_heat(oa, nb, h_ab, heat_weight)
                        + _pair_heat(oa, ob, h_ba, heat_weight)
                    )
                    d_h = d_h + jnp.where(is_c, fix_h, 0.0)

            # The bonds and the angles of the three beads reach two beads to either side. They are
            # scored one bead after another on a window of beads around the move, wide enough for
            # the orientation sites that read the move.
            width = min(ORN_WINDOW, n)
            w0 = jnp.clip(jnp.where(is_c, cs, p) - ORN_WINDOW // 2, 0, n - width)
            w_idx = w0 + np.arange(width)
            work = pos[w_idx]
            dtn_w = dtn[jnp.minimum(w_idx, dtn.shape[0] - 1)]
            n_w = jnp.clip(n_active - w0, 0, width)
            d_s = zero
            for part in range(3):
                lq = tri[part] - w0
                old_q = work[lq]
                new_q = old_q + f[part] * disp
                d_s = d_s + (
                    _local_smooth_at(
                        work, new_q, lq, dtn_w, stretch_k, squeeze_k, ang_k, dist_w, ang_w, n_w
                    )
                    - _local_smooth_at(
                        work, old_q, lq, dtn_w, stretch_k, squeeze_k, ang_k, dist_w, ang_w, n_w
                    )
                )
                work = work.at[lq].set(new_q)
            # A bead that moves alone is the first of its three equal parts.
            new_tri = jnp.where(is_c, new, jnp.broadcast_to(new[0], (3, 3)))

            ss_new = ss + d_s
            se_new = se + 2.0 * d_e
            sh_new = sh + 2.0 * d_h
            sc_new = sc + d_c
            delta = d_s + 2.0 * d_e + 2.0 * d_h + d_c

            if use_orn:
                d_o, site_has, site_k, site_vec = _sites_delta(
                    work, w0, n_active - 1, move_sites[p], anchor_orn, anchor_ar, is_L, nbr_idx,
                    nbr_w, nbr_valid, motif_weight, symmetric,
                )  # fmt: skip
                so_new = so + 2.0 * d_o
                delta = delta + 2.0 * d_o
            else:
                so_new = so
                site_has = np.zeros(1, dtype=np.bool_)
                site_k = np.zeros(1, dtype=np.int32)
                site_vec = np.zeros((1, 3), dtype=np.float32)

            score_new = score + delta
            ok_unc = delta < 0.0
            can_jump = jnp.logical_and(T > 0, score > 0)
            exponent = -jc * (score_new / jnp.maximum(score, 1e-30)) / jnp.maximum(T, 1e-30)
            exponent = jnp.clip(exponent, -80.0, 80.0)
            p_acc = js * jnp.exp(exponent)
            ok = jnp.logical_or(ok_unc, jnp.logical_and(can_jump, u < p_acc))
            if use_wall:
                # The whole move may not add a pair under the radius or deepen one.
                worse = jnp.logical_or(d_cnt > 0, jnp.logical_and(d_cnt == 0, d_dep > 0.0))
                ok = jnp.logical_and(ok, jnp.logical_not(worse))
            return (
                ok, tri, new_tri, ss_new, se_new, sh_new, so_new, sc_new, site_has, site_k,
                site_vec,
            )  # fmt: skip

        def apply_centre(
            carry: Any, p: Any, ok: Any, res: tuple[Any, ...], T_next: Any, slot: Any = 0
        ) -> Any:
            (pos, ss, se, sh, so, sc, anchor_orn, _T, n_ok), grid = carry
            (
                _ok, tri, new_tri, ss_new, se_new, sh_new, so_new, sc_new, site_has, site_k,
                site_vec,
            ) = res  # fmt: skip
            pos_next = pos
            for part in range(3):
                q = tri[part]
                pos_next = pos_next.at[q].set(jnp.where(ok, new_tri[part], pos_next[q]))
                if grid_g > 0:
                    grid = _grid_note(grid, q, ok, slot * 3 + part)
            anchor_orn_next = anchor_orn
            if use_orn:
                anchor_orn_next = _write_sites(anchor_orn, ok, site_has, site_k, site_vec)
            return (
                pos_next,
                jnp.where(ok, ss_new, ss),
                jnp.where(ok, se_new, se),
                jnp.where(ok, sh_new, sh),
                jnp.where(ok, so_new, so),
                jnp.where(ok, sc_new, sc),
                anchor_orn_next,
                T_next,
                n_ok + jnp.where(ok, 1, 0),
            ), grid

        if use_centre:
            evaluate = evaluate_centre  # noqa: F811
            apply = apply_centre  # noqa: F811

        init = ((pos0, ss0, se0, sh0, so0, sc0, anchor_orn0, T0_, jnp.int32(0)), None)
        if prefetch <= 1:
            k_p, k_d, k_a = jax.random.split(key, 3)
            idx_picks = jax.random.randint(k_p, (n_steps_per_batch,), 0, n_movable_active)
            ps = movable[idx_picks]
            disps = jax.random.uniform(
                k_d,
                (n_steps_per_batch, 3),
                minval=-step_size,
                maxval=step_size,
                dtype=pos0.dtype,
            )
            accs = jax.random.uniform(k_a, (n_steps_per_batch,), dtype=pos0.dtype)

            if grid_g == 0:

                def body(i: Any, carry: Any) -> Any:
                    (pos, ss, se, sh, so, sc, anchor_orn, T, _n_ok), grid = carry
                    p = ps[i]
                    res = evaluate(
                        pos, ss, se, sh, so, sc, anchor_orn, T, p, disps[i], accs[i], grid
                    )
                    return apply(carry, p, res[0], res, T * dt)

                return jax.lax.fori_loop(0, n_steps_per_batch, body, init)[0]

            # With the grid the steps run in periods and the table is rebuilt at the start of each.
            n_periods = -(-n_steps_per_batch // grid_period)

            def period(q: Any, state: Any) -> Any:
                pos_q = state[0]
                grid_q = _grid_build(pos_q, n_active, r0)

                def step(r: Any, carry: Any) -> Any:
                    (pos, ss, se, sh, so, sc, anchor_orn, T, _n_ok), grid = carry
                    i = q * grid_period + r
                    live = i < n_steps_per_batch
                    ii = jnp.minimum(i, n_steps_per_batch - 1)
                    p = ps[ii]
                    res = evaluate(
                        pos, ss, se, sh, so, sc, anchor_orn, T, p, disps[ii], accs[ii], grid
                    )
                    ok = jnp.logical_and(res[0], live)
                    return apply(carry, p, ok, res, jnp.where(live, T * dt, T), r)

                out, _g = jax.lax.fori_loop(0, grid_period, step, (state, grid_q))
                return out

            return jax.lax.fori_loop(0, n_periods, period, init[0])

        # `prefetch` proposals are scored against one state, each on its own rung of the temperature
        # ladder. The first accepted in draw order is applied. The proposals up to it are consumed
        # and the rest are drawn again. A round still delivers `n_steps_per_batch` serial steps.
        eval_many = jax.vmap(evaluate, in_axes=(None,) * 7 + (0, 0, 0, 0, None))
        ladder = dt ** jnp.arange(prefetch, dtype=pos0.dtype)

        def cond_many(state: Any) -> Any:
            return state[1] < n_steps_per_batch

        def batch(b: Any, carry: Any, slot: Any) -> tuple[Any, Any]:
            (pos, ss, se, sh, so, sc, anchor_orn, T, _n_ok), grid = carry
            kb_p, kb_d, kb_a = jax.random.split(jax.random.fold_in(key, b), 3)
            pk = movable[jax.random.randint(kb_p, (prefetch,), 0, n_movable_active)]
            dk = jax.random.uniform(
                kb_d, (prefetch, 3), minval=-step_size, maxval=step_size, dtype=pos0.dtype
            )
            uk = jax.random.uniform(kb_a, (prefetch,), dtype=pos0.dtype)
            res = eval_many(pos, ss, se, sh, so, sc, anchor_orn, T * ladder, pk, dk, uk, grid)
            ok_v = res[0]
            any_ok = jnp.any(ok_v)
            j = jnp.argmax(ok_v)
            used = jnp.where(any_ok, j + 1, prefetch)

            def at_j(a: Any) -> Any:
                return a[j]

            pick = jax.tree_util.tree_map(at_j, res)
            return apply(carry, pk[j], any_ok, pick, T * (dt**used), slot), used

        if grid_g == 0:

            def body_many(state: Any) -> Any:
                carry, consumed, b = state
                carry, used = batch(b, carry, 0)
                return carry, consumed + used, b + 1

            final, _consumed, _b = jax.lax.while_loop(
                cond_many, body_many, (init, jnp.int32(0), jnp.int32(0))
            )
            return final[0]

        # With the grid the batches run in periods of `grid_period`. A round may exceed its step
        # count by less than one period.
        def body_period(state: Any) -> Any:
            carry, consumed, b = state
            grid_q = _grid_build(carry[0][0], n_active, r0)

            def one(r: Any, st: Any) -> Any:
                c, used_sum = st
                c, used = batch(b + r, c, r)
                return c, used_sum + used

            (carry, used_sum) = jax.lax.fori_loop(
                0, grid_period, one, ((carry[0], grid_q), jnp.int32(0))
            )
            return (carry[0], None), consumed + used_sum, b + grid_period

        final, _consumed, _b = jax.lax.while_loop(
            cond_many, body_period, (init, jnp.int32(0), jnp.int32(0))
        )
        return final[0]

    # vmap over K chains. The problem data and the schedule are shared.
    in_axes = (
        0,
        0,
        0,
        0,
        0,
        0,  # pos, ss, se, sh, so, sc
        0,  # anchor_orn
        0,  # T0, per chain, a prefetching batch cools by what it consumed
        None,
        None,  # dtn, movable
        None,  # heat_dist
        None,  # comp_cls
        None,
        None,  # anchor_ar, move_sites
        None,
        None,
        None,  # nbr_idx, nbr_w, nbr_valid
        None,  # is_L
        None,
        None,
        None,
        None,  # step_size, dt, js, jc
        None,
        None,
        None,
        None,
        None,  # stretch..ang_w
        None,
        None,  # r0, excl_w
        None,  # heat_weight
        None,
        None,  # motif_weight, symmetric
        None,
        None,
        None,
        None,
        None,  # conf_cx, conf_cy, conf_cz, conf_R, conf_w
        None,
        None,
        None,
        None,  # comp_r0, comp_w, comp_ea, comp_eb
        0,  # key
        None,  # n_active (shared)
        None,  # n_movable_active (shared)
        None,  # cap_home (shared across restarts)
        None,  # cap_r (shared)
        None,  # centre_of (shared)
    )
    out_axes = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    batched = jax.vmap(chain_batch, in_axes=in_axes, out_axes=out_axes)

    @jax.jit
    def kernel(
        pos_k: Any,
        ss_k: Any,
        se_k: Any,
        sh_k: Any,
        so_k: Any,
        sc_k: Any,
        anchor_orn_k: Any,
        T: Any,
        dtn: Any,
        movable: Any,
        heat_dist: Any,
        comp_cls: Any,
        anchor_ar: Any,
        move_sites: Any,
        nbr_idx: Any,
        nbr_w: Any,
        nbr_valid: Any,
        is_L: Any,
        step_size: Any,
        dt: Any,
        js: Any,
        jc: Any,
        stretch_k: Any,
        squeeze_k: Any,
        ang_k: Any,
        dist_w: Any,
        ang_w: Any,
        r0: Any,
        excl_w: Any,
        heat_weight: Any,
        motif_weight: Any,
        symmetric: Any,
        conf_cx: Any,
        conf_cy: Any,
        conf_cz: Any,
        conf_R: Any,
        conf_w: Any,
        comp_r0: Any,
        comp_w: Any,
        comp_ea: Any,
        comp_eb: Any,
        keys: Any,
        n_active: Any,
        n_movable_active: Any,
        cap_home: Any,
        cap_r: Any,
        centre_of: Any,
    ) -> Any:
        return batched(
            pos_k,
            ss_k,
            se_k,
            sh_k,
            so_k,
            sc_k,
            anchor_orn_k,
            T,
            dtn,
            movable,
            heat_dist,
            comp_cls,
            anchor_ar,
            move_sites,
            nbr_idx,
            nbr_w,
            nbr_valid,
            is_L,
            step_size,
            dt,
            js,
            jc,
            stretch_k,
            squeeze_k,
            ang_k,
            dist_w,
            ang_w,
            r0,
            excl_w,
            heat_weight,
            motif_weight,
            symmetric,
            conf_cx,
            conf_cy,
            conf_cz,
            conf_R,
            conf_w,
            comp_r0,
            comp_w,
            comp_ea,
            comp_eb,
            keys,
            n_active,
            n_movable_active,
            cap_home,
            cap_r,
            centre_of,
        )

    # The convergence loop runs on the device in one `lax.while_loop`, one MC batch of all K chains
    # per iteration. `_MAX_ITERS` caps the loop.
    _MAX_ITERS: int = 10000

    @jax.jit
    def kernel_full(
        pos_k: Any,
        ss_k: Any,
        se_k: Any,
        sh_k: Any,
        so_k: Any,
        sc_k: Any,
        anchor_orn_k: Any,
        T_init: Any,
        dtn: Any,
        movable: Any,
        heat_dist: Any,
        comp_cls: Any,
        anchor_ar: Any,
        move_sites: Any,
        nbr_idx: Any,
        nbr_w: Any,
        nbr_valid: Any,
        is_L: Any,
        step_size: Any,
        dt: Any,
        js: Any,
        jc: Any,
        stretch_k: Any,
        squeeze_k: Any,
        ang_k: Any,
        dist_w: Any,
        ang_w: Any,
        r0: Any,
        excl_w: Any,
        heat_weight: Any,
        motif_weight: Any,
        symmetric: Any,
        conf_cx: Any,
        conf_cy: Any,
        conf_cz: Any,
        conf_R: Any,
        conf_w: Any,
        comp_r0: Any,
        comp_w: Any,
        comp_ea: Any,
        comp_eb: Any,
        base_key: Any,
        stop_improvement: Any,
        stop_successes: Any,
        score_eps: Any,
        n_active: Any,
        n_movable_active: Any,
        cap_home: Any,
        cap_r: Any,
        centre_of: Any,
    ) -> Any:
        K = pos_k.shape[0]

        def cond_fn(state: Any) -> Any:
            _, _, _, _, _, _, _, _, _, iter_i, _, converged = state
            return jnp.logical_and(jnp.logical_not(converged), iter_i < _MAX_ITERS)

        def body_fn(state: Any) -> Any:
            pos, ss, se, sh, so, sc, anchor_orn, T, ms_score, iter_i, _, _ = state
            iter_key = jax.random.fold_in(base_key, iter_i + 1)
            keys = jax.random.split(iter_key, K)
            pos, ss, se, sh, so, sc, anchor_orn, T, n_ok = batched(
                pos,
                ss,
                se,
                sh,
                so,
                sc,
                anchor_orn,
                T,
                dtn,
                movable,
                heat_dist,
                comp_cls,
                anchor_ar,
                move_sites,
                nbr_idx,
                nbr_w,
                nbr_valid,
                is_L,
                step_size,
                dt,
                js,
                jc,
                stretch_k,
                squeeze_k,
                ang_k,
                dist_w,
                ang_w,
                r0,
                excl_w,
                heat_weight,
                motif_weight,
                symmetric,
                conf_cx,
                conf_cy,
                conf_cz,
                conf_R,
                conf_w,
                comp_r0,
                comp_w,
                comp_ea,
                comp_eb,
                keys,
                n_active,
                n_movable_active,
                cap_home,
                cap_r,
                centre_of,
            )
            score_per_chain = ss + se + sh + so + sc
            best_idx = jnp.argmin(score_per_chain)
            score = score_per_chain[best_idx]
            n_ok_best = n_ok[best_idx]
            plateaued = jnp.logical_and(
                score > stop_improvement * ms_score, n_ok_best < stop_successes
            )
            eps_done = score < score_eps
            rose = jnp.logical_and(ms_score > 0, score > SMOOTH_STOP_RATIO * ms_score)
            converged = jnp.logical_or(jnp.logical_or(plateaued, eps_done), rose)
            return (pos, ss, se, sh, so, sc, anchor_orn, T, score, iter_i + 1, n_ok_best, converged)

        # A very large first `ms_score`. The first batch does not pass the improvement test.
        init_state = (
            pos_k,
            ss_k,
            se_k,
            sh_k,
            so_k,
            sc_k,
            anchor_orn_k,
            jnp.full((K,), T_init, dtype=jnp.float32),
            jnp.float32(1e30),  # ms_score
            jnp.int32(0),  # iter_i
            jnp.int32(0),  # n_ok_best (filler)
            jnp.bool_(False),  # converged
        )
        final = jax.lax.while_loop(cond_fn, body_fn, init_state)
        (
            pos_f,
            ss_f,
            se_f,
            sh_f,
            so_f,
            sc_f,
            anchor_orn_f,
            _T_f,
            final_score,
            iter_f,
            _n_ok_best_f,
            converged_f,
        ) = final
        return (pos_f, ss_f, se_f, sh_f, so_f, sc_f, anchor_orn_f, final_score, iter_f, converged_f)

    def _init_smooth_single(
        pos: Any,
        dtn: Any,
        stretch_k: Any,
        squeeze_k: Any,
        ang_k: Any,
        dist_w: Any,
        ang_w: Any,
        n_active: Any,
    ) -> Any:
        n = pos.shape[0]

        # Summed by `lax.scan` in order. A tree reduction groups a padded array differently from the
        # real one and the MC amplifies that difference. In a scan the masked pad terms add zero,
        # so a bucketed sum equals the real one bit for bit.
        def _bond_body(carry: Any, i: Any) -> tuple[Any, None]:
            val = _smooth_len(pos[i], pos[i + 1], dtn[i], stretch_k, squeeze_k, dist_w)
            return carry + jnp.where(i + 1 < n_active, val, 0.0), None

        def _angle_body(carry: Any, i: Any) -> tuple[Any, None]:
            val = _smooth_ang(pos[i], pos[i + 1], pos[i + 2], ang_k, ang_w)
            return carry + jnp.where(i + 2 < n_active, val, 0.0), None

        bonds_total, _ = jax.lax.scan(_bond_body, jnp.float32(0.0), jnp.arange(n - 1))
        angles_total, _ = jax.lax.scan(_angle_body, jnp.float32(0.0), jnp.arange(n - 2))
        return bonds_total + angles_total

    def _init_excl_single(pos: Any, r0: Any, weight: Any, n_active: Any) -> Any:
        n = pos.shape[0]
        idx = jnp.arange(n)

        def scan_body(carry: Any, i: Any) -> tuple[Any, None]:
            diff = pos - pos[i]
            d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
            rel = jnp.maximum(0.0, (r0 - d) / r0)
            contrib = weight * rel * rel
            # Pad columns are masked and a pad row is zero.
            in_range = jnp.logical_and(jnp.abs(idx - i) > excl_skip, idx < n_active)
            row = jnp.where(i < n_active, jnp.sum(jnp.where(in_range, contrib, 0.0)), 0.0)
            return carry + row, None

        total, _ = jax.lax.scan(scan_body, jnp.float32(0.0), idx)
        return total

    def _init_affinity_single(
        pos: Any,
        comp_cls: Any,
        comp_r0: Any,
        comp_w: Any,
        comp_ea: Any,
        comp_eb: Any,
        n_active: Any,
    ) -> Any:
        n = pos.shape[0]

        def scan_body(carry: Any, i: Any) -> tuple[Any, None]:
            row = _local_affinity_at(
                pos,
                pos[i],
                i,
                comp_cls,
                comp_r0,
                comp_w,
                comp_ea,
                comp_eb,
                n_active,
            )
            return carry + jnp.where(i < n_active, row, 0.0), None

        total, _ = jax.lax.scan(scan_body, jnp.float32(0.0), jnp.arange(n))
        return total

    def _init_heat_single(pos: Any, heat_dist: Any, heat_weight: Any) -> Any:
        n = pos.shape[0]
        idx = jnp.arange(n)

        def scan_body(carry: Any, i: Any) -> tuple[Any, None]:
            diff = pos - pos[i]
            d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
            exp_d = heat_dist[:, i]
            active = jnp.logical_and(idx != i, exp_d >= 1e-6)
            exp_d_safe = jnp.maximum(exp_d, 1e-6)
            rel = (d - exp_d_safe) / exp_d_safe
            contrib = rel * rel
            return carry + jnp.sum(jnp.where(active, contrib, 0.0)), None

        total, _ = jax.lax.scan(scan_body, jnp.float32(0.0), idx)
        return heat_weight * total

    def _init_confine_single(
        pos: Any, cx: Any, cy: Any, cz: Any, R: Any, weight: Any, n_active: Any
    ) -> Any:
        """Returns the sum of the confinement scores of the beads, by `lax.scan`, which keeps
        the float32 order under padding.
        """

        def _body(carry: Any, i: Any) -> tuple[Any, None]:
            c = _local_confine_at(pos[i], cx, cy, cz, R, weight)
            return carry + jnp.where(i < n_active, c, 0.0), None

        total, _ = jax.lax.scan(_body, jnp.float32(0.0), jnp.arange(pos.shape[0]))
        return total

    def _init_anchor_orientations_single(
        pos: Any,
        anchor_ar: Any,
        is_L: Any,
        n_active: Any,
    ) -> Any:
        """Returns the (n_anchors, 3) initial orientation vectors. `is_L` is indexed by bead."""
        return _stretch_vectors(pos, anchor_ar, is_L[anchor_ar], n_active - 1)

    def _init_orientation_score_single(
        anchor_orn: Any,
        nbr_idx: Any,
        nbr_w: Any,
        nbr_valid: Any,
        motif_weight: Any,
        symmetric: Any,
    ) -> Any:
        """Returns the orientation score over all anchors, summed by `lax.scan`,
        which keeps the float32 order under padding.
        """

        def per_anchor(k_idx: Any) -> Any:
            return _local_orientation_at(
                anchor_orn,
                k_idx,
                nbr_idx,
                nbr_w,
                nbr_valid,
                motif_weight,
                symmetric,
            )

        def _scan_body(carry: Any, k_idx: Any) -> tuple[Any, None]:
            return carry + per_anchor(k_idx), None

        total, _ = jax.lax.scan(_scan_body, jnp.float32(0.0), jnp.arange(anchor_orn.shape[0]))
        return total

    init_smooth = jax.jit(
        jax.vmap(
            _init_smooth_single,
            in_axes=(0, None, None, None, None, None, None, None),
        )
    )
    init_excl = jax.jit(jax.vmap(_init_excl_single, in_axes=(0, None, None, None)))
    init_affinity = jax.jit(
        jax.vmap(
            _init_affinity_single,
            in_axes=(0, None, None, None, None, None, None),
        )
    )
    init_heat = jax.jit(jax.vmap(_init_heat_single, in_axes=(0, None, None)))
    init_confine = jax.jit(
        jax.vmap(
            _init_confine_single,
            in_axes=(0, None, None, None, None, None, None),
        )
    )
    init_anchor_orn = jax.jit(
        jax.vmap(_init_anchor_orientations_single, in_axes=(0, None, None, None))
    )
    init_orn_score = jax.jit(
        jax.vmap(
            _init_orientation_score_single,
            in_axes=(0, None, None, None, None, None),
        )
    )
    # The same init functions vmapped over K chains that each carry their own problem data.
    # A launch scores its chains in one call.
    init_k = {
        "smooth": jax.jit(jax.vmap(_init_smooth_single, in_axes=(0, 0, 0, 0, 0, None, None, 0))),
        "excl": jax.jit(jax.vmap(_init_excl_single, in_axes=(0, 0, None, 0))),
        "affinity": jax.jit(
            jax.vmap(_init_affinity_single, in_axes=(0, 0, 0, None, None, None, 0))
        ),
        "heat": jax.jit(jax.vmap(_init_heat_single, in_axes=(0, 0, None))),
        "confine": jax.jit(jax.vmap(_init_confine_single, in_axes=(0, 0, 0, 0, 0, 0, 0))),
        "anchor_orn": jax.jit(jax.vmap(_init_anchor_orientations_single, in_axes=(0, 0, 0, 0))),
        "orn_score": jax.jit(
            jax.vmap(_init_orientation_score_single, in_axes=(0, 0, 0, 0, None, None))
        ),
    }

    # The batched variant runs K different IBs in one kernel. Every array per IB is on axis 0.
    # Convergence is per chain and the loop runs until every chain has converged or the cap is
    # reached. `chain_batch` and the init functions are the same as for one problem.
    in_axes_mp = (
        0,
        0,
        0,
        0,
        0,
        0,  # pos, ss, se, sh, so, sc  (per-chain)
        0,  # anchor_orn (per-chain)
        0,  # T0, per chain
        0,
        0,  # dtn, movable (per-IB)
        0,  # heat_dist (per-IB)
        0,  # comp_cls (per-IB)
        0,
        0,  # anchor_ar, move_sites (per-IB)
        0,
        0,
        0,  # nbr_idx, nbr_w, nbr_valid (per-IB)
        0,  # is_L (per-IB)
        0,  # step_size (per-IB)
        None,
        None,
        None,  # dt, js, jc (shared schedule)
        0,
        0,
        0,  # stretch_k, squeeze_k, ang_k (per-IB; small-IB boost varies)
        None,
        None,  # dist_w, ang_w (global weights)
        0,  # r0 (per-IB auto excl radius)
        None,  # excl_w (global)
        None,  # heat_weight (global)
        None,
        None,  # motif_weight, symmetric (global)
        0,
        0,
        0,
        0,
        0,  # conf_cx, conf_cy, conf_cz, conf_R, conf_w (per-IB)
        0,  # comp_r0 (per-IB auto radius)
        None,  # comp_w (global)
        None,
        None,  # comp_ea, comp_eb (global)
        0,  # keys (per-chain)
        0,  # n_active (per-IB)
        0,  # n_movable_active (per-IB)
        0,  # cap_home (per-IB)
        0,  # cap_r (per-IB)
        0,  # centre_of (per-IB)
    )
    batched_mp = jax.vmap(chain_batch, in_axes=in_axes_mp, out_axes=out_axes)

    @jax.jit
    def kernel_full_mp(
        pos_k: Any,
        ss_k: Any,
        se_k: Any,
        sh_k: Any,
        so_k: Any,
        sc_k: Any,
        anchor_orn_k: Any,
        T_init: Any,
        dtn: Any,
        movable: Any,
        heat_dist: Any,
        comp_cls: Any,
        anchor_ar: Any,
        move_sites: Any,
        nbr_idx: Any,
        nbr_w: Any,
        nbr_valid: Any,
        is_L: Any,
        step_size: Any,
        dt: Any,
        js: Any,
        jc: Any,
        stretch_k: Any,
        squeeze_k: Any,
        ang_k: Any,
        dist_w: Any,
        ang_w: Any,
        r0: Any,
        excl_w: Any,
        heat_weight: Any,
        motif_weight: Any,
        symmetric: Any,
        conf_cx: Any,
        conf_cy: Any,
        conf_cz: Any,
        conf_R: Any,
        conf_w: Any,
        comp_r0: Any,
        comp_w: Any,
        comp_ea: Any,
        comp_eb: Any,
        base_key: Any,
        chain_seed: Any,
        stop_improvement: Any,
        stop_successes: Any,
        score_eps: Any,
        n_active: Any,
        n_movable_active: Any,
        cap_home: Any,
        cap_r: Any,
        centre_of: Any,
    ) -> Any:
        K = pos_k.shape[0]

        def cond_fn(state: Any) -> Any:
            iter_i = state[9]
            converged = state[11]  # (K,)
            return jnp.logical_and(jnp.logical_not(jnp.all(converged)), iter_i < _MAX_ITERS)

        def hold(new: Any, old: Any, done: Any) -> Any:
            """Keeps the value of a converged chain. It stops where it would have stopped
            alone.
            """
            return jnp.where(done.reshape(done.shape + (1,) * (new.ndim - 1)), old, new)

        def body_fn(state: Any) -> Any:
            prev = state
            pos, ss, se, sh, so, sc, anchor_orn, T, ms_score, iter_i, _, conv_prev = state
            iter_key = jax.random.fold_in(base_key, iter_i + 1)
            # Each chain folds its own seed. Its stream then does not depend on the number of chains
            # in the launch or on its place among them.

            def chain_key(sd: Any) -> Any:
                return jax.random.fold_in(iter_key, sd)

            keys = jax.vmap(chain_key)(chain_seed)
            pos, ss, se, sh, so, sc, anchor_orn, T, n_ok = batched_mp(
                pos,
                ss,
                se,
                sh,
                so,
                sc,
                anchor_orn,
                T,
                dtn,
                movable,
                heat_dist,
                comp_cls,
                anchor_ar,
                move_sites,
                nbr_idx,
                nbr_w,
                nbr_valid,
                is_L,
                step_size,
                dt,
                js,
                jc,
                stretch_k,
                squeeze_k,
                ang_k,
                dist_w,
                ang_w,
                r0,
                excl_w,
                heat_weight,
                motif_weight,
                symmetric,
                conf_cx,
                conf_cy,
                conf_cz,
                conf_R,
                conf_w,
                comp_r0,
                comp_w,
                comp_ea,
                comp_eb,
                keys,
                n_active,
                n_movable_active,
                cap_home,
                cap_r,
                centre_of,
            )
            # A converged chain keeps its state while the rest of the launch runs.
            pos = hold(pos, prev[0], conv_prev)
            ss = hold(ss, prev[1], conv_prev)
            se = hold(se, prev[2], conv_prev)
            sh = hold(sh, prev[3], conv_prev)
            so = hold(so, prev[4], conv_prev)
            sc = hold(sc, prev[5], conv_prev)
            anchor_orn = hold(anchor_orn, prev[6], conv_prev)
            n_ok = hold(n_ok, prev[10], conv_prev)

            score = ss + se + sh + so + sc  # (K,)
            # A chain plateaus when its improvement is under the threshold and it accepted too few
            # moves in this batch.
            plateaued = jnp.logical_and(score > stop_improvement * ms_score, n_ok < stop_successes)
            eps_done = score < score_eps
            rose = jnp.logical_and(ms_score > 0, score > SMOOTH_STOP_RATIO * ms_score)
            # Convergence is latched.
            converged = jnp.logical_or(
                jnp.logical_or(jnp.logical_or(plateaued, eps_done), rose), conv_prev
            )
            return (pos, ss, se, sh, so, sc, anchor_orn, T, score, iter_i + 1, n_ok, converged)

        init_state = (
            pos_k,
            ss_k,
            se_k,
            sh_k,
            so_k,
            sc_k,
            anchor_orn_k,
            jnp.full((K,), T_init, dtype=jnp.float32),
            jnp.full((K,), 1e30, dtype=jnp.float32),  # ms_score per-chain
            jnp.int32(0),  # iter_i
            jnp.zeros((K,), dtype=jnp.int32),  # n_ok (filler)
            jnp.zeros((K,), dtype=jnp.bool_),  # converged per-chain
        )
        final = jax.lax.while_loop(cond_fn, body_fn, init_state)
        (
            pos_f,
            ss_f,
            se_f,
            sh_f,
            so_f,
            sc_f,
            anchor_orn_f,
            _T_f,
            final_score,
            iter_f,
            _n_ok_f,
            converged_f,
        ) = final
        return (pos_f, ss_f, se_f, sh_f, so_f, sc_f, anchor_orn_f, final_score, iter_f, converged_f)

    bundle = (
        kernel,  # one batch, for diagnostics
        kernel_full,  # full convergence on the device
        init_smooth,
        init_excl,
        init_heat,
        init_confine,
        init_anchor_orn,
        init_orn_score,
        kernel_full_mp,  # batched, K different IBs
        init_affinity,
        init_k,
    )
    _kernel_cache[cache_key] = bundle
    return bundle


def mc_smooth_jax(
    pos: np.ndarray[Any, Any],
    dtn: np.ndarray[Any, Any],
    fixed: np.ndarray[Any, Any],
    step_size: float,
    settings: "Settings",
    char_orientations: np.ndarray[Any, Any] | None = None,
    anchor_neighbors: dict[int, list[int]] | None = None,
    anchor_neighbor_weights: dict[int, list[float]] | None = None,
    heat_dist: np.ndarray[Any, Any] | None = None,
    compartment: np.ndarray[Any, Any] | None = None,
    pos_batch: np.ndarray[Any, Any] | None = None,
    return_all: bool = False,
    centre: np.ndarray[Any, Any] | None = None,
) -> Any:
    """Anneals one chain on JAX and returns the final score of the best chain. Writes its final
    positions into `pos`.

    With `pos_batch` of shape (B, N, 3) the kernel runs B anneals from those starts in one launch
    and shares `dtn`, `fixed`, the heat target and the schedule. The loop stops when the best
    chain converges. With `return_all` it returns `(scores, finals)` of shapes (B,) and (B, N, 3)
    and does not change `pos`. A fixed bead of `centre` moves with the bead before it and the
    bead after it and the three keep their centre of mass, as in `mc_smooth_numba`.
    """
    if not jax_is_available():
        raise RuntimeError(
            "settings.mc_backend='jax' but JAX is not installed.  "
            "Install with `pip install gnome3d[jax]` or set mc_backend='numba'."
        )
    import jax
    import jax.numpy as jnp

    n: int = pos.shape[0]
    if n <= 2:
        return 0.0

    use_wall: bool = bool(getattr(settings, "smooth_hard_wall", False))
    cap_frac: float = float(getattr(settings, "smooth_anchor_cap", 0.0))
    use_cap: bool = cap_frac > 0.0
    prefetch: int = max(1, int(getattr(settings, "smooth_prefetch", 1)))
    use_centre: bool = centre is not None and not use_cap
    movable_np: I64Array = _movable_beads(fixed, centre if use_centre else None, use_cap)
    if len(movable_np) == 0:
        return 0.0

    if pos_batch is not None:
        if pos_batch.ndim != 3 or pos_batch.shape[1:] != (n, 3):
            raise ValueError(f"pos_batch must have shape (B, {n}, 3); got {pos_batch.shape}")
        K = int(pos_batch.shape[0])
    else:
        K = max(1, int(settings.mc_smooth_chains))
    n_steps_per_batch: int = int(settings.mc_stop_steps_smooth)

    use_excl: bool = bool(settings.use_excluded_volume) and bool(settings.exclusion_apply_to_smooth)
    excl_skip: int = int(settings.exclusion_skip_neighbors)
    excl_w_v: float = float(settings.exclusion_weight) if use_excl else 0.0
    if use_excl:
        excl_r0: float = float(settings.exclusion_radius_smooth)
        if excl_r0 <= 0.0:
            factor = float(settings.exclusion_auto_factor_smooth)
            excl_r0 = factor * float(np.asarray(dtn).mean())
    else:
        excl_r0 = 1.0  # not read, must be valid

    # One resolver for both backends.
    from gnome3d.mc.numba.common import affinity_params  # noqa: PLC0415

    _aff = affinity_params(
        settings,
        "smooth",
        float(np.asarray(dtn).mean()) if np.asarray(dtn).size else 1.0,
        compartment,
    )
    use_aff: bool = _aff.any_on

    use_heat: bool = heat_dist is not None
    heat_weight_v: float = float(settings.subanchor_heatmap_dist_weight) if use_heat else 0.0
    use_orn: bool = (
        char_orientations is not None
        and anchor_neighbors is not None
        and anchor_neighbor_weights is not None
    )
    motif_weight_v: float = orientation_weight(settings) if use_orn else 0.0
    motifs_symmetric_v: bool = bool(getattr(settings, "motifs_symmetric", True))

    use_conf: bool = bool(settings.use_confinement) and bool(settings.confinement_apply_to_smooth)
    if use_conf:
        conf_cx_v: float = float(pos[:, 0].mean())
        conf_cy_v: float = float(pos[:, 1].mean())
        conf_cz_v: float = float(pos[:, 2].mean())
        conf_R_v: float = float(settings.confinement_radius_smooth)
        if conf_R_v <= 0.0:
            avg_bond = float(np.asarray(dtn).mean()) if dtn.size > 0 else 1.0
            pf = float(settings.confinement_packing_factor_smooth)
            conf_R_v = pf * avg_bond * (n ** (1.0 / 3.0))
        conf_w_v: float = float(settings.confinement_weight)
    else:
        conf_cx_v = conf_cy_v = conf_cz_v = 0.0
        conf_R_v = 1.0
        conf_w_v = 0.0

    if use_orn:
        assert char_orientations is not None and anchor_neighbors is not None
        assert anchor_neighbor_weights is not None
        anchor_ar_np: I32Array = np.array([int(i) for i in np.where(fixed)[0]], dtype=np.int32)
        n_anchors = int(len(anchor_ar_np))
        nbr_idx_np, nbr_w_np, nbr_valid_np, move_sites_np = _orientation_tables(
            n, anchor_ar_np, anchor_neighbors, anchor_neighbor_weights,
            centre if use_centre else None, 0,
        )  # fmt: skip
        max_nbrs = int(nbr_idx_np.shape[1])
        is_L_np = np.array([c == "L" for c in char_orientations], dtype=np.bool_)
        # The anchor indexed arrays are padded to the (A, M) buckets. Pad anchors and edges have
        # `nbr_valid` False and add exactly 0.
        if bool(settings.mc_executor_jax_bucket_shapes):
            A = jax_bucket_for(n_anchors, ANCHOR_BUCKETS)
            M = jax_bucket_for(max_nbrs, NBR_BUCKETS)
            ap, mp = A - n_anchors, M - max_nbrs
            if ap > 0 or mp > 0:
                anchor_ar_np = np.concatenate([anchor_ar_np, np.zeros(ap, dtype=np.int32)])
                nbr_idx_np = np.pad(nbr_idx_np, ((0, ap), (0, mp)))
                nbr_w_np = np.pad(nbr_w_np, ((0, ap), (0, mp)))
                nbr_valid_np = np.pad(nbr_valid_np, ((0, ap), (0, mp)))  # pads are False
                n_anchors, max_nbrs = A, M
    else:
        n_anchors = 1  # placeholder shape
        max_nbrs = 1
        anchor_ar_np = np.zeros(1, dtype=np.int32)
        nbr_idx_np = np.zeros((1, 1), dtype=np.int32)
        nbr_w_np = np.zeros((1, 1), dtype=np.float32)
        nbr_valid_np = np.zeros((1, 1), dtype=np.bool_)
        move_sites_np = np.full((n, 1), -1, dtype=np.int32)
        is_L_np = np.zeros(n, dtype=np.bool_)

    pos_f32: F32Array = pos.astype(np.float32)
    if pos_batch is not None:
        # K distinct starts.
        pos_k_np = np.ascontiguousarray(pos_batch.astype(np.float32))
    else:
        pos_k_np = np.broadcast_to(pos_f32, (K, n, 3)).copy()
    dtn_np: F32Array = dtn.astype(np.float32)
    heat_np: F32Array
    if use_heat:
        assert heat_dist is not None
        heat_np = heat_dist.astype(np.float32)
    else:
        heat_np = np.zeros((1, 1), dtype=np.float32)  # placeholder

    # Shape bucketing pads N to a bucket. That XLA reuses one compiled kernel. Pad beads do
    # nothing: `n_active` masks the chain, excluded volume and confinement, their heat rows are zero
    # and the sampler draws from the first `n_movable_active` movable beads. Every bead indexed
    # input is padded to B.
    n_active_v: int = n
    n_movable_v: int = int(movable_np.shape[0])
    if bool(settings.mc_executor_jax_bucket_shapes):
        B: int = jax_bucket_for(n)
    else:
        B = n
    if B > n:
        n_pad = B - n
        pos_k_np = np.concatenate(
            [pos_k_np, np.zeros((pos_k_np.shape[0], n_pad, 3), dtype=np.float32)], axis=1
        )
        dtn_np = np.concatenate([dtn_np, np.ones(n_pad, dtype=np.float32)], axis=0)
        if use_heat:
            heat_pad = np.zeros((B, B), dtype=np.float32)
            heat_pad[:n, :n] = heat_np
            heat_np = heat_pad
        move_sites_np = np.pad(move_sites_np, ((0, n_pad), (0, 0)), constant_values=-1)
        if is_L_np.shape[0] == n:  # indexed by bead when use_orn is off
            is_L_np = np.concatenate([is_L_np, np.zeros(n_pad, dtype=np.bool_)])
        movable_np = np.concatenate(
            [movable_np, np.zeros(B - movable_np.shape[0], dtype=movable_np.dtype)]
        )

    grid_g, grid_cap = 0, 0
    if (
        bool(getattr(settings, "smooth_jax_grid", False))
        and use_excl
        and B >= int(getattr(settings, "smooth_jax_grid_min_beads", 4096))
    ):
        grid_g, grid_cap = _grid_static([(pos_f32, excl_r0)])
    bundle = _build_smooth_kernel(
        n_steps_per_batch, excl_skip, use_heat, use_orn, max_nbrs, use_aff, use_wall, use_cap,
        prefetch, grid_g, grid_cap, use_centre,
    )  # fmt: skip
    (
        _kernel_one_batch,
        kernel_full,
        init_smooth,
        init_excl,
        init_heat,
        init_confine,
        init_anchor_orn,
        init_orn_score,
        _kernel_full_mp,  # used by the batched entry
        init_affinity,
        _init_k,
    ) = bundle

    pos_k = jnp.asarray(pos_k_np)
    dtn_j = jnp.asarray(dtn_np)
    movable_j = jnp.asarray(movable_np)
    # The home of the cap is the current place of the anchors.
    cap_r_np = np.zeros(pos_k_np.shape[1], dtype=np.float32)
    if use_cap:
        cap_r_np[:n] = np.where(fixed, cap_frac * float(dtn_np[:n].mean()), 0.0)
    cap_home_j = jnp.asarray(pos_k_np[0])
    cap_r_j = jnp.asarray(cap_r_np)
    centre_of_np = _centre_table(centre if use_centre else None, int(pos_k_np.shape[1]))
    heat_j = jnp.asarray(heat_np)
    anchor_ar_j = jnp.asarray(anchor_ar_np)
    move_sites_j = jnp.asarray(move_sites_np)
    nbr_idx_j = jnp.asarray(nbr_idx_np)
    nbr_w_j = jnp.asarray(nbr_w_np)
    nbr_valid_j = jnp.asarray(nbr_valid_np)
    is_L_j = jnp.asarray(is_L_np)
    n_active_j = jnp.int32(n_active_v)
    n_movable_active_j = jnp.int32(n_movable_v)

    # Padded to the bucket width. A pad entry is class 0 and adds nothing.
    B_pad = int(pos_k.shape[1])
    if use_aff:
        _c = np.zeros(B_pad, dtype=np.int8)
        if compartment is not None:
            _cc = np.asarray(compartment, dtype=np.int8)
            _c[: _cc.shape[0]] = _cc
        comp_cls_j = jnp.asarray(_c)
    else:
        comp_cls_j = jnp.zeros((B_pad,), dtype=jnp.int8)
    comp_r0_j = jnp.float32(_aff.comp_r0)
    comp_w_j = jnp.float32(_aff.comp_weight if _aff.use_comp else 0.0)
    comp_ea_j = jnp.float32(_aff.comp_ea)
    comp_eb_j = jnp.float32(_aff.comp_eb)
    seed_offset: int = stable_seed_offset(log.current())

    ss_k = init_smooth(
        pos_k,
        dtn_j,
        jnp.float32(settings.spring_stretch),
        jnp.float32(settings.spring_squeeze),
        jnp.float32(settings.spring_angular),
        jnp.float32(settings.smooth_dist_weight),
        jnp.float32(settings.smooth_angle_weight),
        n_active_j,
    )
    se_k = (
        init_excl(pos_k, jnp.float32(excl_r0), jnp.float32(excl_w_v), n_active_j)
        if use_excl
        else jnp.zeros((K,), dtype=jnp.float32)
    )
    if use_aff:
        # Added to the excluded volume score, which is counted the same way.
        se_k = se_k + init_affinity(
            pos_k,
            comp_cls_j,
            comp_r0_j,
            comp_w_j,
            comp_ea_j,
            comp_eb_j,
            n_active_j,
        )
    sh_k = (
        init_heat(pos_k, heat_j, jnp.float32(heat_weight_v))
        if use_heat
        else jnp.zeros((K,), dtype=jnp.float32)
    )
    sc_k = (
        init_confine(
            pos_k,
            jnp.float32(conf_cx_v),
            jnp.float32(conf_cy_v),
            jnp.float32(conf_cz_v),
            jnp.float32(conf_R_v),
            jnp.float32(conf_w_v),
            n_active_j,
        )
        if use_conf
        else jnp.zeros((K,), dtype=jnp.float32)
    )
    if use_orn:
        anchor_orn_k = init_anchor_orn(pos_k, anchor_ar_j, is_L_j, n_active_j)
        so_k = init_orn_score(
            anchor_orn_k,
            nbr_idx_j,
            nbr_w_j,
            nbr_valid_j,
            jnp.float32(motif_weight_v),
            jnp.bool_(motifs_symmetric_v),
        )
    else:
        anchor_orn_k = jnp.zeros((K, n_anchors, 3), dtype=jnp.float32)
        so_k = jnp.zeros((K,), dtype=jnp.float32)

    T = jnp.float32(settings.max_temp_smooth)
    dt = jnp.float32(settings.dt_temp_smooth)
    js = jnp.float32(settings.jump_scale_smooth)
    jc = jnp.float32(settings.jump_coef_smooth)
    stretch_k_j = jnp.float32(settings.spring_stretch)
    squeeze_k_j = jnp.float32(settings.spring_squeeze)
    ang_k_j = jnp.float32(settings.spring_angular)
    dist_w_j = jnp.float32(settings.smooth_dist_weight)
    ang_w_j = jnp.float32(settings.smooth_angle_weight)
    r0_j = jnp.float32(excl_r0)
    excl_w_j = jnp.float32(excl_w_v)
    heat_w_j = jnp.float32(heat_weight_v)
    motif_w_j = jnp.float32(motif_weight_v)
    symmetric_j = jnp.bool_(motifs_symmetric_v)
    conf_cx_j = jnp.float32(conf_cx_v)
    conf_cy_j = jnp.float32(conf_cy_v)
    conf_cz_j = jnp.float32(conf_cz_v)
    conf_R_j = jnp.float32(conf_R_v)
    conf_w_j = jnp.float32(conf_w_v)
    step_size_j = jnp.float32(step_size)

    stop_improvement = jnp.float32(settings.mc_stop_improvement_smooth)
    stop_successes = jnp.int32(settings.mc_stop_successes_smooth)
    score_eps = jnp.float32(1e-6)
    base_key = jax.random.PRNGKey(seed_offset)

    # One JAX call runs the whole convergence loop on the device.
    (
        pos_k,
        ss_k,
        se_k,
        sh_k,
        so_k,
        sc_k,
        _anchor_orn_k_final,
        final_score_best,
        iter_count,
        converged_flag,
    ) = kernel_full(
        pos_k,
        ss_k,
        se_k,
        sh_k,
        so_k,
        sc_k,
        anchor_orn_k,
        T,
        dtn_j,
        movable_j,
        heat_j,
        comp_cls_j,
        anchor_ar_j,
        move_sites_j,
        nbr_idx_j,
        nbr_w_j,
        nbr_valid_j,
        is_L_j,
        step_size_j,
        dt,
        js,
        jc,
        stretch_k_j,
        squeeze_k_j,
        ang_k_j,
        dist_w_j,
        ang_w_j,
        r0_j,
        excl_w_j,
        heat_w_j,
        motif_w_j,
        symmetric_j,
        conf_cx_j,
        conf_cy_j,
        conf_cz_j,
        conf_R_j,
        conf_w_j,
        comp_r0_j,
        comp_w_j,
        comp_ea_j,
        comp_eb_j,
        base_key,
        stop_improvement,
        stop_successes,
        score_eps,
        n_active_j,
        n_movable_active_j,
        cap_home_j,
        cap_r_j,
        centre_of_np,
    )

    score_per_chain = np.asarray(ss_k + se_k + sh_k + so_k + sc_k)
    iter_n = int(iter_count)
    converged_v = bool(converged_flag)

    if LOG.isEnabledFor(logging.DEBUG):
        tail = "[done]" if converged_v else "[max-iters reached]"
        LOG.debug(
            "step %7s  score=%.4f  batches=%d  %s",
            f"{iter_n * n_steps_per_batch:,}",
            float(final_score_best),
            iter_n,
            tail,
        )

    if return_all:
        # Batched mode returns every chain without the bucket padding and leaves `pos` unchanged.
        return (
            score_per_chain.astype(np.float64),
            np.asarray(pos_k[:, :n]).astype(np.float32),
        )

    best_k: int = int(np.argmin(score_per_chain))
    # The bucket padding is cut off.
    pos[:] = np.asarray(pos_k[best_k][:n]).astype(pos.dtype)
    return float(score_per_chain[best_k])


def _pad_track(arr: np.ndarray[Any, Any] | None, B: int, dtype: Any) -> np.ndarray[Any, Any]:
    """Returns a per bead track padded to the bucket width, or zeros without one."""
    out = np.zeros(B, dtype=dtype)
    if arr is not None:
        a = np.asarray(arr, dtype=dtype)
        out[: min(a.shape[0], B)] = a[:B]
    return out


def _prep_smooth_problem_np(
    pos: np.ndarray[Any, Any],
    dtn: np.ndarray[Any, Any],
    fixed: np.ndarray[Any, Any],
    settings: "Settings",
    char_orientations: np.ndarray[Any, Any] | None,
    anchor_neighbors: dict[int, list[int]] | None,
    anchor_neighbor_weights: dict[int, list[float]] | None,
    heat_dist: np.ndarray[Any, Any] | None,
    B: int,
    A: int,
    M: int,
    compartment: np.ndarray[Any, Any] | None = None,
    heat_out: np.ndarray[Any, Any] | None = None,
    centre: np.ndarray[Any, Any] | None = None,
) -> dict[str, Any]:
    """Builds the kernel inputs of one IB as numpy arrays padded to a common bucket of B beads,
    A anchors and M neighbours.

    `heat_out` is an optional (B, B) buffer of the caller that receives the padded heat. The
    caller must zero it. Only the [:n, :n] block is written. Writing into a slice of the
    (K, B, B) array of the batch saves a second copy of the only quadratic input.
    """
    n = int(pos.shape[0])
    use_excl = bool(settings.use_excluded_volume) and bool(settings.exclusion_apply_to_smooth)
    use_heat = heat_dist is not None
    use_orn = (
        char_orientations is not None
        and anchor_neighbors is not None
        and anchor_neighbor_weights is not None
    )
    use_conf = bool(settings.use_confinement) and bool(settings.confinement_apply_to_smooth)

    excl_r0 = 1.0
    if use_excl:
        excl_r0 = float(settings.exclusion_radius_smooth)
        if excl_r0 <= 0.0:
            excl_r0 = float(settings.exclusion_auto_factor_smooth) * float(np.asarray(dtn).mean())

    if use_conf:
        conf_cx = float(pos[:, 0].mean())
        conf_cy = float(pos[:, 1].mean())
        conf_cz = float(pos[:, 2].mean())
        conf_R = float(settings.confinement_radius_smooth)
        if conf_R <= 0.0:
            avg_bond = float(np.asarray(dtn).mean()) if dtn.size > 0 else 1.0
            pf = float(settings.confinement_packing_factor_smooth)
            conf_R = pf * avg_bond * (n ** (1.0 / 3.0))
        conf_w = float(settings.confinement_weight)
    else:
        conf_cx = conf_cy = conf_cz = 0.0
        conf_R = 1.0
        conf_w = 0.0

    cap_frac = float(getattr(settings, "smooth_anchor_cap", 0.0))
    if cap_frac > 0.0:
        centre = None
    if use_orn:
        assert char_orientations is not None and anchor_neighbors is not None
        assert anchor_neighbor_weights is not None
        anchor_ar = np.array([int(i) for i in np.where(fixed)[0]], dtype=np.int32)
        n_anchors = int(len(anchor_ar))
        nbr_idx, nbr_w, nbr_valid, move_sites = _orientation_tables(
            n, anchor_ar, anchor_neighbors, anchor_neighbor_weights, centre, M
        )
        is_L = np.array([c == "L" for c in char_orientations], dtype=np.bool_)
        # Pad anchors have `nbr_valid` False and score 0.
        if A > n_anchors:
            ap = A - n_anchors
            anchor_ar = np.concatenate([anchor_ar, np.zeros(ap, dtype=np.int32)])
            nbr_idx = np.pad(nbr_idx, ((0, ap), (0, 0)))
            nbr_w = np.pad(nbr_w, ((0, ap), (0, 0)))
            nbr_valid = np.pad(nbr_valid, ((0, ap), (0, 0)))
    else:
        anchor_ar = np.zeros(A, dtype=np.int32)
        nbr_idx = np.zeros((A, M), dtype=np.int32)
        nbr_w = np.zeros((A, M), dtype=np.float32)
        nbr_valid = np.zeros((A, M), dtype=np.bool_)
        move_sites = np.full((n, MOVE_SITES), -1, dtype=np.int32)
        is_L = np.zeros(n, dtype=np.bool_)

    movable = _movable_beads(fixed, centre, cap_frac > 0.0)
    centre_of = _centre_table(centre, B)
    n_movable = int(movable.shape[0])
    pos_pad = pos.astype(np.float32)
    dtn_pad = dtn.astype(np.float32)
    if not use_heat:
        heat_pad = np.zeros((1, 1), dtype=np.float32)
    elif heat_out is not None:
        heat_out[:n, :n] = heat_dist  # the caller zeroed the pad region
        heat_pad = heat_out
    else:
        heat_pad = heat_dist.astype(np.float32)
    if B > n:
        n_pad = B - n
        pos_pad = np.concatenate([pos_pad, np.zeros((n_pad, 3), dtype=np.float32)], axis=0)
        dtn_pad = np.concatenate([dtn_pad, np.ones(n_pad, dtype=np.float32)], axis=0)
        if use_heat and heat_out is None:
            hp = np.zeros((B, B), dtype=np.float32)
            hp[:n, :n] = heat_pad
            heat_pad = hp
        move_sites = np.pad(move_sites, ((0, n_pad), (0, 0)), constant_values=-1)
        is_L = np.concatenate([is_L, np.zeros(n_pad, dtype=np.bool_)])

    # `movable` has n_movable entries and is padded to B on its own. An IB with n == B would
    # otherwise keep a short array and the (K, B) stack would be ragged. `n_movable` bounds the
    # kernel loop. The pads are not read.
    if B > n_movable:
        movable = np.concatenate([movable, np.zeros(B - n_movable, dtype=movable.dtype)])

    return {
        "n": n,
        "pos": pos_pad,  # (B, 3)
        "dtn": dtn_pad,  # (B,)
        "movable": movable,  # (B,)
        "heat": heat_pad,  # (B, B) or (1, 1)
        "anchor_ar": anchor_ar,  # (A,)
        "move_sites": move_sites,  # (B, MOVE_SITES)
        "nbr_idx": nbr_idx,  # (A, M)
        "nbr_w": nbr_w,  # (A, M)
        "nbr_valid": nbr_valid,  # (A, M)
        "is_L": is_L,  # (B,)
        "n_active": n,
        "n_movable": n_movable,
        "excl_r0": excl_r0,
        "cap_home": pos_pad.copy(),  # (B, 3)
        "cap_r": _cap_radii(fixed, dtn, cap_frac, B),  # (B,)
        "centre_of": centre_of,  # (B,)
        # A pad entry is class 0 and adds nothing.
        "comp_cls": _pad_track(compartment, B, np.int8),
        "conf_cx": conf_cx,
        "conf_cy": conf_cy,
        "conf_cz": conf_cz,
        "conf_R": conf_R,
        "conf_w": conf_w,
    }


def _centre_table(centre: np.ndarray[Any, Any] | None, B: int) -> I32Array:
    """Returns for each bead of a chain padded to B the node of its three bead move, -1 for
    none.
    """
    owner: I32Array = np.full(B, -1, dtype=np.int32)
    if centre is not None:
        for c in np.flatnonzero(centre):
            owner[c - 1 : c + 2] = c
    return owner


def _orientation_tables(
    n: int,
    anchor_ar: I32Array,
    anchor_neighbors: dict[int, list[int]],
    anchor_neighbor_weights: dict[int, list[float]],
    centre: np.ndarray[Any, Any] | None,
    width: int,
) -> tuple[I32Array, F32Array, np.ndarray[Any, Any], I32Array]:
    """Builds the orientation tables of one chain of `n` beads with its sites at `anchor_ar`.

    Returns the loop partners of each site, their weights and their mask, padded to `width`
    partners or to the most that a site has at 0, and the sites that the move of each bead
    reads, `MOVE_SITES` to a row padded with -1. The weight of a loop is its share of the
    larger of the sums at its two sites, as `loop_shares`. `centre` marks the nodes that move
    with their two neighbours.
    """
    count = int(anchor_ar.shape[0])
    lists = [list(anchor_neighbors.get(k, [])) for k in range(count)]
    weights = [list(anchor_neighbor_weights.get(k, [])) for k in range(count)]
    most = max(max((len(v) for v in lists), default=1), 1)
    m_width = width if width > 0 else most
    nbr_idx: I32Array = np.zeros((count, m_width), dtype=np.int32)
    nbr_w: F32Array = np.zeros((count, m_width), dtype=np.float32)
    nbr_valid = np.zeros((count, m_width), dtype=np.bool_)
    total = np.array([float(sum(v)) for v in weights], dtype=np.float64)
    for k in range(count):
        for m, (j, w) in enumerate(zip(lists[k], weights[k], strict=True)):
            largest = max(float(total[k]), float(total[int(j)]))
            nbr_idx[k, m] = int(j)
            nbr_w[k, m] = float(w) / largest if largest > 0.0 else 0.0
            nbr_valid[k, m] = True
    offsets: I32Array = np.zeros(count + 1, dtype=np.int32)
    offsets[1:] = np.cumsum([len(v) for v in lists])
    rows = sites_of_moves(
        n, anchor_ar, offsets, None if centre is None else _centre_table(centre, n)
    )
    reach = int((rows >= 0).sum(axis=1).max()) if rows.size else 0
    if reach > MOVE_SITES:
        raise ValueError(
            f"a move reads {reach} orientation sites and the kernel holds {MOVE_SITES}. Two"
            " nodes that move with their neighbours are under three beads apart."
        )
    move_sites: I32Array = np.full((n, MOVE_SITES), -1, dtype=np.int32)
    keep = min(int(rows.shape[1]), MOVE_SITES)
    move_sites[:, :keep] = rows[:, :keep]
    return nbr_idx, nbr_w, nbr_valid, move_sites


def _movable_beads(
    fixed: np.ndarray[Any, Any], centre: np.ndarray[Any, Any] | None, use_cap: bool
) -> I64Array:
    """Returns the beads that the kernel draws from: every bead under the cap, else the free
    beads and the nodes of `centre`.
    """
    if use_cap:
        return np.arange(fixed.shape[0], dtype=np.int64)
    free = ~fixed if centre is None else (~fixed | centre)
    return np.ascontiguousarray(np.where(free)[0], dtype=np.int64)


def _cap_radii(
    fixed: np.ndarray[Any, Any], dtn: np.ndarray[Any, Any], cap_frac: float, B: int
) -> np.ndarray[Any, Any]:
    """Returns the cap radius per bead padded to B, `cap_frac` mean bonds for an anchor and zero
    otherwise.
    """
    out = np.zeros(B, dtype=np.float32)
    if cap_frac > 0.0:
        n = int(fixed.shape[0])
        scale = float(np.asarray(dtn).mean()) if np.asarray(dtn).size else 1.0
        out[:n] = np.where(fixed, cap_frac * scale, 0.0)
    return out


def _smooth_tensor_bytes(B: int, A: int, M: int, use_heat: bool, use_orn: bool) -> int:
    """Returns the device tensor bytes of one IB of the smooth kernel at (B, A, M), the sum of
    every input and output array. The (B, B) heat tensor dominates when heat is on.
    """
    f4, i4, b1 = 4, 4, 1  # bytes of float32, int32 and bool
    A_ = A if use_orn else 1
    M_ = M if use_orn else 1
    inp = (
        B * 3 * f4  # pos_k
        + 5 * f4  # ss/se/sh/so/sc
        + A_ * 3 * f4  # anchor_orn_k
        + B * f4  # dtn_k
        + B * i4  # movable_k
        + B * 3 * f4  # cap_home
        + B * f4  # cap_r
        + (B * B * f4 if use_heat else f4)  # heat_k
        + A_ * i4  # anchor_ar_k
        + B * MOVE_SITES * i4  # move_sites
        + A_ * M_ * (i4 + f4 + b1)  # nbr idx / w / valid
        + B * b1  # is_L_k
        + f4  # step_size_k
        + 6 * f4  # excl_r0 + conf cx,cy,cz,R,w
        + 2 * i4  # n_active / n_movable
        + B * i4  # centre_of
    )
    out = B * 3 * f4 + 5 * f4 + A_ * 3 * f4 + b1  # pos_f, scores, anchor_orn, converged
    return inp + out


def _resolve_smooth_max_k(
    big_b: int, big_a: int, big_m: int, use_heat: bool, use_orn: bool, settings: "Settings"
) -> tuple[int, str]:
    """Resolves the vmap width of the smooth batch, the number of IBs per launch.

    `mc_executor_jax_batch_width_smooth` is an integer or "auto". "auto" takes the bytes per IB,
    applies `memory.XLA_PEAK_OVERHEAD` and solves for the largest K in the device budget. Without
    a budget, on the CPU, it uses `32768 / B`. Returns (max_k, basis).
    """
    w = str(settings.mc_executor_jax_batch_width_smooth).strip().lower()
    if w != "auto":
        return max(1, int(w)), "explicit"
    per_ib = _smooth_tensor_bytes(big_b, big_a, big_m, use_heat, use_orn)
    # The host stages one (K, B, B) heat buffer before the transfer. K is bounded by host RAM
    # too.
    k_host = max_k_for_host(big_b * big_b * 4 if use_heat else per_ib)
    budget = jax_device_budget_bytes()
    if budget is None:
        k = max(1, 32768 // max(1, big_b))
        basis = "auto-fallback"
    else:
        k = max_k_for_bytes(per_ib, 0, budget)
        basis = "auto-bytes"
    if k_host is not None and k_host < k:
        return k_host, basis + "+host"
    return k, basis


def mc_smooth_jax_batch(
    problems: list[dict[str, Any]],
    settings: "Settings",
) -> list[tuple[float, np.ndarray[Any, Any]]]:
    """Anneals K different IBs in one vmapped kernel.

    Each problem is a dict with `pos` (n, 3), `dtn` (n,), `fixed` (n,) and `step_size` and
    optionally `heat_dist` (n, n), `char_orientations` (n,), `anchor_neighbors`,
    `anchor_neighbor_weights` and `centre` (n,), the nodes of the three bead move. All problems
    must share the energy term flags and have `centre` or lack it together.

    Returns one `(score, final_pos)` per problem in input order and does not change the inputs.
    IBs beyond the vmap width of `_resolve_smooth_max_k` run as further launches.
    """
    if not problems:
        return []
    use_heat = problems[0].get("heat_dist") is not None
    use_orn = (
        problems[0].get("char_orientations") is not None
        and problems[0].get("anchor_neighbors") is not None
        and problems[0].get("anchor_neighbor_weights") is not None
    )
    shapes = [_padded_shape(p, settings, use_orn) for p in problems]
    plan = _chunk_plan(shapes, use_heat, use_orn, settings)
    if len(plan) == 1:
        return _mc_smooth_jax_batch_chunk(problems, settings)

    LOG.debug(
        "region-batch[smooth]: %d IBs at B<=%d heat=%s in %d sub-batches",
        len(problems),
        max(s[0] for s in shapes),
        use_heat,
        len(plan),
    )
    out: list[tuple[float, np.ndarray[Any, Any]] | None] = [None] * len(problems)
    for ci, idx in enumerate(plan, start=1):
        got = _mc_smooth_jax_batch_chunk(
            [problems[i] for i in idx], settings, f", chunk {ci}/{len(plan)}"
        )
        for i, r in zip(idx, got, strict=True):
            out[i] = r
    assert all(r is not None for r in out)
    return cast("list[tuple[float, np.ndarray[Any, Any]]]", out)


def _padded_shape(
    problem: dict[str, Any], settings: "Settings", use_orn: bool
) -> tuple[int, int, int]:
    """Returns the bead, anchor and neighbour extents that one IB is padded to in a launch."""
    bucket = bool(settings.mc_executor_jax_bucket_shapes)
    n = int(problem["pos"].shape[0])
    b = jax_bucket_for(n) if bucket else n
    if not use_orn:
        return b, 1, 1
    anchors = int(np.count_nonzero(problem["fixed"]))
    nbrs = max(
        max((len(problem["anchor_neighbors"].get(k, [])) for k in range(anchors)), default=1), 1
    )
    a = jax_bucket_for(anchors, ANCHOR_BUCKETS) if bucket else anchors
    m = jax_bucket_for(nbrs, NBR_BUCKETS) if bucket else nbrs
    return b, a, m


def _chunk_plan(
    shapes: list[tuple[int, int, int]], use_heat: bool, use_orn: bool, settings: "Settings"
) -> list[list[int]]:
    """Splits IB indices into launches, each as wide as device memory allows.

    A launch pads every member to its largest and with heat that padding grows with the square
    of the bead extent. The largest IBs are taken first. The shape of a launch is set by
    its first member.
    """
    order = sorted(range(len(shapes)), key=lambda i: shapes[i], reverse=True)
    plan: list[list[int]] = []
    cap = 0
    for i in order:
        if not plan or len(plan[-1]) >= cap:
            b, a, m = shapes[i]
            cap = _resolve_smooth_max_k(b, a, m, use_heat, use_orn, settings)[0]
            plan.append([i])
        else:
            plan[-1].append(i)
    return plan


def _mc_smooth_jax_batch_chunk(
    problems: list[dict[str, Any]],
    settings: "Settings",
    chunk_tag: str = "",
) -> list[tuple[float, np.ndarray[Any, Any]]]:
    """Runs one vmapped kernel launch for up to `max_k` IBs."""
    if not jax_is_available():
        raise RuntimeError("settings.mc_backend='jax' but JAX is not installed.")
    import jax
    import jax.numpy as jnp

    K = len(problems)
    if K == 0:
        return []

    # Every IB is padded to the largest (B, A, M) of the group.
    bucket = bool(settings.mc_executor_jax_bucket_shapes)
    use_orn = (
        problems[0].get("char_orientations") is not None
        and problems[0].get("anchor_neighbors") is not None
        and problems[0].get("anchor_neighbor_weights") is not None
    )
    use_heat = problems[0].get("heat_dist") is not None
    use_excl = bool(settings.use_excluded_volume) and bool(settings.exclusion_apply_to_smooth)
    use_conf = bool(settings.use_confinement) and bool(settings.confinement_apply_to_smooth)
    # `SmoothStage.batch_key` holds the track flag. Every problem of a group agrees with
    # `problems[0]`.
    use_aff = problems[0].get("compartment") is not None and bool(settings.use_compartments)
    use_wall = bool(getattr(settings, "smooth_hard_wall", False))
    use_cap = float(getattr(settings, "smooth_anchor_cap", 0.0)) > 0.0
    prefetch = max(1, int(getattr(settings, "smooth_prefetch", 1)))
    # The three bead move is on for the whole group, as the settings of the run decide it.
    use_centre = problems[0].get("centre") is not None and not use_cap

    Bs, As, Ms = [], [], []
    for p in problems:
        n_i = int(p["pos"].shape[0])
        Bs.append(jax_bucket_for(n_i) if bucket else n_i)
        if use_orn:
            anchors_i = int(np.count_nonzero(p["fixed"]))
            nbrs_i = max(
                (len(p["anchor_neighbors"].get(k, [])) for k in range(anchors_i)), default=1
            )
            nbrs_i = max(nbrs_i, 1)
            As.append(jax_bucket_for(anchors_i, ANCHOR_BUCKETS) if bucket else anchors_i)
            Ms.append(jax_bucket_for(nbrs_i, NBR_BUCKETS) if bucket else nbrs_i)
        else:
            As.append(1)
            Ms.append(1)
    B, A, M = max(Bs), max(As), max(Ms)

    # Heat is the only quadratic input and is written into one (K, B, B) buffer. Stacking separate
    # arrays would hold every byte twice.
    heat_all = (
        np.zeros((len(problems), B, B), dtype=np.float32)
        if use_heat
        else np.zeros((len(problems), 1, 1), dtype=np.float32)
    )
    preps = [
        _prep_smooth_problem_np(
            p["pos"],
            p["dtn"],
            p["fixed"],
            settings,
            p.get("char_orientations"),
            p.get("anchor_neighbors"),
            p.get("anchor_neighbor_weights"),
            p.get("heat_dist"),
            B,
            A,
            M,
            p.get("compartment"),
            heat_all[i] if use_heat else None,
            p.get("centre") if use_centre else None,
        )
        for i, p in enumerate(problems)
    ]

    def stack(key: str) -> Any:
        return jnp.asarray(np.stack([pr[key] for pr in preps], axis=0))

    pos_k = stack("pos")  # (K, B, 3)
    comp_cls_k = stack("comp_cls")  # (K, B)
    dtn_k = stack("dtn")
    movable_k = stack("movable")
    cap_home_k = stack("cap_home")
    cap_r_k = stack("cap_r")
    centre_of_k = stack("centre_of")
    heat_k = jnp.asarray(heat_all)  # (K, B, B), written by the prep
    anchor_ar_k = stack("anchor_ar")
    move_sites_k = stack("move_sites")
    nbr_idx_k = stack("nbr_idx")
    nbr_w_k = stack("nbr_w")
    nbr_valid_k = stack("nbr_valid")
    is_L_k = stack("is_L")
    n_active_k = jnp.asarray(np.array([pr["n_active"] for pr in preps], dtype=np.int32))
    n_movable_k = jnp.asarray(np.array([pr["n_movable"] for pr in preps], dtype=np.int32))
    excl_r0_k = jnp.asarray(np.array([pr["excl_r0"] for pr in preps], dtype=np.float32))
    conf_cx_k = jnp.asarray(np.array([pr["conf_cx"] for pr in preps], dtype=np.float32))
    conf_cy_k = jnp.asarray(np.array([pr["conf_cy"] for pr in preps], dtype=np.float32))
    conf_cz_k = jnp.asarray(np.array([pr["conf_cz"] for pr in preps], dtype=np.float32))
    conf_R_k = jnp.asarray(np.array([pr["conf_R"] for pr in preps], dtype=np.float32))

    # The affinity radii come from the shared resolver, per IB.
    from gnome3d.mc.numba.common import affinity_params  # noqa: PLC0415

    _affs = [
        affinity_params(
            settings,
            "smooth",
            float(np.asarray(p["dtn"]).mean()) if np.asarray(p["dtn"]).size else 1.0,
            p.get("compartment"),
        )
        for p in problems
    ]
    comp_r0_k = jnp.asarray(np.array([a.comp_r0 for a in _affs], dtype=np.float32))
    comp_w_v = float(_affs[0].comp_weight) if _affs[0].use_comp else 0.0
    comp_ea_v = float(_affs[0].comp_ea)
    comp_eb_v = float(_affs[0].comp_eb)
    conf_w_k = jnp.asarray(np.array([pr["conf_w"] for pr in preps], dtype=np.float32))
    step_size_k = jnp.asarray(np.array([float(p["step_size"]) for p in problems], dtype=np.float32))

    # The springs are those of the settings for every IB. Boosted IBs are batched apart.
    stretch_k = jnp.full((K,), jnp.float32(settings.spring_stretch))
    squeeze_k = jnp.full((K,), jnp.float32(settings.spring_squeeze))
    ang_k = jnp.full((K,), jnp.float32(settings.spring_angular))

    excl_skip = int(settings.exclusion_skip_neighbors)
    n_steps_per_batch = int(settings.mc_stop_steps_smooth)
    heat_weight_v = float(settings.subanchor_heatmap_dist_weight) if use_heat else 0.0
    motif_weight_v = orientation_weight(settings) if use_orn else 0.0
    excl_w_v = float(settings.exclusion_weight) if use_excl else 0.0

    grid_g, grid_cap = 0, 0
    if (
        bool(getattr(settings, "smooth_jax_grid", False))
        and use_excl
        and B >= int(getattr(settings, "smooth_jax_grid_min_beads", 4096))
    ):
        grid_g, grid_cap = _grid_static(
            [
                (np.asarray(p["pos"], dtype=np.float32), float(pr["excl_r0"]))
                for p, pr in zip(problems, preps, strict=True)
            ]
        )
    bundle = _build_smooth_kernel(
        n_steps_per_batch, excl_skip, use_heat, use_orn, M, use_aff, use_wall, use_cap, prefetch,
        grid_g, grid_cap, use_centre,
    )  # fmt: skip
    (
        _kb,
        _kf,
        _init_smooth,
        _init_excl,
        _init_heat,
        _init_confine,
        _init_anchor_orn,
        _init_orn_score,
        kernel_full_mp,
        _init_affinity,
        init_k,
    ) = bundle

    dist_w = jnp.float32(settings.smooth_dist_weight)
    ang_w = jnp.float32(settings.smooth_angle_weight)
    symmetric = jnp.bool_(bool(getattr(settings, "motifs_symmetric", True)))

    zeros_k = jnp.zeros((K,), jnp.float32)
    ss_k = init_k["smooth"](pos_k, dtn_k, stretch_k, squeeze_k, ang_k, dist_w, ang_w, n_active_k)
    se_k = (
        init_k["excl"](pos_k, excl_r0_k, jnp.float32(excl_w_v), n_active_k) if use_excl else zeros_k
    )
    if use_aff:
        # Added to the excluded volume score, which is counted the same way.
        se_k = se_k + init_k["affinity"](
            pos_k,
            comp_cls_k,
            comp_r0_k,
            jnp.float32(comp_w_v),
            jnp.float32(comp_ea_v),
            jnp.float32(comp_eb_v),
            n_active_k,
        )
    sh_k = init_k["heat"](pos_k, heat_k, jnp.float32(heat_weight_v)) if use_heat else zeros_k
    sc_k = (
        init_k["confine"](pos_k, conf_cx_k, conf_cy_k, conf_cz_k, conf_R_k, conf_w_k, n_active_k)
        if use_conf
        else zeros_k
    )
    if use_orn:
        anchor_orn_k = init_k["anchor_orn"](pos_k, anchor_ar_k, is_L_k, n_active_k)
        so_k = init_k["orn_score"](
            anchor_orn_k, nbr_idx_k, nbr_w_k, nbr_valid_k, jnp.float32(motif_weight_v), symmetric
        )
    else:
        anchor_orn_k = jnp.zeros((K, A, 3), jnp.float32)
        so_k = zeros_k

    # The launch key comes from the scope alone and each chain folds its own seed. Neither the
    # composition of the batch nor the slot of a chain reaches the draw.
    base_key = jax.random.PRNGKey(stable_seed_offset(log.current()))
    # A problem without a seed falls back to its slot.
    chain_seed_k = jnp.asarray(
        np.array(
            [int(p.get("seed", i) or 0) & 0x7FFFFFFF for i, p in enumerate(problems)],
            dtype=np.uint32,
        )
    )

    _desc = ("" if not use_heat else ", heat") + (
        "" if not use_orn else f", orientation ({A} anchors, <={M} motif-nbrs/anchor)"
    )
    if use_centre:
        _desc += ", three bead move of a node"
    log_kernel_start(LOG, "smooth", "mc", K, B, f"sequential single-bead{_desc}{chunk_tag}")
    t0 = time.perf_counter()
    out = kernel_full_mp(
        pos_k,
        ss_k,
        se_k,
        sh_k,
        so_k,
        sc_k,
        anchor_orn_k,
        jnp.float32(settings.max_temp_smooth),
        dtn_k,
        movable_k,
        heat_k,
        comp_cls_k,
        anchor_ar_k,
        move_sites_k,
        nbr_idx_k,
        nbr_w_k,
        nbr_valid_k,
        is_L_k,
        step_size_k,
        jnp.float32(settings.dt_temp_smooth),
        jnp.float32(settings.jump_scale_smooth),
        jnp.float32(settings.jump_coef_smooth),
        stretch_k,
        squeeze_k,
        ang_k,
        dist_w,
        ang_w,
        excl_r0_k,
        jnp.float32(excl_w_v),
        jnp.float32(heat_weight_v),
        jnp.float32(motif_weight_v),
        symmetric,
        conf_cx_k,
        conf_cy_k,
        conf_cz_k,
        conf_R_k,
        conf_w_k,
        comp_r0_k,
        jnp.float32(comp_w_v),
        jnp.float32(comp_ea_v),
        jnp.float32(comp_eb_v),
        base_key,
        chain_seed_k,
        jnp.float32(settings.mc_stop_improvement_smooth),
        jnp.int32(settings.mc_stop_successes_smooth),
        jnp.float32(1e-6),
        n_active_k,
        n_movable_k,
        cap_home_k,
        cap_r_k,
        centre_of_k,
    )
    pos_f, ss_f, se_f, sh_f, so_f, sc_f, _ao_f, _final_score, iter_count, converged = out
    score_per_chain = np.asarray(ss_f + se_f + sh_f + so_f + sc_f)  # syncs the device
    pos_f_np = np.asarray(pos_f)

    n_steps_smooth = int(settings.mc_stop_steps_smooth)
    it_s, n_conv_s = int(iter_count), int(np.asarray(converged).sum())
    log_kernel_done(
        LOG,
        "smooth",
        "mc",
        K,
        time.perf_counter() - t0,
        f"{it_s} rounds ({it_s * n_steps_smooth} steps), {n_conv_s}/{K} converged",
    )

    results: list[tuple[float, np.ndarray[Any, Any]]] = []
    for i, pr in enumerate(preps):
        n_i = pr["n"]
        results.append((float(score_per_chain[i]), pos_f_np[i, :n_i].astype(np.float32)))
    return results
