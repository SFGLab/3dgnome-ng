"""Orientation (CTCF-motif) term — SMOOTH only.

This is the special term: it scores the angle between cached per-anchor
orientation vectors (``anchor_orn``), not bead positions directly, and it carries
MUTABLE state (the ``anchor_orn`` cache, recomputed when an anchor bead moves).
Its init and local forms are deliberately DIFFERENT functions, not two views of
one formula:

  * ``score_orientation_full_nb`` (init) — unweighted-vs-... actually arc-WEIGHTED
    global score; the per-arc weights make the incremental local delta exact
    w.r.t. it (no drift).  This INTENTIONALLY diverges from the C++ reference
    ``calcScoreOrientation`` (unweighted, drifts) — see the orientation-MC-fix
    note.  ``energy.py``'s reference stays unweighted for the harness.
  * ``local_score_orientation_nb`` (local) — arc-weighted score for one anchor k.

Because of the mutable cache + the anchor-vs-bead indexing + the CSR (numba) vs
padded-dense (JAX) neighbour representations, this term does NOT share the uniform
``(pos, p, prm)`` local signature of the other terms; the composed driver handles
it specially (per-term state + an orientation-recompute hook).  STATUS: the numba
math is co-located here now (``gnome3d.mc.numba`` aliases these); the JAX side and
the `Term` record land with the generic-driver phase, where the orientation
neighbour representation is unified.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

from gnome3d.mc.terms.base import NJIT, Term, njit


class OrnP(NamedTuple):
    """Orientation parameters (padded-dense neighbour rep, all per-IB / (K,...)
    once stacked).  ``symmetric``/``motif_weight`` are broadcast to (K,) so the
    whole bundle vmaps on axis 0 uniformly."""

    bead_to_anchor_k: Any  # (B,) int: anchor whose orientation bead p affects, or -1
    anchor_ar: Any  # (A,) int: bead-index of each anchor
    is_L: Any  # (B,) bool: orientation flip, indexed by bead
    nbr_idx: Any  # (A, M) int
    nbr_w: Any  # (A, M) float
    nbr_valid: Any  # (A, M) bool
    motif_weight: Any
    symmetric: Any


@njit(**NJIT)
def calc_orientation_nb(pos: Any, cind: int, n: int, is_L: bool) -> tuple[float, float, float]:
    """Normalized orientation vector for the anchor whose bead index is ``cind``."""
    if cind == 0:
        ox = pos[cind + 1, 0] - pos[cind, 0]
        oy = pos[cind + 1, 1] - pos[cind, 1]
        oz = pos[cind + 1, 2] - pos[cind, 2]
    elif cind == n - 1:
        ox = pos[cind, 0] - pos[cind - 1, 0]
        oy = pos[cind, 1] - pos[cind - 1, 1]
        oz = pos[cind, 2] - pos[cind - 1, 2]
    else:
        ox = pos[cind + 1, 0] - pos[cind - 1, 0]
        oy = pos[cind + 1, 1] - pos[cind - 1, 1]
        oz = pos[cind + 1, 2] - pos[cind - 1, 2]
    if is_L:
        ox = -ox
        oy = -oy
        oz = -oz
    nm = math.sqrt(ox * ox + oy * oy + oz * oz)
    if nm > 1e-12:
        ox /= nm
        oy /= nm
        oz /= nm
    return ox, oy, oz


@njit(**NJIT)
def score_orientation_full_nb(
    anchor_orn: Any,
    nbr_offsets: Any,
    nbr_indices: Any,
    nbr_weights: Any,
    motif_weight: float,
    symmetric: bool,
) -> float:
    """Global orientation score with arc weights; used for initialisation only."""
    n_anchors = anchor_orn.shape[0]
    err = 0.0
    for i in range(n_anchors):
        for ki in range(nbr_offsets[i], nbr_offsets[i + 1]):
            j = nbr_indices[ki]
            w = nbr_weights[ki]
            ax = anchor_orn[i, 0]
            ay = anchor_orn[i, 1]
            az = anchor_orn[i, 2]
            bx = anchor_orn[j, 0]
            by = anchor_orn[j, 1]
            bz = anchor_orn[j, 2]
            if not symmetric:
                bx = -bx
                by = -by
                bz = -bz
            dot = ax * bx + ay * by + az * bz
            ang = 1.0 - (dot + 1.0) * 0.5
            err += ang * ang * w
    return err * motif_weight


@njit(**NJIT)
def local_score_orientation_nb(
    anchor_orn: Any,
    k: int,
    nbr_offsets: Any,
    nbr_indices: Any,
    nbr_weights: Any,
    motif_weight: float,
    symmetric: bool,
) -> float:
    """Arc-weighted local orientation score for anchor ``k`` (the incremental
    delta unit: ``score_orn += 2*(local_curr - local_prev)``).  The weights make
    the delta exact w.r.t. `score_orientation_full_nb` — no drift."""
    err = 0.0
    for ki in range(nbr_offsets[k], nbr_offsets[k + 1]):
        j = nbr_indices[ki]
        w = nbr_weights[ki]
        ax = anchor_orn[k, 0]
        ay = anchor_orn[k, 1]
        az = anchor_orn[k, 2]
        bx = anchor_orn[j, 0]
        by = anchor_orn[j, 1]
        bz = anchor_orn[j, 2]
        if not symmetric:
            bx = -bx
            by = -by
            bz = -bz
        dot = ax * bx + ay * by + az * bz
        ang = 1.0 - (dot + 1.0) * 0.5
        err += ang * ang * w
    return err * motif_weight


# ---- JAX -----------------------------------------------------------------
# JAX uses a PADDED-DENSE neighbour representation (nbr_idx/nbr_w/nbr_valid,
# (n_anchors, max_nbrs)) where numba uses CSR (nbr_offsets/indices/weights); the
# scores agree (parity test converts one graph to both).  These are extracted
# from `_build_smooth_kernel`'s closures verbatim; the generic-driver phase wires
# them in (with the mutable anchor_orn cache as the orientation term's state).


def calc_orientation_jax(pos: Any, p: Any, p_pos: Any, ar: Any, is_L: Any) -> Any:
    """Orientation vector for the anchor at bead-index ``ar`` with ``pos[p]``
    replaced by ``p_pos`` (use ``p = -1`` sentinel for "no substitution")."""
    import jax.numpy as jnp

    n = pos.shape[0]
    pp1_idx = jnp.minimum(ar + 1, n - 1)
    pm1_idx = jnp.maximum(ar - 1, 0)
    a_ar = jnp.where(ar == p, p_pos, pos[ar])
    a_pp1 = jnp.where(pp1_idx == p, p_pos, pos[pp1_idx])
    a_pm1 = jnp.where(pm1_idx == p, p_pos, pos[pm1_idx])

    is_first = ar == 0
    is_last = ar == n - 1
    o_first = a_pp1 - a_ar
    o_last = a_ar - a_pm1
    o_mid = a_pp1 - a_pm1
    o = jnp.where(is_first, o_first, jnp.where(is_last, o_last, o_mid))
    o = jnp.where(is_L, -o, o)
    nm = jnp.sqrt(jnp.sum(o * o))
    return jnp.where(nm > 1e-12, o / jnp.maximum(nm, 1e-30), jnp.zeros_like(o))


def local_orientation_jax(
    anchor_orn: Any, k: Any, nbr_idx: Any, nbr_w: Any, nbr_valid: Any, motif_weight: Any, symmetric: Any
) -> Any:
    """Arc-weighted local orientation score for anchor ``k`` over its padded
    neighbour list.  Mirrors `local_score_orientation_nb`."""
    import jax.numpy as jnp

    neighbors_k = nbr_idx[k]
    weights_k = nbr_w[k]
    valid_k = nbr_valid[k]
    a = anchor_orn[k]
    b = anchor_orn[neighbors_k]
    b_signed = jnp.where(symmetric, b, -b)
    dot = jnp.sum(a[None, :] * b_signed, axis=1)
    ang = 1.0 - (dot + 1.0) * 0.5
    contrib = jnp.where(valid_k, ang * ang * weights_k, 0.0)
    return motif_weight * jnp.sum(contrib)


def init_anchor_orientations_jax(pos: Any, anchor_ar: Any, is_L: Any) -> Any:
    """(n_anchors, 3) initial orientation vectors (``is_L`` indexed by bead)."""
    import jax
    import jax.numpy as jnp

    def per_anchor(k_idx: Any) -> Any:
        ar = anchor_ar[k_idx]
        return calc_orientation_jax(pos, jnp.int32(-1), jnp.zeros((3,), dtype=pos.dtype), ar, is_L[ar])

    return jax.vmap(per_anchor)(jnp.arange(anchor_ar.shape[0]))


def init_orientation_score_jax(
    anchor_orn: Any, nbr_idx: Any, nbr_w: Any, nbr_valid: Any, motif_weight: Any, symmetric: Any
) -> Any:
    """Global orientation score (matches `score_orientation_full_nb`).  Sequential
    scan over anchors so anchor-bucket padding doesn't perturb the f32 order."""
    import jax
    import jax.numpy as jnp

    def _scan_body(carry: Any, k_idx: Any) -> tuple[Any, None]:
        loc = local_orientation_jax(anchor_orn, k_idx, nbr_idx, nbr_w, nbr_valid, motif_weight, symmetric)
        return carry + loc, None

    total, _ = jax.lax.scan(_scan_body, jnp.float32(0.0), jnp.arange(anchor_orn.shape[0]))
    return total


def orientation_step_jax(pos: Any, p: Any, old_p: Any, new_p: Any, orn: OrnP, anchor_orn: Any, n_active: Any):
    """Stateful per-step delta for the driver.  When bead ``p`` is adjacent to an
    anchor (``bead_to_anchor_k[p] >= 0``), recompute that anchor's orientation
    with ``p`` moved to ``new_p`` and return ``(local_curr - local_prev,
    anchor_orn_with_that_slot_updated)``.  Mirrors the orientation block of the
    hand-written smooth kernel exactly."""
    import jax.numpy as jnp

    orn_k = orn.bead_to_anchor_k[p]
    has = orn_k >= 0
    safe_k = jnp.maximum(orn_k, 0)
    prev = jnp.where(
        has, local_orientation_jax(anchor_orn, safe_k, orn.nbr_idx, orn.nbr_w, orn.nbr_valid, orn.motif_weight, orn.symmetric), 0.0
    )
    ar_p = orn.anchor_ar[safe_k]
    new_vec = calc_orientation_jax(pos, p, new_p, ar_p, orn.is_L[ar_p])
    trial = anchor_orn.at[safe_k].set(new_vec)
    curr = jnp.where(
        has, local_orientation_jax(trial, safe_k, orn.nbr_idx, orn.nbr_w, orn.nbr_valid, orn.motif_weight, orn.symmetric), 0.0
    )
    return (curr - prev), trial


ORIENTATION = Term(
    name="orientation",
    params=OrnP,
    nb_local=local_score_orientation_nb,
    nb_init=score_orientation_full_nb,
    jax_local=local_orientation_jax,
    jax_init=init_orientation_score_jax,
    delta_factor=2.0,
    jax_step=orientation_step_jax,
)
