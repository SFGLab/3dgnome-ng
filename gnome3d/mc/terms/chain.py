"""Chain structure term: adjacent-bead stretch/squeeze bonds + bend angles.

This is the SMOOTH/IB structure energy (numba ``_local_smooth_nb``/``_init_smooth_nb``,
JAX ``_local_smooth_at``/``_init_smooth_single``).  IB uses it with ``ang_k = 0``
(angles vanish — too few beads for stable angle statistics).  Single-counted
(``delta_factor = 1.0``).

Per-IB bond targets ``dtn`` live in the param bundle alongside the spring
constants, since a term's "params" are everything it needs besides
``pos``/``p``/``p_pos``/``n_active``.  numba reads ``dtn`` as an array field of the
typed tuple; JAX reads it as a pytree leaf (vmaps per-IB).
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

from gnome3d.mc.terms.base import NJIT, Term, njit


class ChainP(NamedTuple):
    """Chain-structure parameters.  ``dtn`` = (n-1,) bond-length targets."""

    dtn: Any
    stretch_k: float
    squeeze_k: float
    ang_k: float
    dist_w: float
    ang_w: float


# ---- numba ---------------------------------------------------------------


@njit(**NJIT)
def _smooth_len_nb(
    pos: Any, dtn: Any, i: int, stretch_k: float, squeeze_k: float, dist_w: float
) -> float:
    dx = pos[i, 0] - pos[i + 1, 0]
    dy = pos[i, 1] - pos[i + 1, 1]
    dz = pos[i, 2] - pos[i + 1, 2]
    d = math.sqrt(dx * dx + dy * dy + dz * dz)
    e = dtn[i]
    if e < 1e-6:
        e = 1e-6
    rel = (d - e) / e
    k = stretch_k if rel >= 0.0 else squeeze_k
    return rel * rel * k * dist_w


@njit(**NJIT)
def _smooth_ang_nb(pos: Any, i: int, ang_k: float, ang_w: float) -> float:
    v1x = pos[i, 0] - pos[i + 1, 0]
    v1y = pos[i, 1] - pos[i + 1, 1]
    v1z = pos[i, 2] - pos[i + 1, 2]
    v2x = pos[i + 1, 0] - pos[i + 2, 0]
    v2y = pos[i + 1, 1] - pos[i + 2, 1]
    v2z = pos[i + 1, 2] - pos[i + 2, 2]
    n1 = math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
    n2 = math.sqrt(v2x * v2x + v2y * v2y + v2z * v2z)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cos_a = (v1x * v2x + v1y * v2y + v1z * v2z) / (n1 * n2)
    if cos_a > 1.0:
        cos_a = 1.0
    if cos_a < -1.0:
        cos_a = -1.0
    ang = 1.0 - (cos_a + 1.0) * 0.5
    return ang * ang * ang * ang_k * ang_w


@njit(**NJIT)
def chain_local_nb(pos: Any, p: int, prm: ChainP) -> float:
    n = pos.shape[0]
    sc = 0.0
    i = p - 1
    if 0 <= i < n - 1:
        sc += _smooth_len_nb(pos, prm.dtn, i, prm.stretch_k, prm.squeeze_k, prm.dist_w)
    if 0 <= p < n - 1:
        sc += _smooth_len_nb(pos, prm.dtn, p, prm.stretch_k, prm.squeeze_k, prm.dist_w)
    for off in range(-2, 1):
        i = p + off
        if 0 <= i < n - 2:
            sc += _smooth_ang_nb(pos, i, prm.ang_k, prm.ang_w)
    return sc


@njit(**NJIT)
def chain_init_nb(pos: Any, prm: ChainP) -> float:
    n = pos.shape[0]
    sc = 0.0
    for i in range(n - 1):
        sc += _smooth_len_nb(pos, prm.dtn, i, prm.stretch_k, prm.squeeze_k, prm.dist_w)
    for i in range(n - 2):
        sc += _smooth_ang_nb(pos, i, prm.ang_k, prm.ang_w)
    return sc


# ---- JAX -----------------------------------------------------------------


def _smooth_len_jax(pa: Any, pb: Any, e: Any, stretch_k: Any, squeeze_k: Any, dist_w: Any) -> Any:
    import jax.numpy as jnp

    diff = pa - pb
    d = jnp.sqrt(jnp.sum(diff * diff))
    e_safe = jnp.maximum(e, 1e-6)
    rel = (d - e_safe) / e_safe
    k = jnp.where(rel >= 0, stretch_k, squeeze_k)
    return rel * rel * k * dist_w


def _smooth_ang_jax(pa: Any, pb: Any, pc: Any, ang_k: Any, ang_w: Any) -> Any:
    import jax.numpy as jnp

    v1 = pa - pb
    v2 = pb - pc
    n1 = jnp.sqrt(jnp.sum(v1 * v1))
    n2 = jnp.sqrt(jnp.sum(v2 * v2))
    scale = jnp.where(jnp.logical_or(n1 < 1e-12, n2 < 1e-12), 0.0, 1.0)
    cos_a = jnp.sum(v1 * v2) / jnp.maximum(n1 * n2, 1e-30)
    cos_a = jnp.clip(cos_a, -1.0, 1.0)
    ang = 1.0 - (cos_a + 1.0) * 0.5
    return scale * ang * ang * ang * ang_k * ang_w


def chain_local_jax(pos: Any, p: Any, p_pos: Any, prm: ChainP, n_active: Any) -> Any:
    import jax.numpy as jnp

    dtn, stretch_k, squeeze_k, ang_k, dist_w, ang_w = prm
    n = pos.shape[0]
    a_pm1 = pos[jnp.maximum(p - 1, 0)]
    bond_L_ok = jnp.logical_and(p - 1 >= 0, p - 1 < n_active - 1)
    bond_L = jnp.where(
        bond_L_ok,
        _smooth_len_jax(a_pm1, p_pos, dtn[jnp.maximum(p - 1, 0)], stretch_k, squeeze_k, dist_w),
        0.0,
    )
    a_pp1 = pos[jnp.minimum(p + 1, n - 1)]
    bond_R_ok = jnp.logical_and(p >= 0, p < n_active - 1)
    bond_R = jnp.where(
        bond_R_ok,
        _smooth_len_jax(p_pos, a_pp1, dtn[jnp.minimum(p, n - 2)], stretch_k, squeeze_k, dist_w),
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
        return jnp.where(valid, _smooth_ang_jax(a0, a1, a2, ang_k, ang_w), 0.0)

    return bond_L + bond_R + angle_at(-2) + angle_at(-1) + angle_at(0)


def chain_init_jax(pos: Any, prm: ChainP, n_active: Any) -> Any:
    import jax
    import jax.numpy as jnp

    dtn, stretch_k, squeeze_k, ang_k, dist_w, ang_w = prm
    n = pos.shape[0]

    def _bond_body(carry: Any, i: Any) -> tuple[Any, None]:
        val = _smooth_len_jax(pos[i], pos[i + 1], dtn[i], stretch_k, squeeze_k, dist_w)
        return carry + jnp.where(i + 1 < n_active, val, 0.0), None

    def _angle_body(carry: Any, i: Any) -> tuple[Any, None]:
        val = _smooth_ang_jax(pos[i], pos[i + 1], pos[i + 2], ang_k, ang_w)
        return carry + jnp.where(i + 2 < n_active, val, 0.0), None

    bonds_total, _ = jax.lax.scan(_bond_body, jnp.float32(0.0), jnp.arange(n - 1))
    angles_total, _ = jax.lax.scan(_angle_body, jnp.float32(0.0), jnp.arange(n - 2))
    return bonds_total + angles_total


CHAIN = Term(
    name="chain",
    params=ChainP,
    nb_local=chain_local_nb,
    nb_init=chain_init_nb,
    jax_local=chain_local_jax,
    jax_init=chain_init_jax,
    delta_factor=1.0,
)
