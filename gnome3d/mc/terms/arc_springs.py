"""Arc-springs structure term (ARCS kernel).

All-pairs interaction from the expected-distance matrix ``exp``:
  * ``exp[i,p] < 0``   → repulsion ``1/d`` (d clamped to 1e-10);
  * ``exp[i,p] >= 1e-6`` → asymmetric spring ``((d-e)/e)^2 * k``;
  * else (``[0, 1e-6)``) → no contribution.

Single-counted (``delta_factor = 1.0``).  Verbatim from numba ``_local_arcs_nb``/
``_init_arcs_nb`` and JAX ``_local_arcs_at``/``_init_arcs`` (init sums the upper
triangle ``i<j``).
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

from gnome3d.mc.terms.base import NJIT, Term, njit


class ArcP(NamedTuple):
    """Arc-springs parameters.  ``exp`` = (n, n) expected-distance matrix."""

    exp: Any
    stretch_k: float
    squeeze_k: float


# ---- numba ---------------------------------------------------------------


@njit(**NJIT)
def arcs_local_nb(pos: Any, p: int, prm: ArcP) -> float:
    exp = prm.exp
    n = pos.shape[0]
    sc = 0.0
    for i in range(n):
        if i == p:
            continue
        e = exp[i, p]
        dx = pos[p, 0] - pos[i, 0]
        dy = pos[p, 1] - pos[i, 1]
        dz = pos[p, 2] - pos[i, 2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if e < 0.0:
            sc += 1.0 / (d if d > 1e-10 else 1e-10)
        elif e >= 1e-6:
            rel = (d - e) / e
            sc += rel * rel * (prm.stretch_k if rel >= 0.0 else prm.squeeze_k)
    return sc


@njit(**NJIT)
def arcs_init_nb(pos: Any, prm: ArcP) -> float:
    exp = prm.exp
    n = pos.shape[0]
    sc = 0.0
    for i in range(n):
        row_sc = 0.0
        for j in range(i + 1, n):
            e = exp[i, j]
            if -1e-10 < e < 1e-6:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if e < 0.0:
                row_sc += 1.0 / (d if d > 1e-10 else 1e-10)
            else:
                rel = (d - e) / e
                row_sc += rel * rel * (prm.stretch_k if rel >= 0.0 else prm.squeeze_k)
        sc += row_sc
    return sc


# ---- JAX -----------------------------------------------------------------


def arcs_local_jax(pos: Any, p: Any, p_pos: Any, prm: ArcP, n_active: Any) -> Any:
    import jax.numpy as jnp

    exp_mat, stretch_k, squeeze_k = prm
    n = pos.shape[0]
    diff = pos - p_pos
    d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
    e = exp_mat[:, p]
    idx = jnp.arange(n)
    not_self = idx != p
    is_repulse = jnp.logical_and(not_self, e < 0.0)
    is_spring = jnp.logical_and(not_self, e >= 1e-6)

    d_safe = jnp.maximum(d, 1e-10)
    rep = 1.0 / d_safe

    e_safe = jnp.maximum(e, 1e-6)
    rel = (d - e_safe) / e_safe
    k = jnp.where(rel >= 0, stretch_k, squeeze_k)
    spring = rel * rel * k

    contrib = jnp.where(is_repulse, rep, jnp.where(is_spring, spring, 0.0))
    return jnp.sum(contrib)


def arcs_init_jax(pos: Any, prm: ArcP, n_active: Any) -> Any:
    import jax
    import jax.numpy as jnp

    exp_mat, stretch_k, squeeze_k = prm
    n = pos.shape[0]
    idx = jnp.arange(n)

    def scan_body(carry: Any, i: Any) -> tuple[Any, None]:
        diff = pos - pos[i]
        d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
        e = exp_mat[:, i]
        above = idx > i
        is_repulse = jnp.logical_and(above, e <= -1e-10)
        is_spring = jnp.logical_and(above, e >= 1e-6)

        d_safe = jnp.maximum(d, 1e-10)
        rep = 1.0 / d_safe
        e_safe = jnp.maximum(e, 1e-6)
        rel = (d - e_safe) / e_safe
        k = jnp.where(rel >= 0, stretch_k, squeeze_k)
        spring = rel * rel * k

        row = jnp.where(is_repulse, rep, jnp.where(is_spring, spring, 0.0))
        return carry + jnp.sum(row), None

    total, _ = jax.lax.scan(scan_body, jnp.float32(0.0), idx)
    return total


ARC_SPRINGS = Term(
    name="arc_springs",
    params=ArcP,
    nb_local=arcs_local_nb,
    nb_init=arcs_init_nb,
    jax_local=arcs_local_jax,
    jax_init=arcs_init_jax,
    delta_factor=1.0,
)
