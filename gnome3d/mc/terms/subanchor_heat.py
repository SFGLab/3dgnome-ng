"""Subanchor-heat term (SMOOTH kernel, optional).

All-pairs distance matching against a per-IB expected-distance matrix
``heat_dist`` (zero/sub-1e-6 entries skipped).  Double-counted
(``delta_factor = 2.0``), matching the heat-energy convention.  Verbatim from
numba ``_local_heat_nb``/``_init_heat_nb`` and JAX ``_local_heat_at``/
``_init_heat_single``.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

from gnome3d.mc.terms.base import NJIT, Term, njit


class HeatP(NamedTuple):
    """Subanchor-heat parameters.  ``heat_dist`` = (n, n) expected distances."""

    heat_dist: Any
    heat_weight: float


# ---- numba ---------------------------------------------------------------


@njit(**NJIT)
def heat_local_nb(pos: Any, p: int, prm: HeatP) -> float:
    heat_dist = prm.heat_dist
    n = pos.shape[0]
    err = 0.0
    for i in range(n):
        if i == p:
            continue
        exp_d = heat_dist[i, p]
        if exp_d < 1e-6:
            continue
        dx = pos[i, 0] - pos[p, 0]
        dy = pos[i, 1] - pos[p, 1]
        dz = pos[i, 2] - pos[p, 2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        rel = (d - exp_d) / exp_d
        err += rel * rel
    return err * prm.heat_weight


@njit(**NJIT)
def heat_init_nb(pos: Any, prm: HeatP) -> float:
    heat_dist = prm.heat_dist
    n = pos.shape[0]
    err = 0.0
    for i in range(n):
        row_err = 0.0
        for j in range(n):
            if i == j:
                continue
            exp_d = heat_dist[i, j]
            if exp_d < 1e-6:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            rel = (d - exp_d) / exp_d
            row_err += rel * rel
        err += row_err
    return err * prm.heat_weight


# ---- JAX -----------------------------------------------------------------


def heat_local_jax(pos: Any, p: Any, p_pos: Any, prm: HeatP, n_active: Any) -> Any:
    import jax.numpy as jnp

    heat_dist, heat_weight = prm
    n = pos.shape[0]
    diff = pos - p_pos
    d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
    exp_d = heat_dist[:, p]
    idx = jnp.arange(n)
    active = jnp.logical_and(idx != p, exp_d >= 1e-6)
    exp_d_safe = jnp.maximum(exp_d, 1e-6)
    rel = (d - exp_d_safe) / exp_d_safe
    contrib = rel * rel
    return heat_weight * jnp.sum(jnp.where(active, contrib, 0.0))


def heat_init_jax(pos: Any, prm: HeatP, n_active: Any) -> Any:
    import jax
    import jax.numpy as jnp

    heat_dist, heat_weight = prm
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


SUBANCHOR_HEAT = Term(
    name="subanchor_heat",
    params=HeatP,
    nb_local=heat_local_nb,
    nb_init=heat_init_nb,
    jax_local=heat_local_jax,
    jax_init=heat_init_jax,
    delta_factor=2.0,
)
