"""Confinement term: soft spherical envelope, per bead.

    E(p) = weight * ((|r_p - c| - R) / R)^2   if |r_p - c| > R
         = 0                                   otherwise

Single-counted globally (``delta_factor = 1.0``).  Verbatim math from
``gnome3d.mc.numba`` (njit) and ``gnome3d.mc.jax`` (jnp), pinned by the parity
test.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

from gnome3d.mc.terms.base import NJIT, Term, njit


class ConfP(NamedTuple):
    """Confinement parameters — sphere centre ``(cx, cy, cz)``, radius ``R``."""

    cx: float
    cy: float
    cz: float
    R: float
    weight: float


# ---- numba ---------------------------------------------------------------


@njit(**NJIT)
def conf_local_nb(pos: Any, p: int, prm: ConfP) -> float:
    dx = pos[p, 0] - prm.cx
    dy = pos[p, 1] - prm.cy
    dz = pos[p, 2] - prm.cz
    r = math.sqrt(dx * dx + dy * dy + dz * dz)
    if r <= prm.R:
        return 0.0
    rel = (r - prm.R) / prm.R
    return prm.weight * rel * rel


@njit(**NJIT)
def conf_init_nb(pos: Any, prm: ConfP) -> float:
    n = pos.shape[0]
    err = 0.0
    for p in range(n):
        err += conf_local_nb(pos, p, prm)
    return err


# ---- JAX -----------------------------------------------------------------


def conf_local_jax(pos: Any, p: Any, p_pos: Any, prm: ConfP, n_active: Any) -> Any:
    import jax.numpy as jnp

    dx = p_pos[0] - prm.cx
    dy = p_pos[1] - prm.cy
    dz = p_pos[2] - prm.cz
    r = jnp.sqrt(dx * dx + dy * dy + dz * dz)
    rel = (r - prm.R) / jnp.maximum(prm.R, 1e-30)
    contrib = prm.weight * rel * rel
    return jnp.where(r > prm.R, contrib, 0.0)


def conf_init_jax(pos: Any, prm: ConfP, n_active: Any) -> Any:
    import jax
    import jax.numpy as jnp

    def _body(carry: Any, i: Any) -> tuple[Any, None]:
        c = conf_local_jax(pos, i, pos[i], prm, n_active)
        return carry + jnp.where(i < n_active, c, 0.0), None

    total, _ = jax.lax.scan(_body, jnp.float32(0.0), jnp.arange(pos.shape[0]))
    return total


CONFINEMENT = Term(
    name="confinement",
    params=ConfP,
    nb_local=conf_local_nb,
    nb_init=conf_init_nb,
    jax_local=conf_local_jax,
    jax_init=conf_init_jax,
    delta_factor=1.0,
)
