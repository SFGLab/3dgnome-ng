"""Excluded-volume term: harmonic soft repulsion, cutoff at ``r0``.

    E_pair(d) = weight * ((r0 - d) / r0)^2   if d < r0
              = 0                            otherwise

Global score double-counts pairs (matches the heat-energy convention):
``sum_{i != j, |i-j| > skip} E_pair(d_ij)``, so the per-step delta is
``2 * (local_curr - local_prev)`` (``delta_factor = 2.0``).

The numba (`*_nb`) and JAX (`*_jax`) bodies are the verbatim math from
``gnome3d.mc.numba`` and ``gnome3d.mc.jax`` respectively; ``gnome3d.mc.numba``
now imports the njit helpers from here (thin wrappers preserve its old call
signatures), and the parity test pins ``nb == jax``.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

from gnome3d.mc.terms.base import NJIT, Term, njit


class ExclP(NamedTuple):
    """Excluded-volume parameters — shared by both backends (numba: typed tuple,
    JAX: pytree).  ``skip`` excludes chain neighbours within ``|i-j| <= skip``."""

    r0: float
    weight: float
    skip: int


# ---- numba ---------------------------------------------------------------


@njit(**NJIT)
def _excl_pair_nb(d: float, r0: float, weight: float) -> float:
    if d >= r0:
        return 0.0
    rel = (r0 - d) / r0
    return weight * rel * rel


@njit(**NJIT)
def ev_local_nb(pos: Any, p: int, prm: ExclP) -> float:
    n = pos.shape[0]
    err = 0.0
    for i in range(n):
        diff = i - p
        if diff < 0:
            diff = -diff
        if diff <= prm.skip:
            continue
        dx = pos[i, 0] - pos[p, 0]
        dy = pos[i, 1] - pos[p, 1]
        dz = pos[i, 2] - pos[p, 2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        err += _excl_pair_nb(d, prm.r0, prm.weight)
    return err


@njit(**NJIT)
def ev_init_nb(pos: Any, prm: ExclP) -> float:
    n = pos.shape[0]
    err = 0.0
    for i in range(n):
        row_err = 0.0
        for j in range(n):
            diff = i - j
            if diff < 0:
                diff = -diff
            if diff <= prm.skip:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            row_err += _excl_pair_nb(d, prm.r0, prm.weight)
        err += row_err
    return err


# ---- JAX -----------------------------------------------------------------


def ev_local_jax(pos: Any, p: Any, p_pos: Any, prm: ExclP, n_active: Any) -> Any:
    import jax.numpy as jnp

    n = pos.shape[0]
    diff = pos - p_pos
    d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
    rel = jnp.maximum(0.0, (prm.r0 - d) / prm.r0)
    contrib = prm.weight * rel * rel
    idx = jnp.arange(n)
    # Exclude pad beads (idx >= n_active) from the pairwise sum; unbucketed
    # n_active == n so this is a no-op.
    in_range = jnp.logical_and(jnp.abs(idx - p) > prm.skip, idx < n_active)
    return jnp.sum(jnp.where(in_range, contrib, 0.0))


def ev_init_jax(pos: Any, prm: ExclP, n_active: Any) -> Any:
    import jax
    import jax.numpy as jnp

    n = pos.shape[0]
    idx = jnp.arange(n)

    def scan_body(carry: Any, i: Any) -> tuple[Any, None]:
        diff = pos - pos[i]
        d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
        rel = jnp.maximum(0.0, (prm.r0 - d) / prm.r0)
        contrib = prm.weight * rel * rel
        in_range = jnp.logical_and(jnp.abs(idx - i) > prm.skip, idx < n_active)
        row = jnp.where(i < n_active, jnp.sum(jnp.where(in_range, contrib, 0.0)), 0.0)
        return carry + row, None

    total, _ = jax.lax.scan(scan_body, jnp.float32(0.0), idx)
    return total


EXCLUDED_VOLUME = Term(
    name="excluded_volume",
    params=ExclP,
    nb_local=ev_local_nb,
    nb_init=ev_init_nb,
    jax_local=ev_local_jax,
    jax_init=ev_init_jax,
    delta_factor=2.0,
)
