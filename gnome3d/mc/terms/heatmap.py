"""Heatmap structure term (HEATMAP kernel — chr/segment level).

Pairwise distance error against an expected-distance matrix ``exp_safe``
(``(d - e)/e`` squared), with a boolean ``skip`` mask (diagonal band + no-data
pairs).  ``exp_safe`` is 1.0 wherever ``skip`` is set, so the division is always
safe.  Double-counted (``delta_factor = 2.0``).

Verbatim from numba ``_local_heatmap_nb``/``_init_heatmap_nb`` and JAX
``_local_heatmap_at``/``_init_heatmap``.  The term holds the FULL ``skip`` matrix
and indexes ``skip[i, p]`` (the numba kernel used to pre-slice the column
``skip[:, p]``; indexing the matrix is the identical bool, and avoids the slice
copy — see the wrapper note in ``gnome3d.mc.numba``).
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

from gnome3d.mc.terms.base import NJIT, Term, njit


class HeatmapP(NamedTuple):
    """Heatmap parameters: ``exp_safe`` (n, n) expected distances (1.0 where
    skipped) and ``skip`` (n, n) bool mask."""

    exp_safe: Any
    skip: Any


# ---- numba ---------------------------------------------------------------


@njit(**NJIT)
def heatmap_local_nb(pos: Any, p: int, prm: HeatmapP) -> float:
    exp_safe, skip = prm.exp_safe, prm.skip
    n = pos.shape[0]
    sc = 0.0
    for i in range(n):
        if skip[i, p]:
            continue
        dx = pos[i, 0] - pos[p, 0]
        dy = pos[i, 1] - pos[p, 1]
        dz = pos[i, 2] - pos[p, 2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        e = exp_safe[i, p]
        err = (d - e) / e
        sc += err * err
    return sc


@njit(**NJIT)
def heatmap_init_nb(pos: Any, prm: HeatmapP) -> float:
    exp_safe, skip = prm.exp_safe, prm.skip
    n = pos.shape[0]
    sc = 0.0
    for i in range(n):
        row_sc = 0.0
        for j in range(n):
            if skip[i, j]:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            e = exp_safe[i, j]
            err = (d - e) / e
            row_sc += err * err
        sc += row_sc
    return sc


# ---- JAX -----------------------------------------------------------------


def heatmap_local_jax(pos: Any, p: Any, p_pos: Any, prm: HeatmapP, n_active: Any) -> Any:
    import jax.numpy as jnp

    exp_safe, skip = prm.exp_safe, prm.skip
    diff = pos - p_pos
    d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
    e = exp_safe[:, p]
    skip_col = skip[:, p]
    err = (d - e) / e
    contrib = err * err
    return jnp.sum(jnp.where(skip_col, 0.0, contrib))


def heatmap_init_jax(pos: Any, prm: HeatmapP, n_active: Any) -> Any:
    import jax
    import jax.numpy as jnp

    exp_safe, skip = prm.exp_safe, prm.skip
    n = pos.shape[0]

    def scan_body(carry: Any, i: Any) -> tuple[Any, None]:
        diff = pos - pos[i]
        d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
        e = exp_safe[:, i]
        skip_col = skip[:, i]
        err = (d - e) / e
        contrib = err * err
        return carry + jnp.sum(jnp.where(skip_col, 0.0, contrib)), None

    total, _ = jax.lax.scan(scan_body, jnp.float32(0.0), jnp.arange(n))
    return total


HEATMAP = Term(
    name="heatmap",
    params=HeatmapP,
    nb_local=heatmap_local_nb,
    nb_init=heatmap_init_nb,
    jax_local=heatmap_local_jax,
    jax_init=heatmap_init_jax,
    delta_factor=2.0,
)
