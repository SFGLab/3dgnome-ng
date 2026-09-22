"""The arcs energy and its gradient on the device, for the solver.

`gnome3d.mc.numba.arcs_solver.arcs_energy_grad` visits every pair of a chromosome on the CPU
against a dense target, and at chromosome scope that visit is most of a conformation's wall
while the GPU idles. This is the same energy, pair for pair and term for term, evaluated on the
device in row chunks, so an evaluation costs one read of the target matrix. Float32 on the
device, the rows summed in float64 on the host, so the two agree to about 1e-6 relative and
not to the bit. Float64 on the device would need JAX's 64 bit mode, which is a process wide
switch that changes the smooth kernel's integer draws and breaks its loop carries, so it is
not used.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gnome3d.types import F32Array, F64Array

CHUNK_ROWS = 256


class DeviceArcsEnergy:
    """The energy and gradient of `arcs_energy_grad` as a callable on flattened positions.

    Parameters
    ----------
    exp
        The target matrix, `exp[j, i]` the entry the CPU kernel reads for the pair (i, j).
    args
        The scalar parameters in `arcs_energy_grad`'s order after `exp`.
    """

    def __init__(self, exp: F64Array, args: tuple[Any, ...], w: F32Array | None = None) -> None:
        import jax
        import jax.numpy as jnp

        stretch, squeeze, rep_inv, bg_w = (float(a) for a in args[:4])
        cx, cy, cz, conf_r, conf_w, excl_r0, excl_w = (float(a) for a in args[4:11])
        excl_skip = int(args[11])
        n = int(exp.shape[0])
        self.n = n
        # Rows of the transpose are what the CPU kernel reads for one anchor.
        self._exp_t = jnp.asarray(np.ascontiguousarray(exp.T, dtype=np.float32))
        # The loop weights ride beside the targets, transposed the same way; None is every
        # pair at one and the branch is fixed at trace time.
        use_w = w is not None
        self._w_t = (
            jnp.asarray(np.ascontiguousarray(w.T, dtype=np.float32))
            if w is not None
            else jnp.zeros((1, 1), dtype=jnp.float32)
        )
        idx_all = jnp.arange(n, dtype=jnp.int32)
        centre = jnp.asarray(np.array([cx, cy, cz], dtype=np.float32))
        use_excl = excl_w > 0.0

        def row_terms(xi: Any, i: Any, e: Any, wi: Any, pos: Any) -> tuple[Any, Any]:
            """Anchor i's energy and gradient over every other anchor, then its confinement."""
            diff = xi[None, :] - pos
            d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
            dd = jnp.maximum(d, 1e-10)
            # Background spring, held at bg = -e, the log symmetric form below the target.
            is_bg = e <= -0.75
            bg = -e
            bg_safe = jnp.where(is_bg, bg, 1.0)
            below = d < bg
            rel_bg = jnp.where(below, (dd - bg) / dd, (d - bg) / bg_safe)
            e_bg = 0.5 * rel_bg * rel_bg * bg_w
            w_bg = jnp.where(
                below,
                2.0 * bg_w * rel_bg * bg / (dd * dd * dd),
                2.0 * bg_w * rel_bg / (bg_safe * dd),
            )
            # Truncated repulsion on every other arcless pair.
            is_rep = jnp.logical_and(e < 0.0, jnp.logical_not(is_bg))
            v = 1.0 / dd - rep_inv
            rep_on = jnp.logical_and(is_rep, v > 0.0)
            e_rep = 0.5 * v
            w_rep = -1.0 / (dd * dd * dd)
            # Arc spring.
            is_spr = e >= 1e-6
            e_safe = jnp.where(is_spr, e, 1.0)
            rel_s = (d - e) / e_safe
            k = jnp.where(rel_s >= 0.0, stretch, squeeze)
            if use_w:
                k = k * wi
            e_spr = 0.5 * rel_s * rel_s * k
            w_spr = 2.0 * k * rel_s / (e_safe * dd)
            ener = jnp.where(is_bg, e_bg, jnp.where(rep_on, e_rep, jnp.where(is_spr, e_spr, 0.0)))
            w = jnp.where(is_bg, w_bg, jnp.where(rep_on, w_rep, jnp.where(is_spr, w_spr, 0.0)))
            if use_excl:
                sep = jnp.abs(idx_all - i)
                ex_on = jnp.logical_and(d < excl_r0, sep > excl_skip)
                u = (excl_r0 - d) / excl_r0
                ener = ener + jnp.where(ex_on, excl_w * u * u, 0.0)
                w = w + jnp.where(ex_on, -4.0 * excl_w * u / (excl_r0 * dd), 0.0)
            other = idx_all != i
            ener = jnp.where(other, ener, 0.0)
            w = jnp.where(other, w, 0.0)
            ei = jnp.sum(ener)
            gi = jnp.sum(w[:, None] * diff, axis=0)
            rc = xi - centre
            r = jnp.sqrt(jnp.sum(rc * rc))
            outside = r > conf_r
            u_c = (r - conf_r) / conf_r
            ei = ei + jnp.where(outside, conf_w * u_c * u_c, 0.0)
            w_c = 2.0 * conf_w * u_c / (conf_r * jnp.maximum(r, 1e-10))
            gi = gi + jnp.where(outside, w_c, 0.0) * rc
            return ei, gi

        def energy_grad(x: Any, exp_t: Any, w_t: Any) -> tuple[Any, Any]:
            pos = x.reshape(n, 3)

            if use_w:

                def one(a: tuple[Any, Any, Any, Any]) -> tuple[Any, Any]:
                    return row_terms(a[0], a[1], a[2], a[3], pos)

                ei, gi = jax.lax.map(one, (pos, idx_all, exp_t, w_t), batch_size=CHUNK_ROWS)
            else:

                def one_plain(a: tuple[Any, Any, Any]) -> tuple[Any, Any]:
                    return row_terms(a[0], a[1], a[2], 1.0, pos)

                ei, gi = jax.lax.map(one_plain, (pos, idx_all, exp_t), batch_size=CHUNK_ROWS)
            return ei, gi.reshape(-1)

        self._fn = jax.jit(energy_grad)
        self.evaluations = 0

    def __call__(self, x: F64Array) -> tuple[float, F64Array]:
        import jax.numpy as jnp

        ei, g = self._fn(jnp.asarray(x, dtype=jnp.float32), self._exp_t, self._w_t)
        self.evaluations += 1
        return float(np.asarray(ei, dtype=np.float64).sum()), np.asarray(g, dtype=np.float64)
