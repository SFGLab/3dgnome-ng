"""The arcs energy and its gradient on the device, for the solver.

This is the energy of `gnome3d.mc.numba.arcs_solver.arcs_energy_grad`, term for term,
evaluated in row chunks in float32. The rows are summed in float64 on the host. The two agree
to about 1e-6 relative. The 64 bit mode of JAX is process wide and changes the integer draws of
the smooth kernel. It is not used.
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
        The target matrix. `exp[j, i]` is the entry the CPU kernel reads for the pair (i, j).
    args
        The scalar parameters of `arcs_energy_grad` after `exp`, in its order.
    w
        The weight of each arc spring, None for one.
    node_f
        The factor of each anchor for the excluded volume radius of a pair,
        `excl_r0 * (node_f[i] + node_f[j]) / 2`. None scores every pair at `excl_r0`.
    """

    def __init__(
        self,
        exp: F64Array,
        args: tuple[Any, ...],
        w: F32Array | None = None,
        node_f: F64Array | None = None,
    ) -> None:
        import jax

        from gnome3d.mc.jax.util import jax_is_available

        jax_is_available()
        import jax.numpy as jnp

        stretch, squeeze, rep_inv, bg_w = (float(a) for a in args[:4])
        cx, cy, cz, conf_r, conf_w, excl_r0, excl_w = (float(a) for a in args[4:11])
        excl_skip = int(args[11])
        n = int(exp.shape[0])
        self.n = n
        # A row of the transpose is what the CPU kernel reads for one anchor.
        self._exp_t = jnp.asarray(np.ascontiguousarray(exp.T, dtype=np.float32))
        # The loop weights are transposed the same way. None is one for every pair, fixed at trace
        # time.
        use_w = w is not None
        self._w_t = (
            jnp.asarray(np.ascontiguousarray(w.T, dtype=np.float32))
            if w is not None
            else jnp.zeros((1, 1), dtype=jnp.float32)
        )
        idx_all = jnp.arange(n, dtype=jnp.int32)
        use_f = node_f is not None
        f_host: F32Array = np.ascontiguousarray(
            node_f if node_f is not None else np.ones(1), dtype=np.float32
        )
        f_all: Any = jnp.asarray(f_host)  # pyright: ignore[reportUnknownMemberType]
        centre = jnp.asarray(np.array([cx, cy, cz], dtype=np.float32))
        use_excl = excl_w > 0.0

        def row_terms(xi: Any, i: Any, e: Any, wi: Any, pos: Any) -> tuple[Any, Any]:
            """Returns the energy and gradient of anchor i over every other anchor, then its
            confinement.
            """
            diff = xi[None, :] - pos
            d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
            dd = jnp.maximum(d, 1e-10)
            # Background spring at bg = -e, symmetric in log distance.
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
                r0: Any = excl_r0 * 0.5 * (f_all[i] + f_all) if use_f else excl_r0
                ex_on = jnp.logical_and(d < r0, sep > excl_skip)
                u = (r0 - d) / r0
                ener = ener + jnp.where(ex_on, excl_w * u * u, 0.0)
                w = w + jnp.where(ex_on, -4.0 * excl_w * u / (r0 * dd), 0.0)
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


class DeviceFactoryEnergy:
    """The factory term of `factory_energy_grad` on the device, as a callable on flattened
    positions.

    Two passes in row chunks: the group sums of every active anchor, then the energy and gradient
    of each anchor. Float32 on the device, summed in float64 on the host.

    Parameters
    ----------
    act
        The activity of each anchor, zero for an anchor without the term.
    weight, radius
        `factory_weight` and `factory_radius`.
    """

    def __init__(self, act: F64Array, weight: float, radius: float) -> None:
        import jax

        from gnome3d.mc.jax.util import jax_is_available

        jax_is_available()
        import jax.numpy as jnp

        n = int(act.shape[0])
        self.n = n
        act32 = jnp.asarray(np.ascontiguousarray(act, dtype=np.float32))
        w = float(weight)
        r = float(radius)
        log_a = float(np.log1p(float(act.sum())))
        idx_all = jnp.arange(n, dtype=jnp.int32)
        active = act32 > 0.0

        def group_row(xi: Any, i: Any, pos: Any) -> Any:
            diff = xi[None, :] - pos
            d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
            e = act32 * jnp.exp(-d / r)
            e = jnp.where(jnp.logical_and(idx_all != i, active), e, 0.0)
            return jnp.sum(e)

        def energy_grad(x: Any) -> tuple[Any, Any]:
            pos = x.reshape(n, 3)

            def one_group(a: tuple[Any, Any]) -> Any:
                return group_row(a[0], a[1], pos)

            s_all = jax.lax.map(one_group, (pos, idx_all), batch_size=CHUNK_ROWS)
            s_all = jnp.where(active, s_all, 0.0)
            inv = 1.0 / (1.0 + s_all)

            def row(a: tuple[Any, Any, Any, Any, Any]) -> tuple[Any, Any]:
                xi, i, ai, si, inv_i = a
                diff = xi[None, :] - pos
                d = jnp.sqrt(jnp.sum(diff * diff, axis=1))
                dd = jnp.maximum(d, 1e-10)
                coef = w * ai * act32 * jnp.exp(-d / r) / (r * dd) * (inv_i + inv)
                coef = jnp.where(jnp.logical_and(idx_all != i, active), coef, 0.0)
                gi = jnp.sum(coef[:, None] * diff, axis=0)
                ei = w * ai * (log_a - jnp.log1p(si))
                return ei, gi

            ei, gi = jax.lax.map(row, (pos, idx_all, act32, s_all, inv), batch_size=CHUNK_ROWS)
            ei = jnp.where(active, ei, 0.0)
            gi = jnp.where(active[:, None], gi, 0.0)
            return ei, gi.reshape(-1)

        self._fn = jax.jit(energy_grad)

    def __call__(self, x: F64Array) -> tuple[float, F64Array]:
        import jax.numpy as jnp

        ei, g = self._fn(jnp.asarray(x, dtype=jnp.float32))
        return float(np.asarray(ei, dtype=np.float64).sum()), np.asarray(g, dtype=np.float64)
