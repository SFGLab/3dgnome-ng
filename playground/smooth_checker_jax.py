"""JAX vectorized smooth delta — STEP 1: validate it EXACTLY matches the production numba
single-bead smooth delta (structural bonds/angles + dense heat + EV + confine).

The structural term is the risky vectorization: bead p couples to p-2..p+2 with per-bead
boundary masks (the local_smooth_nb bounds, using n_active as the chain length).  heat/EV
are all-pairs (maxc, B) like arcs.  Once exact, the checkerboard MC loop is bookkeeping.
"""

from __future__ import annotations

import importlib.util
import os
import pickle
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)  # match numba float64 for the exactness test

from gnome3d.settings import Settings

_HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("scnb", os.path.join(_HERE, "smooth_checkerboard.py"))
scnb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scnb)


def _len_E(d, e, sk, qk, dw):
    e_safe = jnp.maximum(e, 1e-6)
    rel = (d - e_safe) / e_safe
    k = jnp.where(rel >= 0.0, sk, qk)
    return rel * rel * k * dw


def _ang_E(v1, v2, ak, aw):
    n1 = jnp.sqrt(jnp.sum(v1 * v1, axis=-1))
    n2 = jnp.sqrt(jnp.sum(v2 * v2, axis=-1))
    cos_a = jnp.clip(jnp.sum(v1 * v2, axis=-1) / jnp.maximum(n1 * n2, 1e-30), -1.0, 1.0)
    ang = 1.0 - (cos_a + 1.0) * 0.5
    e = ang * ang * ang * ak * aw  # numba uses ang^3
    return jnp.where(jnp.logical_or(n1 < 1e-12, n2 < 1e-12), 0.0, e)


def struct_delta(pos, new, idx, dtn, sk, qk, ak, dw, aw, n_active):
    """Structural (bond+angle) delta for moving each bead idx[j] to new[idx[j]] (others fixed).
    Mirrors local_smooth_nb's bounds with n=n_active."""
    B = pos.shape[0]

    def g(off):
        return pos[jnp.clip(idx + off, 0, B - 1)]

    pm2, pm1, p0, pp1, pp2 = g(-2), g(-1), g(0), g(1), g(2)
    newp = new[idx]
    dL = dtn[jnp.clip(idx - 1, 0, B - 1)]
    dR = dtn[jnp.clip(idx, 0, B - 1)]
    nrm = lambda a: jnp.sqrt(jnp.sum(a * a, axis=-1))

    # bonds: (p-1,p) uses dtn[p-1]; (p,p+1) uses dtn[p]
    bL = _len_E(nrm(pm1 - newp), dL, sk, qk, dw) - _len_E(nrm(pm1 - p0), dL, sk, qk, dw)
    bR = _len_E(nrm(newp - pp1), dR, sk, qk, dw) - _len_E(nrm(p0 - pp1), dR, sk, qk, dw)
    validL = jnp.logical_and(idx >= 1, idx < n_active)        # bond at p-1: 1<=p<n_active
    validR = idx < n_active - 1                                # bond at p:   p<n_active-1
    bond_d = jnp.where(validL, bL, 0.0) + jnp.where(validR, bR, 0.0)

    # angles at p-2 (c=p), p-1 (b=p), p (a=p)
    a2 = _ang_E(pm2 - pm1, pm1 - newp, ak, aw) - _ang_E(pm2 - pm1, pm1 - p0, ak, aw)
    a1 = _ang_E(pm1 - newp, newp - pp1, ak, aw) - _ang_E(pm1 - p0, p0 - pp1, ak, aw)
    a0 = _ang_E(newp - pp1, pp1 - pp2, ak, aw) - _ang_E(p0 - pp1, pp1 - pp2, ak, aw)
    v2 = jnp.logical_and(idx >= 2, idx < n_active)             # ang at p-2: 2<=p<n_active
    v1 = jnp.logical_and(idx >= 1, idx < n_active - 1)         # ang at p-1: 1<=p<n_active-1
    v0 = idx < n_active - 2                                     # ang at p:   p<n_active-2
    ang_d = jnp.where(v2, a2, 0.0) + jnp.where(v1, a1, 0.0) + jnp.where(v0, a0, 0.0)
    return bond_d + ang_d


def heat_ev_conf_delta(pos, new, idx, heat, hw, r0, ew, skip, cx, cy, cz, R, cw, n_active):
    """heat (x2, dense all-pairs springs to heat_dist) + EV (x2) + confine (x1) delta."""
    B = pos.shape[0]
    idx_all = jnp.arange(B)
    active = idx_all < n_active
    self_m = idx[:, None] == idx_all[None, :]
    pos_c, new_c = pos[idx], new[idx]
    d_old = jnp.sqrt(jnp.sum((pos_c[:, None, :] - pos[None, :, :]) ** 2, axis=-1))
    d_mov = jnp.sqrt(jnp.sum((new_c[:, None, :] - pos[None, :, :]) ** 2, axis=-1))

    h_c = heat.T[idx]  # heat[i, p] for mover p, partner i
    hmask = jnp.logical_and(jnp.logical_and(h_c > 1e-6, jnp.logical_not(self_m)), active[None, :])
    h_safe = jnp.maximum(h_c, 1e-6)
    rel_o = (d_old - h_safe) / h_safe
    rel_m = (d_mov - h_safe) / h_safe
    heat_d = 2.0 * hw * jnp.sum(jnp.where(hmask, rel_m * rel_m - rel_o * rel_o, 0.0), axis=1)

    far = jnp.logical_and(jnp.abs(idx[:, None] - idx_all[None, :]) > skip, jnp.logical_not(self_m))
    far = jnp.logical_and(far, active[None, :])
    relo = jnp.maximum(0.0, (r0 - d_old) / r0)
    relm = jnp.maximum(0.0, (r0 - d_mov) / r0)
    ev_d = 2.0 * jnp.sum(jnp.where(far, ew * (relm * relm - relo * relo), 0.0), axis=1)

    ctr = jnp.array([cx, cy, cz])
    ro = jnp.sqrt(jnp.sum((pos_c - ctr) ** 2, axis=-1))
    rn = jnp.sqrt(jnp.sum((new_c - ctr) ** 2, axis=-1))
    co = jnp.where(ro > R, cw * ((ro - R) / R) ** 2, 0.0)
    cn = jnp.where(rn > R, cw * ((rn - R) / R) ** 2, 0.0)
    return heat_d + ev_d + (cn - co)


def smooth_deltas(pos, move, idx, dtn, heat, sk, qk, ak, dw, aw, hw, r0, ew, skip,
                  cx, cy, cz, R, cw, n_active):
    new = pos + move
    return (struct_delta(pos, new, idx, dtn, sk, qk, ak, dw, aw, n_active)
            + heat_ev_conf_delta(pos, new, idx, heat, hw, r0, ew, skip, cx, cy, cz, R, cw, n_active))


def _energy_total(pos, dtn, heat, sk, qk, ak, dw, aw, hw, r0, ew, skip, cx, cy, cz, R, cw, n_active):
    n = pos.shape[0]
    idx = jnp.arange(n)
    active = idx < n_active
    eye = idx[:, None] == idx[None, :]
    d = jnp.sqrt(jnp.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=-1))
    diff = pos[:-1] - pos[1:]                                  # diff[i]=pos[i]-pos[i+1]
    bonds = _len_E(jnp.sqrt(jnp.sum(diff * diff, axis=-1)), dtn[: n - 1], sk, qk, dw)
    tot = jnp.sum(jnp.where(jnp.arange(n - 1) < n_active - 1, bonds, 0.0))
    angs = _ang_E(diff[:-1], diff[1:], ak, aw)
    tot = tot + jnp.sum(jnp.where(jnp.arange(n - 2) < n_active - 2, angs, 0.0))
    hmask = jnp.logical_and(jnp.logical_and(heat > 1e-6, jnp.logical_not(eye)), active[:, None] & active[None, :])
    rel = (d - jnp.maximum(heat, 1e-6)) / jnp.maximum(heat, 1e-6)
    tot = tot + hw * jnp.sum(jnp.where(hmask, rel * rel, 0.0))  # heat double-counted
    far = jnp.logical_and(jnp.abs(idx[:, None] - idx[None, :]) > skip, jnp.logical_not(eye))
    far = jnp.logical_and(far, active[:, None] & active[None, :])
    rl = jnp.maximum(0.0, (r0 - d) / r0)
    tot = tot + jnp.sum(jnp.where(far, ew * rl * rl, 0.0))      # EV double-counted
    ctr = jnp.array([cx, cy, cz])
    rr = jnp.sqrt(jnp.sum((pos - ctr) ** 2, axis=-1))
    return tot + jnp.sum(jnp.where(jnp.logical_and(active, rr > R), cw * ((rr - R) / R) ** 2, 0.0))


@partial(jax.jit, static_argnames=("maxc",))
def run_smooth_checker(pos, dtn, heat, movable, n_sweeps, step, T0, dt, js, jc,
                       sk, qk, ak, dw, aw, hw, r0, ew, skip, cx, cy, cz, R, cw, cell, base_key, maxc):
    """24-color (chain-mod-3 x spatial-8) checkerboard smooth, single chain.  STRICT accept."""
    n = pos.shape[0]
    n_active = n
    eargs = (sk, qk, ak, dw, aw, hw, r0, ew, skip, cx, cy, cz, R, cw, n_active)
    chainc = (jnp.arange(n) % 3) * 8
    score0 = _energy_total(pos, dtn, heat, *eargs)

    def sweep_body(_sw, carry):
        pos, score, T = carry
        ci = jnp.floor(pos / cell).astype(jnp.int32)
        color = chainc + (ci[:, 0] & 1) * 4 + (ci[:, 1] & 1) * 2 + (ci[:, 2] & 1)
        k_m, k_u = jax.random.split(jax.random.fold_in(base_key, _sw + 1))
        move = jax.random.uniform(k_m, (n, 3), minval=-step, maxval=step, dtype=pos.dtype)
        u = jax.random.uniform(k_u, (n,), dtype=pos.dtype)

        def color_body(c, c2):
            pos, score, T = c2
            mask_c = jnp.logical_and(color == c, movable)
            count_c = jnp.sum(mask_c)
            idx_c = jnp.nonzero(mask_c, size=maxc, fill_value=0)[0]
            valid = jnp.arange(maxc) < count_c
            delta = smooth_deltas(pos, move, idx_c, dtn, heat, sk, qk, ak, dw, aw, hw,
                                  r0, ew, skip, cx, cy, cz, R, cw, n_active)
            can_jump = jnp.logical_and(T > 0.0, score > 0.0)
            expo = jnp.clip(-jc * ((score + delta) / jnp.maximum(score, 1e-30)) / jnp.maximum(T, 1e-30), -80.0, 80.0)
            ok = jnp.logical_or(delta < 0.0, jnp.logical_and(can_jump, u[idx_c] < js * jnp.exp(expo)))  # STRICT
            ok = jnp.logical_and(ok, valid)
            pos = pos.at[idx_c].add(jnp.where(ok[:, None], move[idx_c], 0.0))
            score = score + jnp.sum(jnp.where(ok, delta, 0.0))
            return pos, score, T * dt ** count_c

        return jax.lax.fori_loop(0, 24, color_body, (pos, score, T))

    pos, _s, _T = jax.lax.fori_loop(0, n_sweeps, sweep_body, (pos, score0, T0))
    return pos, _energy_total(pos, dtn, heat, *eargs)


def main() -> None:
    caps = pickle.load(open("/tmp/smooth_ibs.pkl", "rb"))
    wh = [c for c in caps if c["heat"] is not None]
    wh.sort(key=lambda c: c["pos"].shape[0])
    tests = [c for c in wh if 200 <= c["pos"].shape[0] <= 600][:3]
    s = Settings(); s.load_ini("data/GM12878/config.ini")
    T0s, dts = float(s.max_temp_smooth), float(s.dt_temp_smooth)
    jss, jcs = float(s.jump_scale_smooth), float(s.jump_coef_smooth)
    print("STEP 1: jax vectorized smooth delta vs numba single-bead delta (all movable beads)")
    print(f"{'B':>6} {'max|jax-numba|':>16} {'rel':>10}")
    for c in tests:
        pos = np.asarray(c["pos"], np.float64)
        dtn = np.asarray(c["dtn"], np.float64)
        fixed = np.asarray(c["fixed"], np.bool_)
        heat = np.asarray(c["heat"], np.float64)
        B = pos.shape[0]
        sk, qk, ak, dw, aw, hw, r0, ew, skip, cx, cy, cz, R, cw, use_excl, use_conf = scnb.smooth_params(s, pos, dtn)
        step = float(np.median(dtn[dtn > 1e-6])) if (dtn > 1e-6).any() else 1.0
        movable = np.where(~fixed)[0]
        move = np.zeros((B, 3)); move[movable] = np.random.default_rng(0).uniform(-step, step, size=(len(movable), 3))
        idx = jnp.asarray(movable)
        jd = np.asarray(smooth_deltas(jnp.asarray(pos), jnp.asarray(move), idx, jnp.asarray(dtn),
                                      jnp.asarray(heat), sk, qk, ak, dw, aw, hw, r0, ew, skip,
                                      cx, cy, cz, R, cw, B))
        nd = np.array([scnb._delta(pos, dtn, heat, int(p), B, move[p, 0], move[p, 1], move[p, 2],
                                   sk, qk, ak, dw, aw, hw, r0, ew, skip, cx, cy, cz, R, cw,
                                   True, use_excl, use_conf) for p in movable])
        print(f"{B:>6} {np.abs(jd-nd).max():>16.2e} {np.abs(jd-nd).max()/max(np.abs(nd).max(),1e-12):>10.2e}")
    print("PASS if max|jax-numba| ~ 1e-9 (float64 roundoff)")

    print("\nSTEP 2: full-loop energy parity (jax-checker vs numba-checker, matched budget)")
    print(f"{'B':>6} {'numba_E':>11} {'jax_E':>11} {'jax/numba':>10}")
    for c in tests:
        pos = np.asarray(c["pos"], np.float64)
        dtn = np.asarray(c["dtn"], np.float64)
        fixed = np.asarray(c["fixed"], np.bool_)
        heat = np.asarray(c["heat"], np.float64)
        B = pos.shape[0]
        sk, qk, ak, dw, aw, hw, r0, ew, skip, cx, cy, cz, R, cw, use_excl, use_conf = scnb.smooth_params(s, pos, dtn)
        step = float(np.median(dtn[dtn > 1e-6])) if (dtn > 1e-6).any() else 1.0
        movable = np.where(~fixed)[0].astype(np.int64)
        d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)); np.fill_diagonal(d, 1e30)
        cell = 4.0 * float(np.median(d.min(1)))
        budget = 3_000_000
        nsw = budget // len(movable)
        pn = pos.copy(); np.random.seed(0)
        En = scnb.mc_smooth_checker(pn, dtn, fixed, heat, movable, nsw, step, T0s, dts, jss, jcs,
                                    sk, qk, ak, dw, aw, hw, r0, ew, skip, cx, cy, cz, R, cw,
                                    True, use_excl, use_conf, cell)
        _, Ej = run_smooth_checker(jnp.asarray(pos), jnp.asarray(dtn), jnp.asarray(heat),
                                   jnp.asarray(~fixed), nsw, step, T0s, dts, jss, jcs,
                                   sk, qk, ak, dw, aw, hw, r0, ew, skip, cx, cy, cz, R, cw,
                                   cell, jax.random.PRNGKey(0), max(8, B // 8))
        Ej = float(Ej)
        print(f"{B:>6} {En:>11.2f} {Ej:>11.2f} {Ej/max(En,1e-9):>10.3f}")
    print("PASS if jax/numba ~ 1.0 (same algorithm, different RNG)")


if __name__ == "__main__":
    main()
