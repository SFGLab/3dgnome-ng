"""JAX vectorized spatial-checkerboard arcs MC (GPU-ready formulation).

Validates in two steps before production wiring:
  STEP 1: the vectorized all-anchor delta EXACTLY matches the production numba
          single-bead delta (per anchor).
  STEP 2: the full jitted checkerboard loop reaches the same energy as the
          (already sequential-validated) numba checkerboard at matched budget.

EV/confine are gated by their WEIGHT (0 = off) instead of a Python `if`, so the whole
thing is traceable/jittable (same trick as the production arcs kernel).
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

# the validated numba checkerboard/terms live next to this file
_HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("cbnb", os.path.join(_HERE, "arcs_checkerboard.py"))
cbnb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cbnb)


def _arc_E(d, e, stretch, squeeze):
    rep = 1.0 / jnp.maximum(d, 1e-10)
    e_safe = jnp.maximum(e, 1e-6)
    rel = (d - e_safe) / e_safe
    k = jnp.where(rel >= 0.0, stretch, squeeze)
    return jnp.where(e < 0.0, rep, jnp.where(e >= 1e-6, rel * rel * k, 0.0))


def all_deltas(pos, move, exp, stretch, squeeze, r0, excl_w, skip, cx, cy, cz, R, conf_w):
    """delta[p] = energy change if ONLY anchor p moves by move[p] (others frozen).
    arcs single-count + excl x2 + confine x1
    EV/confine gated by weight (0=off)."""
    n = pos.shape[0]
    new = pos + move
    idx = jnp.arange(n)
    eye = idx[:, None] == idx[None, :]
    expT = exp.T  # row p, col i -> exp[i,p] (matches _local_arcs_nb's e=exp[i,p])
    d_old = jnp.sqrt(jnp.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=-1))
    d_mov = jnp.sqrt(jnp.sum((new[:, None, :] - pos[None, :, :]) ** 2, axis=-1))
    a_old = jnp.where(eye, 0.0, _arc_E(d_old, expT, stretch, squeeze))
    a_mov = jnp.where(eye, 0.0, _arc_E(d_mov, expT, stretch, squeeze))
    delta = jnp.sum(a_mov - a_old, axis=1)

    far = jnp.logical_and(jnp.abs(idx[:, None] - idx[None, :]) > skip, jnp.logical_not(eye))
    rel_o = jnp.maximum(0.0, (r0 - d_old) / r0)
    rel_m = jnp.maximum(0.0, (r0 - d_mov) / r0)
    e_old = jnp.where(far, excl_w * rel_o * rel_o, 0.0)
    e_mov = jnp.where(far, excl_w * rel_m * rel_m, 0.0)
    delta = delta + 2.0 * jnp.sum(e_mov - e_old, axis=1)

    ctr = jnp.array([cx, cy, cz])
    r_o = jnp.sqrt(jnp.sum((pos - ctr) ** 2, axis=-1))
    r_m = jnp.sqrt(jnp.sum((new - ctr) ** 2, axis=-1))
    c_o = jnp.where(r_o > R, conf_w * ((r_o - R) / R) ** 2, 0.0)
    c_m = jnp.where(r_m > R, conf_w * ((r_m - R) / R) ** 2, 0.0)
    return delta + (c_m - c_o)


def _energy(pos, exp, stretch, squeeze, r0, excl_w, skip, cx, cy, cz, R, conf_w):
    n = pos.shape[0]
    idx = jnp.arange(n)
    eye = idx[:, None] == idx[None, :]
    upper = idx[:, None] < idx[None, :]
    d = jnp.sqrt(jnp.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=-1))
    tot = jnp.sum(jnp.where(upper, _arc_E(d, exp, stretch, squeeze), 0.0))  # single count
    far = jnp.logical_and(jnp.abs(idx[:, None] - idx[None, :]) > skip, jnp.logical_not(eye))
    rel = jnp.maximum(0.0, (r0 - d) / r0)
    tot = tot + jnp.sum(jnp.where(far, excl_w * rel * rel, 0.0))  # double count (matches init_excl_nb)
    ctr = jnp.array([cx, cy, cz])
    r = jnp.sqrt(jnp.sum((pos - ctr) ** 2, axis=-1))
    return tot + jnp.sum(jnp.where(r > R, conf_w * ((r - R) / R) ** 2, 0.0))


@jax.jit
def run_checker(pos, exp, n_sweeps, step, T0, dt, js, jc, stretch, squeeze,
                r0, excl_w, skip, cx, cy, cz, R, conf_w, cell, base_key, recompute_period):
    """Jitted spatial-checkerboard MC (single chain). Returns (pos, final_energy)."""
    n = pos.shape[0]
    args = (r0, excl_w, skip, cx, cy, cz, R, conf_w)
    score0 = _energy(pos, exp, stretch, squeeze, *args)

    def sweep_body(sw, carry):
        pos, score, T = carry
        score = jnp.where(sw % recompute_period == 0, _energy(pos, exp, stretch, squeeze, *args), score)
        cellidx = jnp.floor(pos / cell).astype(jnp.int32)
        color = (cellidx[:, 0] & 1) * 4 + (cellidx[:, 1] & 1) * 2 + (cellidx[:, 2] & 1)
        k_m, k_u = jax.random.split(jax.random.fold_in(base_key, sw + 1))
        move = jax.random.uniform(k_m, (n, 3), minval=-step, maxval=step, dtype=pos.dtype)
        u = jax.random.uniform(k_u, (n,), dtype=pos.dtype)

        def color_body(c, c2):
            pos, score, T = c2
            delta = all_deltas(pos, move, exp, stretch, squeeze, *args)
            can_jump = jnp.logical_and(T > 0.0, score > 0.0)
            expo = jnp.clip(-jc * ((score + delta) / jnp.maximum(score, 1e-30)) / jnp.maximum(T, 1e-30), -80.0, 80.0)
            ok = jnp.logical_or(delta <= 0.0, jnp.logical_and(can_jump, u < js * jnp.exp(expo)))
            do_c = jnp.logical_and(color == c, ok)
            pos = pos + jnp.where(do_c[:, None], move, 0.0)
            score = score + jnp.sum(jnp.where(do_c, delta, 0.0))
            T = T * dt ** jnp.sum(color == c)
            return pos, score, T

        return jax.lax.fori_loop(0, 8, color_body, (pos, score, T))

    pos, _s, _T = jax.lax.fori_loop(0, n_sweeps, sweep_body, (pos, score0, T0))
    return pos, _energy(pos, exp, stretch, squeeze, *args)


@partial(jax.jit, static_argnames=("maxc",))
def run_checker_gather(pos, exp, n_sweeps, step, T0, dt, js, jc, stretch, squeeze,
                       r0, excl_w, skip, cx, cy, cz, R, conf_w, cell, base_key, recompute_period, maxc):
    """COLOR-GATHER checkerboard: per color, gather only that color's <=maxc anchors
    (jnp.nonzero static size) and compute the (maxc, N) delta instead of the full
    (N, N).  ~N/maxc less work than run_checker.  Returns (pos, final_E, max_color_cnt)
    so the caller can verify maxc was big enough (no silent drops)."""
    n = pos.shape[0]
    args = (r0, excl_w, skip, cx, cy, cz, R, conf_w)
    expT = exp.T
    idx_all = jnp.arange(n)
    ctr = jnp.array([cx, cy, cz])
    score0 = _energy(pos, exp, stretch, squeeze, *args)

    def sweep_body(sw, carry):
        pos, score, T, mx = carry
        score = jnp.where(sw % recompute_period == 0, _energy(pos, exp, stretch, squeeze, *args), score)
        cellidx = jnp.floor(pos / cell).astype(jnp.int32)
        color = (cellidx[:, 0] & 1) * 4 + (cellidx[:, 1] & 1) * 2 + (cellidx[:, 2] & 1)
        k_m, k_u = jax.random.split(jax.random.fold_in(base_key, sw + 1))
        move = jax.random.uniform(k_m, (n, 3), minval=-step, maxval=step, dtype=pos.dtype)
        u = jax.random.uniform(k_u, (n,), dtype=pos.dtype)

        def color_body(c, c2):
            pos, score, T, mx = c2
            mask_c = color == c
            count_c = jnp.sum(mask_c)
            idx_c = jnp.nonzero(mask_c, size=maxc, fill_value=0)[0]   # (maxc,) real idx + pad 0
            valid = jnp.arange(maxc) < count_c                        # real slots
            pos_c = pos[idx_c]                                        # (maxc, 3)
            new_c = pos_c + move[idx_c]
            exp_c = expT[idx_c]                                       # (maxc, N) = exp[i, mover]
            self_m = idx_c[:, None] == idx_all[None, :]              # (maxc, N)
            d_old = jnp.sqrt(jnp.sum((pos_c[:, None, :] - pos[None, :, :]) ** 2, axis=-1))
            d_mov = jnp.sqrt(jnp.sum((new_c[:, None, :] - pos[None, :, :]) ** 2, axis=-1))
            a_old = jnp.where(self_m, 0.0, _arc_E(d_old, exp_c, stretch, squeeze))
            a_mov = jnp.where(self_m, 0.0, _arc_E(d_mov, exp_c, stretch, squeeze))
            delta = jnp.sum(a_mov - a_old, axis=1)
            far = jnp.logical_and(jnp.abs(idx_c[:, None] - idx_all[None, :]) > skip, jnp.logical_not(self_m))
            rel_o = jnp.maximum(0.0, (r0 - d_old) / r0)
            rel_m = jnp.maximum(0.0, (r0 - d_mov) / r0)
            delta = delta + 2.0 * jnp.sum(jnp.where(far, excl_w * (rel_m * rel_m - rel_o * rel_o), 0.0), axis=1)
            ro = jnp.sqrt(jnp.sum((pos_c - ctr) ** 2, axis=-1))
            rn = jnp.sqrt(jnp.sum((new_c - ctr) ** 2, axis=-1))
            co = jnp.where(ro > R, conf_w * ((ro - R) / R) ** 2, 0.0)
            cn = jnp.where(rn > R, conf_w * ((rn - R) / R) ** 2, 0.0)
            delta = delta + (cn - co)
            can_jump = jnp.logical_and(T > 0.0, score > 0.0)
            expo = jnp.clip(-jc * ((score + delta) / jnp.maximum(score, 1e-30)) / jnp.maximum(T, 1e-30), -80.0, 80.0)
            ok = jnp.logical_or(delta <= 0.0, jnp.logical_and(can_jump, u[idx_c] < js * jnp.exp(expo)))
            ok = jnp.logical_and(ok, valid)
            pos = pos.at[idx_c].add(jnp.where(ok[:, None], move[idx_c], 0.0))
            score = score + jnp.sum(jnp.where(ok, delta, 0.0))
            T = T * dt ** count_c
            return pos, score, T, jnp.maximum(mx, count_c.astype(jnp.int32))

        return jax.lax.fori_loop(0, 8, color_body, (pos, score, T, mx))

    pos, _s, _T, mx = jax.lax.fori_loop(0, n_sweeps, sweep_body, (pos, score0, T0, jnp.int32(0)))
    return pos, _energy(pos, exp, stretch, squeeze, *args), mx


def _numeric_prm(prm):
    """prm=(use_excl,r0,excl_w,skip,use_conf,cx,cy,cz,R,conf_w) -> traced-safe numerics
    (weight 0 + safe r0/R when a term is off)."""
    use_excl, r0, excl_w, skip, use_conf, cx, cy, cz, R, conf_w = prm
    return (
        float(r0) if use_excl else 1.0, float(excl_w) if use_excl else 0.0, int(skip),
        float(cx), float(cy), float(cz),
        float(R) if use_conf else 1.0, float(conf_w) if use_conf else 0.0,
    )


def main() -> None:
    ibs = pickle.load(open("/tmp/arcs_conv_ibs.pkl", "rb"))
    by_n = {t[1].shape[0]: t for t in ibs}
    s = Settings()
    s.load_ini("data/GM12878/config.ini")
    sk, qk = float(s.spring_stretch_arcs), float(s.spring_squeeze_arcs)
    T0, dt, js, jc = float(s.max_temp), float(s.dt_temp), float(s.jump_scale), float(s.jump_coef)

    print("=== STEP 1: vectorized delta exactness (jax vs numba single-bead) ===")
    print(f"{'N':>6} {'max|jax-numba|':>16} {'rel':>10}")
    for N in (462, 664, 1146):
        pos0, exp, step = by_n[N]
        pos = np.asarray(pos0, np.float64).copy()
        exp = np.asarray(exp, np.float64)
        step = float(step)
        prm = cbnb.arc_params(s, pos, exp)
        move = np.random.default_rng(0).uniform(-step, step, size=(N, 3))
        jd = np.asarray(all_deltas(jnp.asarray(pos), jnp.asarray(move), jnp.asarray(exp), sk, qk, *_numeric_prm(prm)))
        nd = np.array([cbnb._delta(pos, exp, p, move[p, 0], move[p, 1], move[p, 2], sk, qk, *prm) for p in range(N)])
        print(f"{N:>6} {np.abs(jd-nd).max():>16.2e} {np.abs(jd-nd).max()/max(np.abs(nd).max(),1e-12):>10.2e}")

    print("\n=== STEP 2: full-loop energy parity (jax-checker vs numba-checker, matched budget) ===")
    print(f"{'N':>6} {'budget':>12} {'numba_E':>11} {'jax_E':>11} {'jax/numba':>10}")
    for N in (462, 664):
        pos0, exp, step = by_n[N]
        pos = np.asarray(pos0, np.float64)
        exp = np.asarray(exp, np.float64)
        step = float(step)
        prm = cbnb.arc_params(s, pos, exp)
        d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, 1e30)
        cell = 4.0 * float(np.median(d.min(1)))
        budget = 4_000_000
        nsw = budget // N
        pn = pos.copy()
        np.random.seed(0)
        En = cbnb.mc_checker(pn, exp, nsw, step, T0, dt, js, jc, sk, qk, *prm, cell, 50)
        _, Ej = run_checker(jnp.asarray(pos), jnp.asarray(exp), nsw, step, T0, dt, js, jc, sk, qk,
                            *_numeric_prm(prm), cell, jax.random.PRNGKey(0), 50)
        Ej = float(Ej)
        print(f"{N:>6} {budget:>12,} {En:>11.1f} {Ej:>11.1f} {Ej/max(En,1e-9):>10.2f}")
    print("PASS if jax/numba ~ 1.0 (same algorithm, different RNG -> near-equal energy)")

    print("\n=== STEP 3: color-gather parity (gather vs full, same RNG) + max color count ===")
    print(f"{'N':>6} {'maxc':>6} {'full_E':>11} {'gather_E':>11} {'g/full':>8} {'max_cnt':>8} {'overflow?':>9}")
    for N in (462, 664, 1146):
        pos0, exp, step = by_n[N]
        pos = np.asarray(pos0, np.float64)
        exp = np.asarray(exp, np.float64)
        step = float(step)
        prm = cbnb.arc_params(s, pos, exp)
        d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, 1e30)
        cell = 4.0 * float(np.median(d.min(1)))
        nsw = 2_000_000 // N
        num = _numeric_prm(prm)
        _, Ef = run_checker(jnp.asarray(pos), jnp.asarray(exp), nsw, step, T0, dt, js, jc, sk, qk,
                            *num, cell, jax.random.PRNGKey(0), 50)
        maxc = int(N // 4)  # gamble: 8-color parity => ~N/8 per color; N/4 = 2x margin
        _, Eg, mx = run_checker_gather(jnp.asarray(pos), jnp.asarray(exp), nsw, step, T0, dt, js, jc, sk, qk,
                                       *num, cell, jax.random.PRNGKey(0), 50, maxc)
        Ef, Eg, mx = float(Ef), float(Eg), int(mx)
        print(f"{N:>6} {maxc:>6} {Ef:>11.1f} {Eg:>11.1f} {Eg/max(Ef,1e-9):>8.2f} {mx:>8} "
              f"{'OVERFLOW' if mx > maxc else 'ok':>9}")
    print("PASS if g/full ~ 1.0 AND max_cnt <= maxc (no silent drops).")


if __name__ == "__main__":
    main()
