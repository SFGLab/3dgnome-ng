"""Smooth checkerboard QUALITY GATE (lever #2 ported to smooth).

Smooth energy = chain bonds + angles (LOCAL, span ±2) + heat (all-pairs dense springs to
heat_dist) + EV (all-pairs spatial) + confine.  Fixed anchor beads never move.

Coloring: `color = (p mod 3) * 8 + spatial-parity(pos[p])` = 24 colors. chain-mod-3 makes
the structural (bond/angle) delta EXACT — a color-c bead's structural neighbours (p±2) are
always colours c+1/c+2, never moving.  spatial-8 handles the all-pairs EV + heat the same
way arcs does (far-pair stale, near-dominated).  Reuses the production numba smooth terms
so seq vs checker is exact.  Run on real captured smooth IBs (/tmp/smooth_ibs.pkl).
"""

from __future__ import annotations

import math
import pickle
import time

import numpy as np
from numba import njit

from gnome3d.mc.numba.terms import (
    _local_confine_nb,
    _local_excl_nb,
    init_confine_nb,
    init_excl_nb,
    init_heat_nb,
    init_smooth_nb,
    local_heat_nb,
    local_smooth_nb,
)
from gnome3d.settings import Settings

CACHE = "/tmp/smooth_ibs.pkl"


@njit(fastmath=True, nogil=True)
def _energy(pos, dtn, heat, sk, qk, ak, dw, aw, hw, r0, ew, skip, cx, cy, cz, R, cw,
            use_heat, use_excl, use_conf):
    e = init_smooth_nb(pos, dtn, sk, qk, ak, dw, aw)
    if use_heat:
        e += init_heat_nb(pos, heat, hw)
    if use_excl:
        e += init_excl_nb(pos, r0, ew, skip)
    if use_conf:
        e += init_confine_nb(pos, cx, cy, cz, R, cw)
    return e


@njit(fastmath=True, nogil=True)
def _delta(pos, dtn, heat, p, n, dx, dy, dz, sk, qk, ak, dw, aw, hw, r0, ew, skip,
           cx, cy, cz, R, cw, use_heat, use_excl, use_conf):
    """Energy change if movable bead p moves by (dx,dy,dz).  struct x1 + heat x2 + excl x2
    + confine x1 (matches the production smooth step)."""
    s0 = local_smooth_nb(pos, dtn, p, n, sk, qk, ak, dw, aw)
    h0 = local_heat_nb(pos, heat, p, hw) if use_heat else 0.0
    e0 = _local_excl_nb(pos, p, r0, ew, skip) if use_excl else 0.0
    c0 = _local_confine_nb(pos, p, cx, cy, cz, R, cw) if use_conf else 0.0
    pos[p, 0] += dx; pos[p, 1] += dy; pos[p, 2] += dz
    s1 = local_smooth_nb(pos, dtn, p, n, sk, qk, ak, dw, aw)
    h1 = local_heat_nb(pos, heat, p, hw) if use_heat else 0.0
    e1 = _local_excl_nb(pos, p, r0, ew, skip) if use_excl else 0.0
    c1 = _local_confine_nb(pos, p, cx, cy, cz, R, cw) if use_conf else 0.0
    pos[p, 0] -= dx; pos[p, 1] -= dy; pos[p, 2] -= dz
    return (s1 - s0) + 2.0 * (h1 - h0) + 2.0 * (e1 - e0) + (c1 - c0)


@njit(fastmath=True, nogil=True)
def mc_smooth_seq(pos, dtn, fixed, heat, movable, n_steps, step, T0, dt, js, jc,
                  sk, qk, ak, dw, aw, hw, r0, ew, skip, cx, cy, cz, R, cw,
                  use_heat, use_excl, use_conf):
    """Reference single-bead smooth MC (STRICT acceptance, matches production)."""
    n = pos.shape[0]
    nm = movable.shape[0]
    score = _energy(pos, dtn, heat, sk, qk, ak, dw, aw, hw, r0, ew, skip, cx, cy, cz, R, cw,
                    use_heat, use_excl, use_conf)
    T = T0
    for _ in range(n_steps):
        p = movable[np.random.randint(0, nm)]
        dx = np.random.uniform(-step, step); dy = np.random.uniform(-step, step); dz = np.random.uniform(-step, step)
        d = _delta(pos, dtn, heat, p, n, dx, dy, dz, sk, qk, ak, dw, aw, hw, r0, ew, skip,
                   cx, cy, cz, R, cw, use_heat, use_excl, use_conf)
        ok = d < 0.0  # smooth is STRICT
        if (not ok) and T > 0.0 and score > 0.0:
            ok = np.random.random() < js * math.exp(-jc * ((score + d) / score) / T)
        if ok:
            pos[p, 0] += dx; pos[p, 1] += dy; pos[p, 2] += dz
            score += d
        T *= dt
    return score


@njit(fastmath=True, nogil=True)
def mc_smooth_checker(pos, dtn, fixed, heat, movable, n_sweeps, step, T0, dt, js, jc,
                      sk, qk, ak, dw, aw, hw, r0, ew, skip, cx, cy, cz, R, cw,
                      use_heat, use_excl, use_conf, cell):
    """24-color (chain-mod-3 x spatial-8) checkerboard smooth.  One proposal per movable
    bead per sweep; same-color beads updated against the same base positions."""
    n = pos.shape[0]
    score = _energy(pos, dtn, heat, sk, qk, ak, dw, aw, hw, r0, ew, skip, cx, cy, cz, R, cw,
                    use_heat, use_excl, use_conf)
    T = T0
    color = np.full(n, -1, np.int64)
    adx = np.zeros(n); ady = np.zeros(n); adz = np.zeros(n)
    acc = np.zeros(n, np.bool_)
    for _sw in range(n_sweeps):
        for p in range(n):
            if fixed[p]:
                continue
            ix = int(math.floor(pos[p, 0] / cell)); iy = int(math.floor(pos[p, 1] / cell)); iz = int(math.floor(pos[p, 2] / cell))
            spatial = (ix & 1) * 4 + (iy & 1) * 2 + (iz & 1)
            color[p] = (p % 3) * 8 + spatial
        for c in range(24):
            sum_d = 0.0
            for p in range(n):
                if color[p] != c:
                    continue
                dx = np.random.uniform(-step, step); dy = np.random.uniform(-step, step); dz = np.random.uniform(-step, step)
                d = _delta(pos, dtn, heat, p, n, dx, dy, dz, sk, qk, ak, dw, aw, hw, r0, ew, skip,
                           cx, cy, cz, R, cw, use_heat, use_excl, use_conf)
                ok = d < 0.0
                if (not ok) and T > 0.0 and score > 0.0:
                    ok = np.random.random() < js * math.exp(-jc * ((score + d) / score) / T)
                acc[p] = ok
                if ok:
                    adx[p] = dx; ady[p] = dy; adz[p] = dz
                    sum_d += d
                T *= dt
            for p in range(n):
                if color[p] == c and acc[p]:
                    pos[p, 0] += adx[p]; pos[p, 1] += ady[p]; pos[p, 2] += adz[p]
            score += sum_d
    return _energy(pos, dtn, heat, sk, qk, ak, dw, aw, hw, r0, ew, skip, cx, cy, cz, R, cw,
                   use_heat, use_excl, use_conf)


def smooth_params(s: Settings, pos, dtn):
    use_excl = bool(s.use_excluded_volume) and bool(s.exclusion_apply_to_smooth)
    skip = int(s.exclusion_skip_neighbors)
    ew = float(s.exclusion_weight)
    r0 = float(s.exclusion_radius_smooth)
    if use_excl and r0 <= 0.0:
        m = np.asarray(dtn) > 1e-6
        r0 = float(s.exclusion_auto_factor_smooth) * float(np.asarray(dtn)[m].mean()) if m.any() else 1.0
    r0 = r0 if r0 > 0 else 1.0
    use_conf = bool(s.use_confinement) and bool(s.confinement_apply_to_smooth)
    cx = cy = cz = 0.0; R = 1.0; cw = 0.0
    if use_conf:
        cx, cy, cz = float(pos[:, 0].mean()), float(pos[:, 1].mean()), float(pos[:, 2].mean())
        R = float(s.confinement_radius_smooth)
        if R <= 0.0:
            m = np.asarray(dtn) > 1e-6
            ab = float(np.asarray(dtn)[m].mean()) if m.any() else 1.0
            R = float(s.confinement_packing_factor_smooth) * ab * (pos.shape[0] ** (1.0 / 3.0))
        cw = float(s.confinement_weight)
    return (float(s.spring_stretch), float(s.spring_squeeze), float(s.spring_angular),
            float(s.smooth_dist_weight), float(s.smooth_angle_weight), float(s.subanchor_heatmap_dist_weight),
            r0, ew, skip, cx, cy, cz, R, cw, use_excl, use_conf)


def main() -> None:
    caps = pickle.load(open(CACHE, "rb"))
    # test on medium IBs WITH heat (fast on CPU; the large ones are GPU-only)
    withheat = [c for c in caps if c["heat"] is not None]
    withheat.sort(key=lambda c: c["pos"].shape[0])
    tests = [c for c in withheat if 200 <= c["pos"].shape[0] <= 1500][:3]
    s = Settings(); s.load_ini("data/GM12878/config.ini")
    T0, dt = float(s.max_temp_smooth), float(s.dt_temp_smooth)
    js, jc = float(s.jump_scale_smooth), float(s.jump_coef_smooth)
    BUDGET = 8_000_000
    print(f"smooth: T0={T0} dt={dt}  EV+confine per config; heat dense.  budget={BUDGET:,}")
    print(f"{'B':>6} {'fixed%':>7} {'seq_E':>11} {'chk_E':>11} {'chk/seq':>8} {'maxcolor':>9}")
    for c in tests:
        pos0 = np.asarray(c["pos"], np.float64)
        dtn = np.asarray(c["dtn"], np.float64) if c.get("dtn") is not None else np.zeros(pos0.shape[0])
        fixed = np.asarray(c["fixed"], np.bool_)
        heat = np.asarray(c["heat"], np.float64)
        B = pos0.shape[0]
        movable = np.where(~fixed)[0].astype(np.int64)
        sk, qk, ak, dw, aw, hw, r0, ew, skip, cx, cy, cz, R, cw, use_excl, use_conf = smooth_params(s, pos0, dtn)
        use_heat = True
        # step_size wasn't captured; use the bond scale (median dtn) — same for seq & checker
        step = float(np.median(dtn[dtn > 1e-6])) if (dtn > 1e-6).any() else 1.0
        # cell = 4 * median nn over movable
        d = np.sqrt(((pos0[:, None, :] - pos0[None, :, :]) ** 2).sum(-1)); np.fill_diagonal(d, 1e30)
        cell = 4.0 * float(np.median(d.min(1)))
        nsw = BUDGET // max(len(movable), 1)
        # sequential
        p = pos0.copy(); np.random.seed(0)
        Es = mc_smooth_seq(p, dtn, fixed, heat, movable, BUDGET, step, T0, dt, js, jc,
                           sk, qk, ak, dw, aw, hw, r0, ew, skip, cx, cy, cz, R, cw, use_heat, use_excl, use_conf)
        # checker
        p = pos0.copy(); np.random.seed(0)
        Ec = mc_smooth_checker(p, dtn, fixed, heat, movable, nsw, step, T0, dt, js, jc,
                               sk, qk, ak, dw, aw, hw, r0, ew, skip, cx, cy, cz, R, cw, use_heat, use_excl, use_conf, cell)
        # max color occupancy (sanity for the 24-color balance)
        print(f"{B:>6} {100*fixed.sum()/B:>6.1f}% {Es:>11.1f} {Ec:>11.1f} {Ec/max(Es,1e-9):>8.3f} {'-':>9}", flush=True)
    print("PASS if chk/seq ~ 1.0 (24-color checkerboard smooth matches sequential energy)")


if __name__ == "__main__":
    main()
