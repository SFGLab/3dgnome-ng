"""Lever #2 QUALITY GATE: does spatial-checkerboard arcs MC reach the same final
energy/structure as sequential single-bead MC on REAL large IBs?

Arcs energy = few springs (target ~0.3) + ALL-PAIRS 1/d repulsion (+EV +confine).
Exact coloring is impossible (all-pairs), so we test the APPROXIMATION: update
spatially-separated anchors (8-color 3D parity, cell >= cell_size apart) simultaneously,
treating far-pair repulsion as stale within a color (delta is near-dominated since the
1/d force decays). Both seq and checker reuse the SAME production njit energy terms, so
the comparison is apples-to-apples. Validate on real IBs BEFORE building a GPU kernel.
"""

from __future__ import annotations

import math
import pickle
import time

import numpy as np
from numba import njit

from gnome3d.mc.numba.terms import (
    _local_arcs_nb,
    _local_confine_nb,
    _local_excl_nb,
    init_arcs_nb,
    init_confine_nb,
    init_excl_nb,
)
from gnome3d.settings import Settings

CACHE = "/tmp/arcs_conv_ibs.pkl"

# --- tunables (overridden in main) ---
BUDGET = 15_000_000          # proposals per run (~convergence for N<=664)
FACTORS = (4.0, 6.0)         # cell_size = factor * median nn distance (sweet spot 4-6)
IB_SIZES = (462, 664, 1146)  # real chr1 large IBs to validate (must be in the cache)


@njit(fastmath=True, nogil=True)
def _energy(pos, exp, stretch_k, squeeze_k, use_excl, r0, excl_w, skip, use_conf, cx, cy, cz, R, conf_w):
    e = init_arcs_nb(pos, exp, stretch_k, squeeze_k)
    if use_excl:
        e += init_excl_nb(pos, r0, excl_w, skip)
    if use_conf:
        e += init_confine_nb(pos, cx, cy, cz, R, conf_w)
    return e


@njit(fastmath=True, nogil=True)
def _delta(pos, exp, p, dx, dy, dz, stretch_k, squeeze_k, use_excl, r0, excl_w, skip,
           use_conf, cx, cy, cz, R, conf_w):
    """Local energy change if bead p moves by (dx,dy,dz). Mirrors the production step:
    arcs single-counted, excl x2, confine x1."""
    a0 = _local_arcs_nb(pos, exp, p, stretch_k, squeeze_k)
    e0 = _local_excl_nb(pos, p, r0, excl_w, skip) if use_excl else 0.0
    c0 = _local_confine_nb(pos, p, cx, cy, cz, R, conf_w) if use_conf else 0.0
    pos[p, 0] += dx
    pos[p, 1] += dy
    pos[p, 2] += dz
    a1 = _local_arcs_nb(pos, exp, p, stretch_k, squeeze_k)
    e1 = _local_excl_nb(pos, p, r0, excl_w, skip) if use_excl else 0.0
    c1 = _local_confine_nb(pos, p, cx, cy, cz, R, conf_w) if use_conf else 0.0
    pos[p, 0] -= dx
    pos[p, 1] -= dy
    pos[p, 2] -= dz
    return (a1 - a0) + 2.0 * (e1 - e0) + (c1 - c0)


@njit(fastmath=True, nogil=True)
def mc_seq(pos, exp, n_steps, step, T0, dt, js, jc, stretch_k, squeeze_k,
           use_excl, r0, excl_w, skip, use_conf, cx, cy, cz, R, conf_w):
    """Reference single-bead MC (matches production terms/acceptance/schedule)."""
    n = pos.shape[0]
    score = _energy(pos, exp, stretch_k, squeeze_k, use_excl, r0, excl_w, skip,
                    use_conf, cx, cy, cz, R, conf_w)
    T = T0
    for _ in range(n_steps):
        p = np.random.randint(0, n)
        dx = np.random.uniform(-step, step)
        dy = np.random.uniform(-step, step)
        dz = np.random.uniform(-step, step)
        d = _delta(pos, exp, p, dx, dy, dz, stretch_k, squeeze_k, use_excl, r0, excl_w, skip,
                   use_conf, cx, cy, cz, R, conf_w)
        ok = d <= 0.0
        if (not ok) and T > 0.0 and score > 0.0:
            ok = np.random.random() < js * math.exp(-jc * ((score + d) / score) / T)
        if ok:
            pos[p, 0] += dx
            pos[p, 1] += dy
            pos[p, 2] += dz
            score += d
        T *= dt
    return score


@njit(fastmath=True, nogil=True)
def mc_checker(pos, exp, n_sweeps, step, T0, dt, js, jc, stretch_k, squeeze_k,
               use_excl, r0, excl_w, skip, use_conf, cx, cy, cz, R, conf_w,
               cell, recompute_period):
    """Spatial-checkerboard MC: 8-color 3D parity, one proposal per anchor per sweep.
    Within a color, deltas are computed against the SAME base positions (approx:
    same-color cross-terms ignored — small for far-apart cells). T decays per proposal."""
    n = pos.shape[0]
    score = _energy(pos, exp, stretch_k, squeeze_k, use_excl, r0, excl_w, skip,
                    use_conf, cx, cy, cz, R, conf_w)
    T = T0
    color = np.empty(n, np.int64)
    adx = np.zeros(n)
    ady = np.zeros(n)
    adz = np.zeros(n)
    acc = np.zeros(n, np.bool_)
    for sweep in range(n_sweeps):
        if recompute_period > 0 and sweep % recompute_period == 0:
            score = _energy(pos, exp, stretch_k, squeeze_k, use_excl, r0, excl_w, skip,
                            use_conf, cx, cy, cz, R, conf_w)
        # recolor by spatial parity (current positions)
        for p in range(n):
            ix = int(math.floor(pos[p, 0] / cell))
            iy = int(math.floor(pos[p, 1] / cell))
            iz = int(math.floor(pos[p, 2] / cell))
            color[p] = (ix & 1) * 4 + (iy & 1) * 2 + (iz & 1)
        for c in range(8):
            sum_delta = 0.0
            for p in range(n):
                if color[p] != c:
                    continue
                dx = np.random.uniform(-step, step)
                dy = np.random.uniform(-step, step)
                dz = np.random.uniform(-step, step)
                d = _delta(pos, exp, p, dx, dy, dz, stretch_k, squeeze_k, use_excl, r0, excl_w, skip,
                           use_conf, cx, cy, cz, R, conf_w)
                ok = d <= 0.0
                if (not ok) and T > 0.0 and score > 0.0:
                    ok = np.random.random() < js * math.exp(-jc * ((score + d) / score) / T)
                acc[p] = ok
                if ok:
                    adx[p] = dx
                    ady[p] = dy
                    adz[p] = dz
                    sum_delta += d
                T *= dt
            for p in range(n):  # apply accepted moves of color c simultaneously
                if color[p] == c and acc[p]:
                    pos[p, 0] += adx[p]
                    pos[p, 1] += ady[p]
                    pos[p, 2] += adz[p]
            score += sum_delta
    return _energy(pos, exp, stretch_k, squeeze_k, use_excl, r0, excl_w, skip,
                   use_conf, cx, cy, cz, R, conf_w)


def dmap_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of upper-tri pairwise distances (scale/rotation invariant)."""
    iu = np.triu_indices(a.shape[0], 1)
    da = np.sqrt(((a[:, None, :] - a[None, :, :]) ** 2).sum(-1))[iu]
    db = np.sqrt(((b[:, None, :] - b[None, :, :]) ** 2).sum(-1))[iu]
    return float(np.corrcoef(da, db)[0, 1])


def arc_params(s: Settings, pos: np.ndarray, exp: np.ndarray):
    """Resolve arcs energy params, mirroring mc_arcs_jax. EV forced ON (user applies it)."""
    use_excl = True  # user applies EV to arcs (NOT the repo default)
    skip = int(s.exclusion_skip_neighbors)
    excl_w = float(s.exclusion_weight)
    r0 = float(s.exclusion_radius_arcs)
    if r0 <= 0.0:
        m = np.asarray(exp) > 1e-6
        r0 = float(s.exclusion_auto_factor_arcs) * float(np.asarray(exp)[m].mean()) if m.any() else 1.0
    use_conf = bool(s.use_confinement) and bool(s.confinement_apply_to_arcs)
    cx = cy = cz = 0.0
    R = 1.0
    conf_w = 0.0
    if use_conf:
        cx, cy, cz = float(pos[:, 0].mean()), float(pos[:, 1].mean()), float(pos[:, 2].mean())
        R = float(s.confinement_radius_arcs)
        if R <= 0.0:
            m = np.asarray(exp) > 1e-6
            ab = float(np.asarray(exp)[m].mean()) if m.any() else 1.0
            R = float(s.confinement_packing_factor_arcs) * ab * (pos.shape[0] ** (1.0 / 3.0))
        conf_w = float(s.confinement_weight)
    return use_excl, r0, excl_w, skip, use_conf, cx, cy, cz, R, conf_w


def main() -> None:
    ibs = pickle.load(open(CACHE, "rb"))
    by_n = {t[1].shape[0]: t for t in ibs}
    s = Settings()
    s.load_ini("data/GM12878/config.ini")
    sk, qk = float(s.spring_stretch_arcs), float(s.spring_squeeze_arcs)
    T0, dt = float(s.max_temp), float(s.dt_temp)
    js, jc = float(s.jump_scale), float(s.jump_coef)
    print(f"schedule: T0={T0} dt={dt} js={js} jc={jc} stretch={sk} squeeze={qk}  budget={BUDGET:,} proposals")
    print(f"{'N':>5} {'mode':>16} {'final_E':>11} {'E/E_seq':>8} {'dmap_corr':>10} {'wall_s':>7}")
    for N in IB_SIZES:
        if N not in by_n:
            print(f"{N}: not captured")
            continue
        pos0, exp, step = by_n[N]
        pos0 = np.asarray(pos0, np.float64)
        exp = np.asarray(exp, np.float64)
        step = float(step)
        prm = arc_params(s, pos0, exp)
        # sequential reference
        p = pos0.copy()
        np.random.seed(0)
        t = time.perf_counter()
        Es = mc_seq(p, exp, BUDGET, step, T0, dt, js, jc, sk, qk, *prm)
        ws = time.perf_counter() - t
        pos_seq = p.copy()
        print(f"{N:>5} {'sequential':>16} {Es:>11.1f} {1.0:>8.2f} {1.0:>10.3f} {ws:>7.0f}", flush=True)
        # median nn distance (for cell sizing) from the seq result
        d = np.sqrt(((pos_seq[:, None, :] - pos_seq[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, 1e30)
        nn = float(np.median(d.min(1)))
        n_sweeps = max(1, BUDGET // N)
        for f in FACTORS:
            cell = f * nn
            p = pos0.copy()
            np.random.seed(0)
            t = time.perf_counter()
            Ec = mc_checker(p, exp, n_sweeps, step, T0, dt, js, jc, sk, qk, *prm, cell, 50)
            wc = time.perf_counter() - t
            print(f"{N:>5} {('checker f='+str(f)):>16} {Ec:>11.1f} {Ec/max(Es,1e-9):>8.2f} {dmap_corr(pos_seq, p):>10.3f} {wc:>7.0f}", flush=True)


if __name__ == "__main__":
    main()
