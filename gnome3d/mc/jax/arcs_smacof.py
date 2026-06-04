"""Fast arc reconstruction via stress majorization (SMACOF) - a drop-in
replacement for the single-bead arc Monte Carlo, ~100x faster at matching energy.

Arc reconstruction places `a` anchors so their pairwise distances match the
expected-distance matrix `exp` (springs), with optional `1/d` repulsion for
negative `exp` entries, excluded volume, and confinement.  The reference anneals
this with ~200k sequential single-bead MC steps - the slowest component once
smooth runs on GPU.

The spring energy `sum_{i<j} k*((d-e)/e)^2` is *exactly* weighted-MDS stress
(weight `w = k/e^2`), so we solve it with **SMACOF**: the Guttman transform
`X <- V^+ B(X) X`, which decreases stress monotonically.  Pure descent gets stuck
in local minima (the MC's annealing escapes them), so we **basin-hop**: perturb
the best layout and re-SMACOF, keeping improvements.  An analytic-gradient Adam
polish then settles the basin minimum and folds in the non-stress terms (repulsion
/ EV / confinement / asymmetric stretch!=squeeze).  Reaches MC-parity energy.

Runs in float64 on the host (the Laplacian solve is ill-conditioned in f32) and
is threaded per-IB by the existing arcs executor.  This is a DELIBERATE divergence
from the MC trajectory - it minimises the same energy, far faster - so it is
opt-in via `settings.mc_executor_jax_arcs_solver = "smacof"` and should be
ensemble-parity checked before replacing the MC default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from gnome3d import log

if TYPE_CHECKING:
    from gnome3d.settings import Settings
    from gnome3d.types import F32Array, F64Array

LOG = log.get("mc.arcs_smacof")

_SMACOF_ITERS = 60
# Final Adam polish on the full energy: settles the basin minimum that SMACOF's
# fixed iterations leave short, AND folds in the non-stress terms (repulsion / EV /
# confinement / asymmetric stretch!=squeeze).  Run unconditionally - it's what
# takes the result from ~1.2x to MC-parity energy.
_POLISH_ITERS = 80
_POLISH_LR = 0.03
# SMACOF descent gets stuck in local minima (the MC's annealing escapes them).
# Basin hopping recovers MC-parity energy: perturb the best layout by a random
# kick (fraction of the structure scale) and re-run SMACOF, keeping improvements.
# Cheap - the O(n^3) pinv is shared across all hops.
_HOPS = 25
_KICK_FRAC = 0.4


def _dists(pos: F64Array) -> F64Array:
    diff = pos[:, None, :] - pos[None, :, :]
    return np.sqrt((diff * diff).sum(-1) + 1e-300)


def _smacof_setup(exp: F64Array, k_sym: float) -> tuple[F64Array, F64Array, F64Array]:
    """Per-problem SMACOF constants (shared across restarts): spring weights
    `w_ij = k_sym / e_ij^2`, target distances `delta`, and `pinv(V)` of the
    weighted Laplacian `V`.  pinv (NOT ridge-then-invert): V is rank-(n-1) with the
    constant/translation null space, which pinv projects out cleanly."""
    spring = exp >= 1e-6
    np.fill_diagonal(spring, False)
    w = np.where(spring, k_sym / np.maximum(exp * exp, 1e-300), 0.0)
    delta = np.where(spring, exp, 0.0)
    v = -w.copy()
    np.fill_diagonal(v, w.sum(1))
    return w, delta, np.linalg.pinv(v)


def _smacof_run(
    w: F64Array, delta: F64Array, vinv: F64Array, x0: F64Array, iters: int
) -> F64Array:
    """Guttman-transform iterations from start `x0` (n, 3); O(n^2) per iter."""
    x = x0.copy()
    for _ in range(iters):
        d = _dists(x)
        np.fill_diagonal(d, 1.0)
        b = -w * delta / d
        np.fill_diagonal(b, 0.0)
        np.fill_diagonal(b, -b.sum(1))
        x = vinv @ (b @ x)
    return x


def _energy_and_grad(
    pos: F64Array, exp: F64Array, sk: float, qk: float, r0: float, excl_w: float,
    excl_skip: int, cx: float, cy: float, cz: float, R: float, conf_w: float,
) -> tuple[float, F64Array]:
    """Full arc energy + analytic gradient (springs + repulsion + EV + confinement),
    matching `init_arcs_nb` + `init_excl_nb` + `init_confine_nb`."""
    n = pos.shape[0]
    diff = pos[:, None, :] - pos[None, :, :]  # (n,n,3): i - j
    d = np.sqrt((diff * diff).sum(-1) + 1e-300)
    np.fill_diagonal(d, 1.0)
    idx = np.arange(n)

    # springs (i<j) + repulsion (i<j); pair energy with per-pair coeff, gradient via chain rule
    e_safe = np.maximum(exp, 1e-6)
    rel = (d - e_safe) / e_safe
    k = np.where(rel >= 0, sk, qk)
    spring_e = rel * rel * k
    spring_dE_dd = 2.0 * k * rel / e_safe  # d/dd of k*rel^2
    rep_e = 1.0 / np.maximum(d, 1e-10)
    rep_dE_dd = -1.0 / np.maximum(d, 1e-10) ** 2
    is_spring = exp >= 1e-6
    is_rep = exp <= -1e-10
    pair_e = np.where(is_spring, spring_e, np.where(is_rep, rep_e, 0.0))
    pair_dEdd = np.where(is_spring, spring_dE_dd, np.where(is_rep, rep_dE_dd, 0.0))

    triu = np.triu(np.ones((n, n), bool), 1)
    E = float(pair_e[triu].sum())
    # gradient: each i<j pair contributes to both i and j; sum the full symmetric dE/dd
    coeff = np.where(triu | triu.T, pair_dEdd, 0.0) / d  # (n,n)
    grad = (coeff[:, :, None] * diff).sum(1)  # (n,3)

    # excluded volume (double-counted, |i-j|>skip, d<r0)
    if excl_w > 0.0 and r0 > 0.0:
        far = np.abs(idx[:, None] - idx[None, :]) > excl_skip
        rel_ev = np.maximum(0.0, (r0 - d) / r0)
        active = far & (rel_ev > 0)
        E += float(excl_w * (rel_ev[active] ** 2).sum())  # double-counted (full matrix), matches init_excl_nb
        # grad of a double-counted symmetric pair energy: 2 * (per-entry derivative)
        # (the same factor the MC applies as `2 * (loc_curr - loc_prev)`).
        ev_dEdd = np.where(active, -4.0 * excl_w * rel_ev / r0, 0.0)
        grad += ((ev_dEdd / d)[:, :, None] * diff).sum(1)

    # confinement (per bead)
    if conf_w > 0.0:
        cen = np.array([cx, cy, cz])
        rv = pos - cen
        r = np.sqrt((rv * rv).sum(1) + 1e-300)
        out = r > R
        rel_c = (r - R) / max(R, 1e-30)
        E += float(conf_w * (rel_c[out] ** 2).sum())
        gc = np.where(out, 2.0 * conf_w * rel_c / max(R, 1e-30) / r, 0.0)
        grad += gc[:, None] * rv

    return E, grad


def _polish(pos: F64Array, args: tuple[Any, ...], iters: int, lr: float) -> tuple[F64Array, float]:
    """Adam on the full energy from a SMACOF start.  Returns (pos, energy)."""
    x = pos.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2 = 0.9, 0.999
    E = 0.0
    for t in range(1, iters + 1):
        E, g = _energy_and_grad(x, *args)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        x = x - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + 1e-8)
    E, _ = _energy_and_grad(x, *args)
    return x, E


def solve_arcs(
    pos: F32Array, exp_dist: F32Array, step_size: float, settings: Settings, seed: int
) -> tuple[float, F32Array]:
    """SMACOF + basin-hopping + Adam-polish arc reconstruction.

    Drop-in for `mc_arcs_numba`: same signature shape, returns `(energy, pos)` and
    writes the best layout.  ~40-100x faster than the MC at MC-parity energy.
    `seed` makes the basin-hop restarts deterministic; `step_size` is unused (no
    MC step here) but kept for signature compatibility."""
    n = pos.shape[0]
    if n <= 1:
        return 0.0, pos
    exp = exp_dist.astype(np.float64)
    sk = float(settings.spring_stretch_arcs)
    qk = float(settings.spring_squeeze_arcs)
    k_sym = 0.5 * (sk + qk)

    use_excl = bool(settings.use_excluded_volume) and bool(settings.exclusion_apply_to_arcs)
    r0, excl_w = 0.0, 0.0
    if use_excl:
        excl_w = float(settings.exclusion_weight)
        r0 = float(settings.exclusion_radius_arcs)
        if r0 <= 0.0:
            m = exp > 1e-6
            r0 = float(settings.exclusion_auto_factor_arcs) * (float(exp[m].mean()) if m.any() else 1.0)
    excl_skip = int(settings.exclusion_skip_neighbors)

    use_conf = bool(settings.use_confinement) and bool(settings.confinement_apply_to_arcs)
    cx = cy = cz = 0.0
    R, conf_w = 1.0, 0.0
    if use_conf:
        conf_w = float(settings.confinement_weight)
        R = float(settings.confinement_radius_arcs)
        if R <= 0.0:
            m = exp > 1e-6
            avg = float(exp[m].mean()) if m.any() else 1.0
            R = float(settings.confinement_packing_factor_arcs) * avg * (n ** (1.0 / 3.0))

    eargs = (exp, sk, qk, r0, excl_w, excl_skip, cx, cy, cz, R, conf_w)

    rng = np.random.default_rng(seed & 0x7FFFFFFF)
    scale = float(exp[exp > 1e-6].mean()) if (exp > 1e-6).any() else 1.0
    w, delta, vinv = _smacof_setup(exp, k_sym)  # shared O(n^3) work, once

    # Basin hopping: SMACOF from a random init, then repeatedly perturb the best
    # layout and re-SMACOF, keeping improvements - this escapes the local minima a
    # single descent gets stuck in, reaching MC-parity energy.
    best_x = _smacof_run(w, delta, vinv, rng.standard_normal((n, 3)), _SMACOF_ITERS)
    best_e, _ = _energy_and_grad(best_x, *eargs)
    kick = _KICK_FRAC * scale
    for _ in range(_HOPS):
        x = _smacof_run(w, delta, vinv, best_x + rng.standard_normal((n, 3)) * kick, _SMACOF_ITERS)
        e, _ = _energy_and_grad(x, *eargs)
        if e < best_e:
            best_e, best_x = e, x

    best_x, best_e = _polish(best_x, eargs, _POLISH_ITERS, _POLISH_LR * scale)

    pos[:] = best_x.astype(pos.dtype)
    return best_e, pos
