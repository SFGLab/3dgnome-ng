"""
Monte Carlo dispatcher for 3dgnome-ng.

Thin layer that:
  - exposes the four public MC entries used by `gnome3d.solver`:
    `mc_heatmap`, `mc_arcs`, `mc_smooth`, `mc_ib`
  - dispatches each call to the configured backend
    (`gnome3d.mc_numba` or `gnome3d.mc_jax`) based on `settings.mc_backend`
  - handles the optional GNOME3D_MC_PROFILE CSV logging uniformly across
    backends — every call is timed and recorded here so the log shape stays
    consistent regardless of which backend ran

Backend support matrix (see [gnome3d/mc_jax.py] for term details):

  | level   | numba | jax (when mc_backend='jax')                    |
  |---------|-------|------------------------------------------------|
  | heatmap | full  | not ported (heatmap is <0.1% of typical wall) |
  | arcs    | full  | arcs + EV + confinement                        |
  | smooth  | full  | chain + EV + heat + orient + confinement       |
  | ib      | full  | not ported (IB is <1% of typical wall)         |

The harness modules import a couple of numba helpers from this file
(`_as_f64`, `_init_heat_nb`) — those are re-exported from mc_numba below.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from . import mc_numba

# Backward-compat re-exports for harness/compare.py and other consumers that
# imported numba helpers from gnome3d.mc historically.  When adding new
# consumers, prefer importing directly from gnome3d.mc_numba.
_as_f64 = mc_numba._as_f64  # pyright: ignore[reportPrivateUsage]
_init_heat_nb = mc_numba._init_heat_nb  # pyright: ignore[reportPrivateUsage]

__all__ = [
    "mc_heatmap",
    "mc_arcs",
    "mc_smooth",
    "mc_ib",
    # backward-compat re-exports
    "_as_f64",
    "_init_heat_nb",
]

if TYPE_CHECKING:
    from .settings import Settings


# ---------------------------------------------------------------------------
# MC call profiler — append-only CSV of every top-level MC call.
# ---------------------------------------------------------------------------

_MC_PROFILE_PATH: str | None = os.environ.get("GNOME3D_MC_PROFILE")


def _log_mc_call(
    level: str, n: int, k: int, n_steps: int, wall_s: float, score: float, label: str
) -> None:
    if not _MC_PROFILE_PATH:
        return
    new_file = (not os.path.exists(_MC_PROFILE_PATH)) or os.path.getsize(_MC_PROFILE_PATH) == 0
    with open(_MC_PROFILE_PATH, "a") as f:
        if new_file:
            f.write("level,N,K,n_steps_per_batch,wall_s,score,label\n")
        safe_label = label.replace('"', "'")
        f.write(f'{level},{n},{k},{n_steps},{wall_s:.6f},{score:.6f},"{safe_label}"\n')


def _backend_is_jax(settings: Settings) -> bool:
    return str(settings.mc_backend).strip().lower() == "jax"


# ---------------------------------------------------------------------------
# Public entries — thin dispatchers
# ---------------------------------------------------------------------------


def mc_heatmap(
    pos: np.ndarray[Any, Any],
    exp_dist: np.ndarray[Any, Any],
    diag_size: int,
    step_size: float,
    settings: Settings,
    label: str = "",
    verbose: bool = False,
) -> float:
    """Heatmap-energy MC dispatch.  No JAX backend; always numba.

    Heatmap-MC accounts for <0.1% of total wall time in production profiles
    (single chr-level call at N=3-23), so a JAX port is not worth the
    engineering.  The dispatcher is here for uniform profiling + signature
    parity with the other levels.
    """
    n = pos.shape[0]
    _t0 = time.perf_counter() if _MC_PROFILE_PATH else 0.0

    score = mc_numba.mc_heatmap_numba(
        pos, exp_dist, diag_size, step_size, settings, label=label, verbose=verbose
    )

    if _MC_PROFILE_PATH:
        _log_mc_call(
            "heatmap",
            n,
            int(settings.mc_heatmap_chains),
            int(settings.mc_stop_steps_heatmap),
            time.perf_counter() - _t0,
            score,
            label,
        )
    return score


def mc_arcs(
    pos: np.ndarray[Any, Any],
    exp_dist_mat: np.ndarray[Any, Any],
    step_size: float,
    settings: Settings,
    label: str = "",
    verbose: bool = False,
) -> float:
    """Arc-MC dispatch — routes to JAX when `settings.mc_backend == "jax"`,
    else to numba.  Supported on JAX: arc springs + EV + confinement."""
    n = pos.shape[0]
    if n <= 1:
        return 0.0
    _t0 = time.perf_counter() if _MC_PROFILE_PATH else 0.0

    if _backend_is_jax(settings):
        from . import mc_jax

        if settings.output_level >= 1:
            lbl = f"[{label}] " if label else ""
            terms = ["arcs"]
            if bool(settings.use_excluded_volume) and bool(settings.exclusion_apply_to_arcs):
                terms.append("EV")
            if bool(settings.use_confinement) and bool(settings.confinement_apply_to_arcs):
                terms.append("conf")
            print(
                f"    {lbl}mc_arcs: backend=jax  N={n}  terms=[{'+'.join(terms)}]",
                flush=True,
            )
        score = mc_jax.mc_arcs_jax(
            pos, exp_dist_mat, step_size, settings, label=label, verbose=verbose
        )
    else:
        score = mc_numba.mc_arcs_numba(
            pos, exp_dist_mat, step_size, settings, label=label, verbose=verbose
        )

    if _MC_PROFILE_PATH:
        _log_mc_call(
            "arcs",
            n,
            1,
            int(settings.mc_stop_steps),
            time.perf_counter() - _t0,
            score,
            label,
        )
    return score


def mc_smooth(
    pos: np.ndarray[Any, Any],
    dtn: np.ndarray[Any, Any],
    fixed: np.ndarray[Any, Any],
    step_size: float,
    settings: Settings,
    char_orientations: np.ndarray[Any, Any] | None = None,
    anchor_neighbors: dict[int, list[int]] | None = None,
    anchor_neighbor_weights: dict[int, list[float]] | None = None,
    heat_dist: np.ndarray[Any, Any] | None = None,
    label: str = "",
    verbose: bool = False,
) -> float:
    """Smooth-MC dispatch — routes to JAX when `settings.mc_backend == "jax"`,
    else to numba.  Supported on JAX: chain bonds + angles + EV + heat
    (subanchor heatmap) + CTCF orientation + confinement (the full production
    energy combo)."""
    n = pos.shape[0]
    if n <= 2:
        return 0.0
    _t0 = time.perf_counter() if _MC_PROFILE_PATH else 0.0

    if _backend_is_jax(settings):
        from . import mc_jax

        if settings.output_level >= 1:
            lbl = f"[{label}] " if label else ""
            terms: list[str] = ["chain"]
            if bool(settings.use_excluded_volume) and bool(settings.exclusion_apply_to_smooth):
                terms.append("EV")
            if heat_dist is not None:
                terms.append("heat")
            if char_orientations is not None:
                terms.append("orient")
            if bool(settings.use_confinement) and bool(settings.confinement_apply_to_smooth):
                terms.append("conf")
            print(
                f"    {lbl}mc_smooth: backend=jax  N={n}  "
                f"K={int(settings.mc_smooth_chains)}  "
                f"terms=[{'+'.join(terms)}]",
                flush=True,
            )
        score = mc_jax.mc_smooth_jax(
            pos,
            dtn,
            fixed,
            step_size,
            settings,
            char_orientations=char_orientations,
            anchor_neighbors=anchor_neighbors,
            anchor_neighbor_weights=anchor_neighbor_weights,
            heat_dist=heat_dist,
            label=label,
            verbose=verbose,
        )
    else:
        score = mc_numba.mc_smooth_numba(
            pos,
            dtn,
            fixed,
            step_size,
            settings,
            char_orientations=char_orientations,
            anchor_neighbors=anchor_neighbors,
            anchor_neighbor_weights=anchor_neighbor_weights,
            heat_dist=heat_dist,
            label=label,
            verbose=verbose,
        )

    if _MC_PROFILE_PATH:
        _log_mc_call(
            "smooth",
            n,
            int(settings.mc_smooth_chains),
            int(settings.mc_stop_steps_smooth),
            time.perf_counter() - _t0,
            score,
            label,
        )
    return score


def mc_ib(
    pos: np.ndarray[Any, Any],
    dtn: np.ndarray[Any, Any],
    step_size: float,
    settings: Settings,
    label: str = "",
    verbose: bool = False,
) -> float:
    """IB-centroid chain MC dispatch.  No JAX backend; always numba.

    Per profile, IB-MC is <1% of total wall time on typical chromosomes, so
    a JAX port is not worth the engineering.  The dispatcher is here for
    uniform profiling + signature parity.
    """
    n = pos.shape[0]
    if n <= 1:
        return 0.0
    _t0 = time.perf_counter() if _MC_PROFILE_PATH else 0.0

    score = mc_numba.mc_ib_numba(pos, dtn, step_size, settings, label=label, verbose=verbose)

    if _MC_PROFILE_PATH:
        _log_mc_call(
            "ib",
            n,
            1,
            int(settings.mc_stop_steps_ib),
            time.perf_counter() - _t0,
            score,
            label,
        )
    return score
