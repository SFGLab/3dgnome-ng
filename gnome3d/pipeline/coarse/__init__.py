"""
The coarse half of reconstruction: the cluster-tree spine, as a pipeline package.

Three modules, separated by role:
  * `build`   — `CoarseState` (the cluster graph + contact data) and the free
                functions over it: hierarchy build, the pure heatmap/expected-
                distance helpers, and the RNG-ordered positioning ops + seeding.
  * `heatmap` — the vectorized chr/segment distance-heatmap helpers `build` uses.
  * `stages`  — the `COARSE`-kind pipeline stages + `build_coarse_dag` that wrap
                `build`'s ops into the unified self-expanding DAG.

This `__init__` re-exports the `build` surface so callers can write
``from gnome3d.pipeline import coarse as cb; cb.build_state(...)``.  It does NOT
import `stages` — `stages` imports the `build` ops back through this package, so
keeping `stages` out of here avoids a cycle and keeps ``import ...coarse`` light
(no stage/registry pull-in).  Take `build_coarse_dag` / `ib_node_id` from
`gnome3d.pipeline.coarse.stages` directly (importing it also registers the
COARSE runner).
"""

from __future__ import annotations

from gnome3d.pipeline.coarse.build import (
    COARSE_SEED,
    CoarseState,
    add_long_pet_to_segment_heatmap,
    build_contact_heatmaps,
    build_state,
    calc_anchor_expected_distances,
    compute_segment_bins,
    create_distance_heatmap,
    ib_mc_refine,
    interpolate_children_linear,
    place_single_segment,
    position_interaction_blocks,
    random_walk_segment_level,
    reconstruct_chromosome_level,
    reconstruct_heatmap,
    reconstruct_segment_level,
    seed_global_rng,
    settings_for_ib,
    subanchor_counts_per_arc,
)

__all__ = [
    "COARSE_SEED",
    "CoarseState",
    "add_long_pet_to_segment_heatmap",
    "build_contact_heatmaps",
    "build_state",
    "calc_anchor_expected_distances",
    "compute_segment_bins",
    "create_distance_heatmap",
    "ib_mc_refine",
    "interpolate_children_linear",
    "place_single_segment",
    "position_interaction_blocks",
    "random_walk_segment_level",
    "reconstruct_chromosome_level",
    "reconstruct_heatmap",
    "reconstruct_segment_level",
    "seed_global_rng",
    "settings_for_ib",
    "subanchor_counts_per_arc",
]
