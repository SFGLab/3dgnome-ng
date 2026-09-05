"""Canonical 3dgnome modelling config, baked in so no .ini is needed.

The shipped data/<cell>/config.ini files all share the same modelling parameters. Those are
distance, springs, template, motif, heatmap, MC schedule, and EV and confinement on at weight
0.1. Only the data filenames and data_dir differ. data_dir points at an absolute /Projects/
path that does not exist outside the authors' box. Rather than make the validation harness take
--config and --data-dir, we encode those canonical params here and build a Settings with
Settings.from_dict. The harness wires its own config for any cell line.

settings_for_cell("GM12878") returns a ready Settings. The sweep and validate CLIs then toggle
the EV and confinement knobs they are testing on top.
"""

from __future__ import annotations

import copy
from pathlib import Path

from gnome3d.settings import Settings

# Canonical modelling params from data/GM12878/config.ini, shared across cell lines. Excludes
# the data section, which is built per cell, and the cuda section, which is reference-binary
# only. EV and confinement are on at weight 0.1 here, the real default. The sweep overrides
# them per config.
CANONICAL: dict[str, dict[str, object]] = {
    "main": {
        "output_level": 1,
        "random_walk": "no",
        "loop_density": 5,
        "use_2D": "no",
        "max_pet_length": 1000000,
        "long_pet_power": 2.0,
        "long_pet_scale": 1.0,
        "steps_lvl1": 1,
        "steps_lvl2": 1,
        "steps_arcs": 1,
        "steps_smooth": 1,
        "noise_lvl1": 0.5,
        "noise_lvl2": 0.5,
        "noise_smooth": 5.0,
        "overlap_anchor_strict": "no",
        "drop_zero_length_subanchors": "yes",
        "use_dynamic_loop_density": "yes",
        "target_bp_per_subanchor": 1000,
        "min_subanchors_per_arc": 0,
        "max_subanchors_per_arc": 100,
    },
    "distance": {
        "use_separation_arc_target": "yes",
        "genomic_dist_power": 0.75,
        "genomic_dist_scale": 0.5,
        "genomic_dist_base": 1.0,
        "freq_dist_scale": 25.0,
        "freq_dist_power": -0.6,
        "freq_dist_scale_inter": 120.0,
        "freq_dist_power_inter": -1.0,
        "count_dist_a": 0.2,
        "count_dist_scale": 1.8,
        "count_dist_shift": 8,
        "count_dist_base_level": 0.2,
    },
    "template": {"template_scale": 7.0, "dist_heatmap_scale": 15.0},
    "motif_orientation": {"use_motif_orientation": "yes", "weight": 50.0},
    "anchor_heatmap": {"use_anchor_heatmap": "yes", "heatmap_influence": 0.1},
    "subanchor_heatmap": {
        "use_subanchor_heatmap": "yes",
        "estimate_distances_steps": 4,
        "estimate_distances_replicates": 4,
        "heatmap_influence": 0.1,
        "heatmap_dist_weight": 0.01,
    },
    "heatmaps": {"inter_scaling": 1.0, "distance_heatmap_stretching": 2.5},
    "springs": {
        "use_arcs_chain_bonds": "yes",
        "arcs_chain_bond_scale": 1.5,
        "stretch_constant": 0.1,
        "squeeze_constant": 0.1,
        "angular_constant": 0.1,
        "stretch_constant_arcs": 1.0,
        "squeeze_constant_arcs": 1.0,
    },
    "simulation_backend": {
        "ib_workers": "auto",
        "heatmap_chains": 1,
        "smooth_chains": 1,
        # Arcs runs on the CPU, not the GPU, and it is the one stage where that is true.
        # Measured on a genome scale trio run, where arcs is 89.6 percent of the wall: one
        # launch put 54 blocks together, 53 of them converged by round 2 and one needed 3,753,
        # and a vmapped launch cannot retire a converged chain, so all 54 ran 3,753 rounds.
        # That is 10.1 billion chain steps where independent CPU tasks do 193 million, because
        # each block exits when it converges. On top of that an arcs step reduces over 256 to
        # 2,048 anchors, which is a tight cache resident loop on a core at about 1.8 us and a
        # whole kernel dispatch on the device at 18.4 us measured. Smooth is the opposite shape,
        # eighty chains of 16,384 beads with similar convergence, and stays on the GPU.
        "mc_executor_arcs": "threaded",
        "mc_executor_densify": "threaded",
        "mc_executor_estimate_dist": "batch",
        "mc_executor_smooth": "batch",
        # Cell grid for the excluded volume term. Identical results, so this is only about
        # speed, and it is written out rather than left to the default so a config records it.
        "neighbour_grid": "yes",
    },
    "simulation_ib": {
        "use_ib_mc": "yes",
        "max_temp": 5.0,
        "jump_temp_scale": 50.0,
        "jump_temp_coef": 20.0,
        "delta_temp": 0.9999,
        "stop_condition_improvement_threshold": 0.999,
        "stop_condition_successes_threshold": 100,
        "stop_condition_steps": 50000,
    },
    "simulation_heatmap": {
        "max_temp_heatmap": 5.0,
        "delta_temp_heatmap": 0.9999,
        "jump_temp_scale_heatmap": 50.0,
        "jump_temp_coef_heatmap": 20.0,
        "stop_condition_improvement_threshold_heatmap": 0.999,
        "stop_condition_successes_threshold_heatmap": 10,
        "stop_condition_steps_heatmap": 50000,
    },
    "simulation_arcs": {
        # Solve the stage rather than anneal it. The landscape is a funnel, so a quasi Newton
        # descent lands in the same minimum. Measured over five structures on chr1:1-60Mb the
        # two arms agree on every quality number, Hi-C Pearson 0.403 against 0.405, distance
        # exponent 0.240 against 0.249, and the anchor overlap rate 89.2 against 89.1 per
        # thousand beads. The stage's two calls went from 492s to 6s and from 500s to 23s, and
        # the whole run from 1h57m to 1h13m. The batch executor has no solver in it, so this
        # needs mc_executor_arcs serial or threaded, which is what it is set to above.
        "solver": "lbfgs",
        "max_temp": 5.0,
        "jump_temp_scale": 50.0,
        "jump_temp_coef": 20.0,
        "delta_temp": 0.9999,
        "stop_condition_improvement_threshold": 0.999,
        "stop_condition_successes_threshold": 100,
        "stop_condition_steps": 50000,
    },
    "simulation_arcs_smooth": {
        "dist_weight": 1.0,
        "angle_weight": 1.0,
        "max_temp": 5.0,
        "jump_temp_scale": 50.0,
        "jump_temp_coef": 20.0,
        "delta_temp": 0.9999,
        "stop_condition_improvement_threshold": 0.999,
        "stop_condition_successes_threshold": 50,
        "stop_condition_steps": 50000,
    },
    "excluded_volume": {
        "use_excluded_volume": "yes",
        # EV is a gentle correction, not a dominant term. It weighs far less than the distance,
        # heatmap, and loop energies at dist_weight 1.0. Picked by the subordinate-grid sweep on
        # the unified config. 0.1 cut resolution-normalized overlaps in 20 of 20 GM12878 regions,
        # down 23% vs baseline, at +17% Rg, within the 0.30 guard. Higher weights do not lower
        # overlaps further but cost more Rg. Earlier weights of 1.0 to 2.0 over-expanded. The old
        # explosion was a config divergence bug, not EV. See [[project_config_unification]].
        "weight": 0.1,
        "auto_factor_smooth": 0.7,
        "apply_to_heatmap": "yes",
        "apply_to_arcs": "yes",
        "apply_to_smooth": "yes",
        # Truncate the non-arc 1/d repulsion beyond factor times mean-arc-distance. 0.0 means
        # unbounded, faithful to LooperSolver.cpp:1533, which explodes small or sparse arcs
        # IBs from target 0.4 to Rg 515 and causes the multi-hour arcs polish. Capping the
        # long-range tail at 3x the natural arc scale keeps local de-clashing while stopping the
        # excess expansion.
        "arcs_repulsion_cutoff_factor": 3.0,
    },
    "confinement": {
        "use_confinement": "yes",
        "weight": 0.1,
        "apply_to_arcs": "yes",
        "apply_to_smooth": "yes",
        "apply_to_ib": "yes",
        "packing_factor_ib": 0.75,
    },
    "boundary_stitch": {"use_boundary_stitch": "yes"},
    # Excluded volume across blocks, so the stitched globules cannot interpenetrate. Without it
    # nothing acts between the beads of two blocks once the stitch has moved them together.
    "relax": {"use_cross_block_relax": "yes"},
    "small_ib_boost": {"use_small_ib_boost": "no"},
}

# Per-stage stop_condition_steps by quality. None or "full" keeps the canonical 50000.
_QUALITY_STEPS = {"fast": 1000, "balanced": 5000, "full": 50000}


def cell_data_section(cell: str, data_root: str = "data") -> dict[str, object]:
    """The data section for a cell line, by 3dgnome file-naming convention."""
    return {
        "data_dir": str(Path(data_root) / cell),
        "anchors": f"{cell}_anchors_3+_oriented.bed",
        "clusters": f"{cell}_clusters_3+.bedpe",
        "factors": "CTCF",
        "singletons": f"{cell}_singletons_lessthan3.bedpe",
        "split_singleton_files_by_chr": "no",
        "singletons_inter": "",
        "segment_split": f"ccds_all_hg38_merged100k_{cell}.breakpoints.bed",
        "centromeres": "hg38_centromeres.bed",
    }


def settings_for_cell(
    cell: str,
    data_root: str = "data",
    quality: str | None = None,
    overrides: dict[str, dict[str, object]] | None = None,
) -> Settings:
    """Build a ready Settings for a cell from the canonical params and conventional data paths,
    with no .ini. quality is one of fast, balanced, full and rescales each stage's
    stop_condition_steps. None means full and canonical. overrides deep-merges extra
    {section: {key: value}} on top, for example feature-flag tweaks."""
    params = copy.deepcopy(CANONICAL)
    params["data"] = cell_data_section(cell, data_root)

    if quality and quality in _QUALITY_STEPS and quality != "full":
        steps = _QUALITY_STEPS[quality]
        params["simulation_heatmap"]["stop_condition_steps_heatmap"] = steps
        params["simulation_arcs"]["stop_condition_steps"] = steps
        params["simulation_arcs_smooth"]["stop_condition_steps"] = steps
        params["simulation_ib"]["stop_condition_steps"] = steps

    if overrides:
        for section, kv in overrides.items():
            params.setdefault(section, {}).update(kv)

    return Settings.from_dict(params)


# --- The single place that modifies a built Settings -------------------------------------------
# All validation tools must go through these helpers, or settings_for_cell's overrides. No tool
# should set Settings attributes inline. Keeps config logic in one auditable place.


def apply_flags(s: Settings, flags: dict[str, object]) -> Settings:
    """Return a deep copy of s with the given public attributes set. This is the canonical
    post-build config modifier for feature flags, executor, and data paths."""
    out = copy.deepcopy(s)
    for attr, val in flags.items():
        setattr(out, attr, val)
    return out


def with_arcs_executor(s: Settings, executor: str, workers: int = 0) -> Settings:
    """Set the arcs-stage MC executor, one of batch for GPU, threaded, or serial. threaded uses
    workers, where 0 means cpu_count."""
    import os

    flags: dict[str, object] = {"mc_executor_arcs": executor}
    if executor == "threaded":
        flags["mc_executor_threaded_workers"] = workers if workers > 0 else (os.cpu_count() or 1)
    return apply_flags(s, flags)


def with_singletons(s: Settings, singletons_path: str, singletons_inter: str = "") -> Settings:
    """Point the model at a custom singletons BEDPE, for example Hi-C-derived for the
    self-correlation study. Absolute paths override the data_dir prefix via pathlib join."""
    return apply_flags(
        s, {"data_singletons": singletons_path, "data_singletons_inter": singletons_inter}
    )
