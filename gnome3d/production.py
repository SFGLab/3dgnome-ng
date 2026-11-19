"""The production configuration, as a dict, with no .ini file.

`CANONICAL` holds every modelling section. `settings_for_cell("GM12878")` returns a `Settings`
built from it with the data section of that cell line. `SETTINGS.md` documents each key and its
production column is this dict.
"""

from __future__ import annotations

import copy
from pathlib import Path

from gnome3d.settings import Settings

# The modelling sections shared by every cell line. The data section is built per cell line.
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
        # One bond per proposal, design/anchor-placement.md.
        "noise_smooth": 1.0,
        "overlap_anchor_strict": "no",
        "drop_zero_length_subanchors": "yes",
        "use_dynamic_loop_density": "yes",
        "target_bp_per_subanchor": 1000,
        "min_subanchors_per_arc": 0,
        "max_subanchors_per_arc": 100,
    },
    "motif_orientation": {"use_motif_orientation": "yes", "weight": 50.0},
    "anchor_heatmap": {"use_anchor_heatmap": "yes", "heatmap_influence": 0.1},
    "subanchor_heatmap": {
        # Off. It pulls bonds under their target on deep Hi-C and is skipped on shallow Hi-C.
        "use_subanchor_heatmap": "no",
        "estimate_distances_steps": 4,
        "estimate_distances_replicates": 4,
        "heatmap_influence": 0.1,
        "heatmap_dist_weight": 0.01,
    },
    "heatmaps": {"inter_scaling": 1.0, "distance_heatmap_stretching": 2.5},
    "springs": {
        # Arcless anchor pairs under `background_range_bp` are held at the background of the law.
        "background_weight": 0.1,
        # Every genomic window of anchors is held at the radius of gyration of the law.
        "gyration_weight": 0.1,
        "background_range_bp": 100000,
        # Beyond the range, an arcless pair with more contacts than expected is held at the contact
        # distance of the law.
        "use_contact_background": "yes",
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
        # With the solver, batch evaluates the energy on the JAX device.
        "mc_executor_arcs": "batch",
        "mc_executor_densify": "threaded",
        "mc_executor_estimate_dist": "batch",
        # Faster than the device from four workers, design/kernel-performance.md section 14.
        "mc_executor_smooth": "threaded",
        # Identical results. Stated for the record of a config.
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
        # A quasi Newton solve reaches the minimum the annealer reaches.
        "solver": "lbfgs",
        # The solve stops on the energy. The count is a safety.
        "solver_tol": 1e-6,
        "solver_iters": 5000,
        # Every anchor of a chromosome is solved together.
        "scope": "chromosome",
        # One draw per conformation of the maximum entropy Gaussian whose pair distances meet the
        # arcs targets.
        "start": "maxent",
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
        # Subanchors start on a compact random bridge between their anchors. The wall rejects a move
        # that adds or deepens an overlap.
        "start": "coil",
        "hard_wall": "yes",
        "prefetch": 32,  # proposals per JAX step
        "jax_grid": "yes",  # excluded volume on a cell grid in the JAX kernel
        "anchor_cap": 0.0,
    },
    "excluded_volume": {
        "use_excluded_volume": "yes",
        # A weak correction beside the distance and loop terms.
        "weight": 0.1,
        # 0.7 of a bond. One bond pushes contacts apart.
        "auto_factor_smooth": 0.7,
        "apply_to_heatmap": "yes",
        "apply_to_arcs": "yes",
        "apply_to_smooth": "yes",
        # The reach of the arcless 1/d repulsion in mean arc distances. Read only while the
        # repulsion is on.
        "arcs_repulsion_cutoff_factor": 1.5,
        # Off. The gyration term holds the size and the arcs excluded volume the overlaps.
        "arcs_repulsion": "no",
        "auto_factor_arcs": 1.0,
    },
    "confinement": {
        "use_confinement": "yes",
        "weight": 0.1,
        "apply_to_arcs": "yes",
        # Off. The start sets the size.
        "weight_arcs": 0,
        # 0 derives the arcs sphere from the law.
        "packing_factor_arcs": 0,
        "apply_to_smooth": "yes",
        "apply_to_ib": "yes",
        "packing_factor_ib": 0.75,
    },
    # Off at chromosome scope.
    "boundary_stitch": {"use_boundary_stitch": "no"},
    "relax": {
        # Removes overlaps between blocks.
        "use_cross_block_relax": "yes",
        # Only the beads that touch another block move, with one chain neighbour on either side.
        "local_window": 1,
        # Overlaps inside a block move too, at a radius of one bead.
        "within_blocks": "yes",
        "ev_radius": 1.0,
    },
}

# `stop_condition_steps` of each stage by quality.
_QUALITY_STEPS = {"fast": 1000, "balanced": 5000, "full": 50000}


def cell_data_section(cell: str, data_root: str = "data") -> dict[str, object]:
    """Returns the data section of a cell line by the file name convention.

    The singletons are `<cell>_hic_25kb_singletons.bedpe` when that file exists and the ChIA-PET
    singletons otherwise.
    """
    hic = Path(data_root) / cell / f"{cell}_hic_25kb_singletons.bedpe"
    return {
        "data_dir": str(Path(data_root) / cell),
        "anchors": f"{cell}_anchors_3+_oriented.bed",
        "clusters": f"{cell}_clusters_3+.bedpe",
        "singletons": hic.name if hic.is_file() else f"{cell}_singletons_lessthan3.bedpe",
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
    """Builds the Settings of a cell line from `CANONICAL` and the conventional data paths.

    `quality` is fast, balanced or full and sets the `stop_condition_steps` of each stage. None is
    full. `overrides` is merged on top as {section: {key: value}}.
    """
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


def apply_flags(s: Settings, flags: dict[str, object]) -> Settings:
    """Returns a deep copy of `s` with the given attributes set."""
    out = copy.deepcopy(s)
    for attr, val in flags.items():
        setattr(out, attr, val)
    return out


def with_arcs_executor(s: Settings, executor: str, workers: int = 0) -> Settings:
    """Sets the executor of the arcs stage, batch, threaded or serial. `workers` of 0 uses every
    core.
    """
    import os

    flags: dict[str, object] = {"mc_executor_arcs": executor}
    if executor == "threaded":
        flags["mc_executor_threaded_workers"] = workers if workers > 0 else (os.cpu_count() or 1)
    return apply_flags(s, flags)


def with_singletons(s: Settings, singletons_path: str, singletons_inter: str = "") -> Settings:
    """Points the settings at another singletons BEDPE. An absolute path replaces the `data_dir`
    prefix.
    """
    return apply_flags(
        s, {"data_singletons": singletons_path, "data_singletons_inter": singletons_inter}
    )
