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
        # One bond per proposal. At the reference's 5 the bonds never settle and come out 1.2 to
        # 1.4 times their target, and the distance curve is flat under 100 kb. Swept 2026-09-07.
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
        # Off. Inert on shallow Hi-C, where the pass is skipped for lack of active pairs, and on
        # deep Hi-C it pulls bonds to 0.83 of target once the step lets them settle, costing
        # Hi-C agreement and most of the wall. Measured 2026-09-07.
        "use_subanchor_heatmap": "no",
        "estimate_distances_steps": 4,
        "estimate_distances_replicates": 4,
        "heatmap_influence": 0.1,
        "heatmap_dist_weight": 0.01,
    },
    "heatmaps": {"inter_scaling": 1.0, "distance_heatmap_stretching": 2.5},
    "springs": {
        # Off. Redundant once the short range spring holds every arcless pair under 100 kb,
        # and measured null on H1ESC with that spring on.
        # Every arcless anchor pair under 100 kb held at the law's background. Puts the anchors on
        # the input curve on all three cells; 0.3 and 1.0 pull the closest pairs closer at a
        # steady cost in Hi-C. Measured 2026-09-07.
        "background_weight": 0.1,
        "background_range_bp": 100000,
        # Beyond that range, an arcless pair whose contact cell puts it under the background is
        # held at the law's contact distance. Sparse by construction, and on a thin map it
        # holds next to nothing, which is allowed: no input has to be deep.
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
        # Solve every anchor of a chromosome together, from the block layout, each block's
        # anchors on a walk at the law's distance per gap. Three cell gate on chr1:1-60 Mb
        # against the deep maps, 2026-09-10: Pearson 0.271/0.282/0.301 to 0.291/0.318/0.304,
        # MultiMM 0.607/0.568/0.652 to 0.674/0.667/0.673, SCC level within 0.01, cross block
        # overlaps 320/416/170 to 176/173/143 per thousand. The sphere for the chromosome's
        # span needs weight_arcs below.
        "scope": "chromosome",
        # The start is the long range arrangement: across blocks the joint solve holds no arc
        # and the contact background a few dozen pairs in millions. A Hilbert curve keeps
        # genomic neighbours spatial neighbours at every scale and grows as the cube root of
        # the count, so no sphere is needed to set the size. Like for like on three cells
        # against the walk with the sphere: Pearson up on all, SCC level, MultiMM level on
        # H1ESC and 0.01 to 0.04 down on the others, cross block overlaps 176/173/143 to
        # 163/126/100 per thousand, Rg 28/27/30 to 24/25/26.
        "start": "hilbert",
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
        # 0.7 of a bond. A radius of one bond was gated on 2026-09-11 and reverted the same
        # day: it clears the within block overlaps but pushes every pair under a bond outward,
        # contacts included, and costs Pearson 0.015, SCC 0.02 to 0.03 and the distance
        # exponent 0.045 on all three cells. The overlaps are addressed by the coil start and
        # the hard wall instead, which act on the pairs inside the radius only.
        "auto_factor_smooth": 0.7,
        "apply_to_heatmap": "yes",
        "apply_to_arcs": "yes",
        "apply_to_smooth": "yes",
        # Truncate the non-arc 1/d repulsion beyond factor times mean-arc-distance. 0.0 means
        # unbounded, faithful to LooperSolver.cpp:1533, which explodes small or sparse arcs
        # IBs from target 0.4 to Rg 515 and causes the multi-hour arcs polish. Capping the
        # long-range tail at 3x the natural arc scale keeps local de-clashing while stopping the
        # excess expansion.
        # 1.5 since 2026-09-08. At 3 the reach pushed a block's arcless anchors out to 3.6
        # beads where the law packs them 2.4 apart, and every pair beyond 100 kb ran away by
        # 1.3 to 1.8 times. At 1.5, with the contact background holding the far pairs that
        # carry data, the curve sits on the law to 2 Mb on three cells; 1.0 compacts past it.
        "arcs_repulsion_cutoff_factor": 1.5,
    },
    "confinement": {
        "use_confinement": "yes",
        "weight": 0.1,
        "apply_to_arcs": "yes",
        # Off with the Hilbert start, whose size follows from the curve; the walk start needed
        # 10 here, since at the shared 0.1 a walked chromosome stayed at Rg 44 against 22.
        "weight_arcs": 0,
        # The arcs sphere from the law, no constant. Null on its own, since the old formula
        # landed near it, and the principled form.
        "packing_factor_arcs": 0,
        "apply_to_smooth": "yes",
        "apply_to_ib": "yes",
        "packing_factor_ib": 0.75,
    },
    "boundary_stitch": {"use_boundary_stitch": "yes"},
    # Excluded volume across blocks, so the stitched globules cannot interpenetrate. Without it
    # nothing acts between the beads of two blocks once the stitch has moved them together.
    "relax": {
        "use_cross_block_relax": "yes",
        # Only the beads touching another block move, plus one chain neighbour either side.
        # With every subanchor movable the pass took an hour and fifty five minutes per
        # structure on a trio chr1 whatever the workload, and the last trio array timed out at
        # 24 hours on it. Replayed on the same chromosome a window of one took 129 seconds and
        # cut cross block contacts 31,637 to 362; a wider window only adds rounds.
        "local_window": 1,
    },
}

# Per-stage stop_condition_steps by quality. None or "full" keeps the canonical 50000.
_QUALITY_STEPS = {"fast": 1000, "balanced": 5000, "full": 50000}


def cell_data_section(cell: str, data_root: str = "data") -> dict[str, object]:
    """The data section for a cell line, by 3dgnome file-naming convention.

    The singletons are the Hi-C derived file, `<cell>_hic_25kb_singletons.bedpe`, when it is
    present in the cell's directory, and the ChIA-PET singletons when it is not. A deep map is
    not always available and nothing may require one.
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
