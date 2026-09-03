"""
Configuration for 3dgnome-ng.

Mirrors Reference Settings class.  All defaults match Settings::init() in Settings.cpp.
"""

import configparser
import difflib
import os
from collections.abc import Mapping
from pathlib import Path

from gnome3d import log

LOG = log.get("settings")


def _all_cores() -> int:
    """Usable CPU count for ``auto`` worker settings - honours cgroup / CPU-affinity
    limits on Linux, falls back to the logical core count elsewhere."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


class Settings:
    # ---- output / misc ----
    output_level: int
    log_file: str
    random_walk: bool
    use_2d: bool
    loop_density: int

    # ---- data paths ----
    data_dir: str
    data_anchors: str
    data_pet_clusters: str
    data_singletons: str
    data_singletons_inter: str
    data_factors: str
    data_split_singletons_by_chr: bool
    data_centromeres: str
    data_segment_split: str
    ib_refine_scope: str
    data_segment_heatmap: str
    data_compartments: str
    data_accessibility: str
    data_phasing_track: str

    # ---- template ----
    template_segment: str
    template_scale: float
    dist_heatmap: str
    dist_heatmap_scale: float

    # ---- motif orientation ----
    use_ctcf_motif: bool
    motifs_symmetric: bool
    motif_weight: float

    # ---- anchor heatmap ----
    use_anchor_heatmap: bool
    anchor_heatmap_influence: float
    anchor_heatmap_dist_weight: float

    # ---- subanchor heatmap ----
    use_subanchor_heatmap: bool
    subanchor_heatmap_influence: float
    subanchor_heatmap_dist_weight: float
    subanchor_estimate_steps: int
    subanchor_estimate_replicates: int
    subanchor_batch_trials: bool
    subanchor_heat_min_reduction: float
    # Threads building per-IB contact heatmaps during seed gathering (skeleton).  The
    # build is O(N^2) numpy per IB and embarrassingly parallel across IBs.  >1 parallelises
    # it; `auto` uses all usable CPU cores.  NB the (N,N) f32 matrices are large (N up to
    # ~32k -> ~4 GB each), so keep this modest if you have very large IBs (peak memory
    # ~= workers * largest matrix; `auto` can be a lot of memory there).
    heatmap_workers: int

    # ---- PET / arc length limits ----
    max_pet_length: int
    long_pet_power: float
    long_pet_scale: float

    # ---- heatmap parameters ----
    heatmap_inter_scaling: float
    heatmap_distance_stretching: float

    # ---- distance conversion ----
    genomic_dist_power: float
    genomic_dist_scale: float
    genomic_dist_base: float
    freq_dist_scale: float
    freq_dist_power: float
    freq_dist_scale_inter: float
    freq_dist_power_inter: float
    count_dist_a: float
    count_dist_scale: float
    count_dist_shift: float
    count_dist_base_level: float
    use_separation_arc_target: bool
    arc_target_exponent: float
    arc_target_pivot_kb: float

    # ---- spring constants ----
    spring_stretch: float
    spring_squeeze: float
    spring_angular: float
    spring_stretch_arcs: float
    spring_squeeze_arcs: float

    # ---- simulation steps ----
    steps_lvl1: int
    steps_lvl2: int
    steps_arcs: int
    steps_smooth: int

    # ---- noise coefficients ----
    noise_lvl1: float
    noise_lvl2: float
    noise_smooth: float
    # Note: `noise_arcs` is not kept. The Reference reference computes a
    # multiplied noise_size with noiseCoefficientLevelAnchor (Settings.cpp:364)
    # but the arc-MC call site passes a hardcoded noise_size_small=0.005 instead
    # (LooperSolver.cpp:2136). Python uses the same 0.005, so the setting has no
    # effect either way.

    # ---- MC heatmap ----
    max_temp_heatmap: float
    dt_temp_heatmap: float
    jump_scale_heatmap: float
    jump_coef_heatmap: float
    mc_stop_improvement_heatmap: float
    mc_stop_successes_heatmap: int
    mc_stop_steps_heatmap: int

    # ---- MC parallelism ----
    # `mc_*_chains` > 1 runs K independent MC chains in parallel
    # (numba prange + thread-local RNG) and keeps the best by final score.
    # K=1 disables.  Smooth multichain only triggers when the call uses the
    # simple chain+heat configuration (no orientation/EV/confinement) - more
    # complex configs fall back to single-chain.
    mc_heatmap_chains: int
    mc_smooth_chains: int
    # `ib_workers > 1` processes IBs concurrently (each IB is an independent
    # subproblem). JIT kernels are nogil=True, so Python threading actually
    # parallelises here.  `ib_workers = auto` uses all usable CPU cores.
    mc_executor_threaded_workers: int
    # Per-stage executor: how each IB stage's nodes are scheduled AND which kernel
    # runs them (the executor implies the backend).  One value per stage:
    #   'serial'   - numba, one node at a time (deterministic baseline).
    #   'threaded' - numba, independent nodes across `ib_workers` threads (kernels
    #                are nogil + thread-local RNG, so byte-identical to serial).
    #   'batch'    - JAX, same-(kind,bucket) nodes in one vmapped launch (GPU).
    #   'auto'     - resolve from the legacy mc_backend/ib_workers: jax + the old
    #                apply-flag -> batch; numba + ib_workers>1 -> threaded; else
    #                serial.  (Keeps existing configs working until they migrate.)
    # This replaces the old mc_backend_apply_to_* flags for executor selection;
    # mc_backend now only selects the *coarse* MC kernel (heatmap/ib).
    mc_executor_arcs: str
    mc_executor_densify: str
    mc_executor_estimate_dist: str
    mc_executor_smooth: str
    # Pad each JAX kernel's bead count up to a fixed bucket ladder so XLA
    # compiles ~one kernel per bucket instead of one per distinct region size.
    # Bounds total compiles regardless of how many distinct-N regions exist.
    # Padding is inert (pad beads never move + contribute zero energy), so
    # results are unchanged; this is a pure compile-time optimization.
    mc_executor_jax_bucket_shapes: bool
    # Cap on the region-batch vmap width (IBs per kernel launch) for the batched
    # JAX kernels, per kernel.  Excess IBs run in sequential sub-batches.  The cap
    # exists only to bound device memory (a wider launch is never slower than more
    # serial sub-batches).  "auto" sizes it from the device's memory limit and the
    # per-IB footprint at the group's bucket; an integer is a flat max IBs/launch.
    # Falls back to a fixed heuristic when device memory can't be queried (CPU).
    mc_executor_jax_batch_width_smooth: str
    mc_executor_jax_batch_width_arcs: str

    # Arcs JAX kernel: "mc" = sequential single-bead region-batch (default, byte-exact
    # port); "checker" = approximate color-gather spatial-checkerboard MC (much faster on
    # GPU for large IBs; a deliberate divergence from sequential dynamics, equal-energy).
    mc_executor_jax_arcs_kernel: str
    # Smooth JAX kernel, same choices.  The "checker" path OMITS the (constant) CTCF
    # orientation term from the score; the produced structures are correct.
    mc_executor_jax_smooth_kernel: str
    mc_executor_jax_estimate_kernel: str
    hybrid_polish_renoise: float

    # How the batch strategy uses several visible GPUs.  "groups" runs whole batch groups
    # side by side, one group per device, which keeps each group's launch intact and so draws
    # the same RNG as a one-device run.  "within" splits one group across devices, which only
    # pays off while groups hold more IBs than there are devices.  "off" pins to one device.
    mc_multigpu_mode: str

    # IB placement scores chain bonds between block centroids, excluded volume and confinement,
    # and discards every arc crossing a block boundary, so two blocks joined by many CTCF loops
    # are placed no closer than two joined by none. When on, cross-block arcs become a pairwise
    # target between centroids. Attraction only: a pair is given a target solely when its arc
    # support implies a distance SHORTER than its genomic separation already does. Measured on a
    # 20 Mb region it closes about a fifth of the cross-block distance penalty, because a
    # centroid is a coarse handle on where a block's edge anchors actually sit.
    use_ib_arcs: bool
    ib_arcs_weight: float

    # Anchors enter the per-block arc MC collapsed on their block's centroid, and that MC sees
    # only arcs internal to its block, so a cross-block arc never constrains anything. When on, a
    # segment-scope anchor pass runs first: every anchor of every block in a segment is placed in
    # one arc MC, where cross-block arcs are ordinary in-chain arcs. The per-block MC then refines
    # from those positions, and smooth MC holds anchors fixed, so the joint placement survives.

    # ---- MC arcs ----
    max_temp: float
    dt_temp: float
    jump_scale: float
    jump_coef: float
    mc_stop_improvement: float
    mc_stop_successes: int
    mc_stop_steps: int

    # ---- excluded volume ----
    use_excluded_volume: bool
    exclusion_weight: float
    exclusion_apply_to_arcs: bool
    exclusion_apply_to_smooth: bool
    exclusion_apply_to_heatmap: bool
    exclusion_apply_to_ib: bool
    exclusion_skip_neighbors: int
    # Per-level radius (one knob per MC level).  0.0 = auto = factor * mean
    # of that level's natural bond / expected distance.  Each level has its
    # own factor (default 0.5 - half the typical bead-bead target).
    exclusion_radius_arcs: float
    arcs_repulsion_cutoff_factor: float
    exclusion_radius_smooth: float
    exclusion_radius_heatmap: float
    exclusion_radius_ib: float
    exclusion_auto_factor_arcs: float
    exclusion_auto_factor_smooth: float
    exclusion_auto_factor_heatmap: float
    exclusion_auto_factor_ib: float
    use_genomic_floor: bool
    genomic_floor_factor: float
    genomic_floor_exponent: float
    genomic_floor_scale: float
    genomic_floor_polish_temp: float
    genomic_floor_weight: float

    # ---- IB-level MC pass (chain bonds + EV between IB centroids) ----
    # IB MC is a peer stage to smooth/arcs/heatmap, not a sub-mode of smooth.
    # It owns its own MC schedule, chain spring constants, and step noise.
    use_ib_mc: bool
    max_temp_ib: float
    dt_temp_ib: float
    jump_scale_ib: float
    jump_coef_ib: float
    mc_stop_improvement_ib: float
    mc_stop_successes_ib: int
    mc_stop_steps_ib: int
    spring_stretch_ib: float
    spring_squeeze_ib: float
    dist_weight_ib: float
    noise_ib: float

    # ---- confinement ----
    use_confinement: bool
    confinement_weight: float
    confinement_apply_to_arcs: bool
    confinement_apply_to_smooth: bool
    confinement_apply_to_ib: bool
    confinement_radius_arcs: float
    confinement_radius_smooth: float
    confinement_radius_ib: float
    confinement_packing_factor_arcs: float
    confinement_packing_factor_smooth: float
    confinement_packing_factor_ib: float

    # ---- boundary stitch ----
    use_boundary_stitch: bool
    boundary_stitch_spring_weight: float
    boundary_stitch_ev_weight: float
    boundary_stitch_max_iter: int

    # ---- A/B compartments ----
    use_compartments: bool
    compartment_weight: float
    compartment_energy_a: float
    compartment_energy_b: float
    compartment_apply_to_heatmap: bool
    compartment_apply_to_ib: bool
    compartment_apply_to_smooth: bool
    compartment_radius_heatmap: float
    compartment_radius_ib: float
    compartment_radius_smooth: float
    compartment_auto_factor_heatmap: float
    compartment_auto_factor_ib: float
    compartment_auto_factor_smooth: float

    # ---- chromatin accessibility ----
    use_bridging: bool
    bridging_weight: float
    bridging_apply_to_heatmap: bool
    bridging_apply_to_ib: bool
    bridging_apply_to_smooth: bool
    bridging_radius_heatmap: float
    bridging_radius_ib: float
    bridging_radius_smooth: float
    bridging_auto_factor_heatmap: float
    bridging_auto_factor_ib: float
    bridging_auto_factor_smooth: float
    use_fibre_compaction: bool
    fibre_compaction: float
    accessibility_mode: str
    accessibility_percentile: float

    # ---- nuclear forces ----
    use_lamina: bool
    lamina_weight: float
    use_central_force: bool
    central_weight: float
    use_chromosomal_blocks: bool
    chrom_block_kc: float
    chrom_block_weight: float
    nucleus_radius: float
    nucleus_packing_factor: float
    nucleus_inner_fraction: float

    # ---- overlapping-anchor handling (densification) ----
    overlap_anchor_strict: bool
    drop_zero_length_subanchors: bool

    # ---- dynamic loop density ----
    use_dynamic_loop_density: bool
    target_bp_per_subanchor: int
    min_subanchors_per_arc: int
    max_subanchors_per_arc: int

    # ---- MC smooth ----
    max_temp_smooth: float
    dt_temp_smooth: float
    jump_scale_smooth: float
    jump_coef_smooth: float
    mc_stop_improvement_smooth: float
    mc_stop_successes_smooth: int
    mc_stop_steps_smooth: int
    smooth_dist_weight: float
    smooth_angle_weight: float

    def __init__(self) -> None:
        self._set_defaults()

    def _set_defaults(self) -> None:
        # ---- output / misc ----
        self.output_level = 0
        # Optional path for a full-detail (DEBUG) structured log sink, in
        # addition to stdout.  Empty -> stdout only.  Handy for reconstructing
        # parallel (ib_workers>1 / n_structures>1) runs after the fact.  The
        # --log-file CLI flag overrides this.
        self.log_file = ""
        self.random_walk = False
        self.use_2d = False
        self.loop_density = 5

        # ---- data paths ----
        self.data_dir = ""
        self.data_anchors = ""
        self.data_pet_clusters = ""
        self.data_singletons = ""
        self.data_singletons_inter = ""
        self.data_factors = ""
        self.data_split_singletons_by_chr = False
        self.data_centromeres = ""
        self.data_segment_split = ""
        # What forms one chain in the IB placement MC. "segment" refines each
        # segment's blocks separately, which is the prior behaviour and the
        # default. "chromosome" refines them all together, removing the dependency
        # on segment grouping but coupling every block pair through excluded
        # volume; measured over four GM12878 regions that dropped mean simulated
        # contact density from 0.087 to 0.035 and worsened block cohesion about
        # threefold, so it needs its own EV and confinement tuning.
        self.ib_refine_scope = "segment"
        self.data_segment_heatmap = ""
        self.data_compartments = ""
        self.data_accessibility = ""
        self.data_phasing_track = ""

        # ---- template ----
        self.template_segment = ""
        self.template_scale = 1.0
        self.dist_heatmap = ""
        self.dist_heatmap_scale = 1.0

        # ---- motif orientation ----
        self.use_ctcf_motif = False
        self.motifs_symmetric = True
        self.motif_weight = 1.0

        # ---- anchor heatmap ----
        self.use_anchor_heatmap = False
        self.anchor_heatmap_influence = 0.5
        self.anchor_heatmap_dist_weight = 1.0

        # ---- subanchor heatmap ----
        self.use_subanchor_heatmap = False
        self.subanchor_heatmap_influence = 0.5
        self.subanchor_heatmap_dist_weight = 1.0
        self.subanchor_estimate_steps = 2
        self.subanchor_estimate_replicates = 5
        # Opt-in (default off): run the IB estimate's n_reps*n_steps independent
        # anneals as ONE vmapped JAX kernel instead of a sequential python loop.
        # ~3-6x faster at large N on GPU (the per-step kernel is latency-bound,
        # leaving the GPU idle at chains=1).  JAX smooth backend only; falls back
        # to the sequential loop otherwise.  Diverges from the parity baseline.
        self.subanchor_batch_trials = False
        # Opt-in (default 0.0 = off, parity preserved): skip an IB's subanchor
        # heat-dist entirely when its signal is too sparse to matter.  The
        # active-pair fraction (n_active / n_pairs) is a provable upper bound on
        # the mean target-distance reduction the heat term can produce, and it is
        # known from the raw heatmap BEFORE any dry-smooth trials.  When that
        # bound is below this threshold, the (expensive) estimate trials are
        # skipped and the IB smooths without heat.  E.g. 0.001 skips IBs whose
        # heat could move mean pair distance by <0.1%.  Diverges from parity.
        self.subanchor_heat_min_reduction = 0.0
        self.heatmap_workers = 1

        # ---- PET / arc length limits ----
        self.max_pet_length = 1_000_000
        self.long_pet_power = 2.0
        self.long_pet_scale = 10.0

        # ---- heatmap parameters ----
        self.heatmap_inter_scaling = 1.0
        self.heatmap_distance_stretching = 2.0

        # ---- distance conversion ----
        self.genomic_dist_power = 0.5
        self.genomic_dist_scale = 1.0
        self.genomic_dist_base = 0.0
        self.freq_dist_scale = 100.0
        self.freq_dist_power = -0.333
        self.freq_dist_scale_inter = 100.0
        self.freq_dist_power_inter = -1.0
        self.count_dist_a = 0.5
        self.count_dist_scale = 20.0
        self.count_dist_shift = 1.0
        self.count_dist_base_level = 0.01
        # Separation aware arc target. The parity law maps PET count alone, so a 1 Mb arc
        # with four PETs targets the same distance as a 100 kb one. With the flag the target is
        # multiplied by max(1, s_kb / pivot)^exponent. See gnome3d/util.py.
        self.use_separation_arc_target = False
        self.arc_target_exponent = 0.285
        self.arc_target_pivot_kb = 10.0

        # ---- spring constants ----
        self.spring_stretch = 0.1
        self.spring_squeeze = 0.1
        self.spring_angular = 0.1
        self.spring_stretch_arcs = 1.0
        self.spring_squeeze_arcs = 1.0

        # ---- simulation steps ----
        self.steps_lvl1 = 2
        self.steps_lvl2 = 2
        self.steps_arcs = 5
        self.steps_smooth = 5

        # ---- noise coefficients ----
        self.noise_lvl1 = 1.0
        self.noise_lvl2 = 0.1
        self.noise_smooth = 0.5

        # ---- MC heatmap ----
        self.max_temp_heatmap = 20.0
        self.dt_temp_heatmap = 0.99995
        self.jump_scale_heatmap = 50.0
        self.jump_coef_heatmap = 20.0
        self.mc_stop_improvement_heatmap = 0.995
        self.mc_stop_successes_heatmap = 5
        self.mc_stop_steps_heatmap = 10000

        # ---- MC backend ----
        self.mc_heatmap_chains = 1
        self.mc_smooth_chains = 1
        self.mc_executor_arcs = "auto"
        self.mc_executor_densify = "auto"
        self.mc_executor_estimate_dist = "auto"
        self.mc_executor_smooth = "auto"
        self.mc_executor_threaded_workers = 1
        self.mc_executor_jax_bucket_shapes = False
        self.mc_executor_jax_batch_width_smooth = "auto"
        self.mc_executor_jax_batch_width_arcs = "auto"
        self.mc_executor_jax_arcs_kernel = "mc"
        self.mc_executor_jax_smooth_kernel = "mc"
        self.mc_executor_jax_estimate_kernel = "auto"  # auto = follow smooth (hybrid->hybrid)
        self.hybrid_polish_renoise = (
            1.0  # re-noise (x step) on hybrid-smooth polish init; recovers diversity
        )
        self.mc_multigpu_mode = "groups"
        self.use_ib_arcs = False
        self.ib_arcs_weight = 1.0

        # ---- MC arcs ----
        self.max_temp = 20.0
        self.dt_temp = 0.99995
        self.jump_scale = 50.0
        self.jump_coef = 20.0
        self.mc_stop_improvement = 0.995
        self.mc_stop_successes = 5
        self.mc_stop_steps = 10000

        # ---- excluded volume ----
        # One radius knob per MC level, with auto-derivation when set to 0.0
        # so the user doesn't need to know the typical bead-bead distance for
        # each level (anchor MC is unit-scale, smooth MC is unit-scale, heatmap
        # MC is at heatmap-distance scale, IB MC is at the genomic-distance
        # scale between IB midpoints). Auto picks `factor * mean(bond)` from
        # that level's own data - each level has its own factor (default 0.5).
        self.use_excluded_volume = False
        self.exclusion_weight = 0.5  # k: multiplier (comparable to spring_*)
        self.exclusion_apply_to_arcs = False
        self.exclusion_apply_to_smooth = True
        self.exclusion_apply_to_heatmap = False
        self.exclusion_apply_to_ib = True  # IB-level MC (default on with use_ib_mc)
        self.exclusion_skip_neighbors = 1  # skip pairs with |i-j| <= this (1 = skip bonded)
        # Per-level radius: 0.0 = auto from this level's bond-length mean.
        self.exclusion_radius_arcs = 0.0
        self.exclusion_radius_smooth = 0.0
        self.exclusion_radius_heatmap = 0.0
        self.exclusion_radius_ib = 0.0
        # Truncate the arcs non-arc 1/d repulsion at factor x mean-arc-distance (0 = off =
        # unbounded, faithful to the reference; ~2.5 fixes small/sparse IBs blowing up to huge Rg).
        self.arcs_repulsion_cutoff_factor = 0.0
        # Per-level auto factor: used only when the matching radius is 0.0.
        # 0.5 means "EV kicks in once beads get closer than half the typical
        # bond distance at this level".
        self.exclusion_auto_factor_arcs = 0.5
        self.exclusion_auto_factor_smooth = 0.5
        self.exclusion_auto_factor_heatmap = 0.5
        self.exclusion_auto_factor_ib = 0.5
        # The genomic floor. Arcless anchor pairs in the arcs MC get an excluded volume
        # radius of scale * (separation / 1000)^exponent instead of the 1/d repulsion.
        # scale 0 derives it as factor times the median consecutive anchor distance of a
        # first anneal. See gnome3d/pipeline/ib/floor.py.
        self.use_genomic_floor = False
        self.genomic_floor_factor = 0.44
        self.genomic_floor_exponent = 0.285
        self.genomic_floor_scale = 0.0
        # Starting temperature of the floor pass as a fraction of max_temp. The pass is a
        # polish of the first anneal, so it must not re-heat the structure. Zero accepts
        # only moves that do not raise the score.
        self.genomic_floor_polish_temp = 0.0
        # The floor's own weight. It replaces the 1/d repulsion, which reaches 5 at a distance
        # of 0.2, so it cannot ride the excluded volume term's 0.1: at that weight the whole
        # block collapsed to half its floor on chr1:1-60Mb.
        self.genomic_floor_weight = 1.0

        # ---- IB-level MC pass ----
        # When enabled, each segment runs a small chain-spring + EV MC pass over
        # its child IB centroids after the initial random-walk / interpolation
        # placement. Pushes IBs apart so each IB's smooth-MC sphere has room
        # to breathe - addresses the "central blob" pathology with dynamic
        # loop density and many subanchors per IB.  EV inside this pass is
        # controlled by `exclusion_apply_to_ib`, `exclusion_radius_ib`, and
        # `exclusion_auto_factor_ib` under [excluded_volume]. IB MC owns its
        # own MC schedule + chain spring constants; defaults mirror the smooth
        # stage so existing configs behave identically.
        self.use_ib_mc = False
        self.max_temp_ib = 20.0
        self.dt_temp_ib = 0.99995
        self.jump_scale_ib = 50.0
        self.jump_coef_ib = 20.0
        self.mc_stop_improvement_ib = 0.995
        self.mc_stop_successes_ib = 5
        self.mc_stop_steps_ib = 10000
        self.spring_stretch_ib = 0.1
        self.spring_squeeze_ib = 0.1
        self.dist_weight_ib = 1.0
        self.noise_ib = 0.5

        # ---- confinement ----
        # Soft sphere around per-MC-call centroid; pulls beads back inside.
        # Each level has its own radius and packing factor: anchor MC, smooth MC
        # and IB MC operate at different spatial scales, so the typical "ball
        # radius" is also different. radius = 0 auto-derives from that level's
        # own bond data as `packing_factor * mean(bond) * N^(1/3)`.
        self.use_confinement = False
        self.confinement_weight = 0.5
        self.confinement_apply_to_arcs = True
        self.confinement_apply_to_smooth = True
        self.confinement_apply_to_ib = True
        self.confinement_radius_arcs = 0.0
        self.confinement_radius_smooth = 0.0
        self.confinement_radius_ib = 0.0
        # Packing factor for the auto formula; defaults tuned per level.
        # IB chains are short and should pack tighter (the original blob
        # pathology comes from over-extending the IB chain) so default < 1.
        self.confinement_packing_factor_arcs = 1.5
        self.confinement_packing_factor_smooth = 1.5
        self.confinement_packing_factor_ib = 0.75

        # ---- boundary stitch ----
        # Rigid post pass that closes the gap between adjacent blocks' edge anchors.
        # See gnome3d/pipeline/stitch.py.
        self.use_boundary_stitch = False
        self.boundary_stitch_spring_weight = 1.0
        self.boundary_stitch_ev_weight = 1.0
        self.boundary_stitch_max_iter = 500

        # ---- A/B compartments ----
        # Block-copolymer segregation over a per-bead compartment call, ported
        # from MultiMM's compartment blocks.  The well is written non-negative
        # because 3dgnome's Metropolis rule
        # reads a score ratio and needs a positive total.  Requires a
        # compartment track under [data]; inert when none is loaded.
        self.use_compartments = False
        self.compartment_weight = 1.0
        self.compartment_energy_a = 1.0  # MultiMM COB_EA
        self.compartment_energy_b = 2.0  # MultiMM COB_EB
        self.compartment_apply_to_heatmap = True
        self.compartment_apply_to_ib = True
        self.compartment_apply_to_smooth = True
        # Interaction range: 0.0 = auto = factor * mean(bond) at that level.
        # MultiMM uses r_comp = 1.5 * b0, which is the default factor.
        self.compartment_radius_heatmap = 0.0
        self.compartment_radius_ib = 0.0
        self.compartment_radius_smooth = 0.0
        self.compartment_auto_factor_heatmap = 1.5
        self.compartment_auto_factor_ib = 1.5
        self.compartment_auto_factor_smooth = 1.5

        # ---- chromatin accessibility ----
        # Bridging: accessible beads attract each other, HiP-HoP's diffusing
        # bridges integrated out into an effective pairwise well.  Defaults to
        # smooth only because accessibility varies bead-to-bead at subanchor
        # scale and is near-constant over a coarse bead.
        # Fibre compaction: closed chromatin shortens the chain bond target,
        # standing in for HiP-HoP's extra i,i+2 springs.
        self.use_bridging = False
        self.bridging_weight = 1.0
        self.bridging_apply_to_heatmap = False
        self.bridging_apply_to_ib = False
        self.bridging_apply_to_smooth = True
        self.bridging_radius_heatmap = 0.0
        self.bridging_radius_ib = 0.0
        self.bridging_radius_smooth = 0.0
        self.bridging_auto_factor_heatmap = 1.5
        self.bridging_auto_factor_ib = 1.5
        self.bridging_auto_factor_smooth = 1.5
        self.use_fibre_compaction = False
        self.fibre_compaction = 0.3  # 0 = off, 1 = fully collapse closed chromatin
        # How a raw accessibility track becomes the [0, 1] scale the terms read.
        # "log" is log-then-minmax.  "binary" is HiP-HoP's open/closed state, open
        # at or above `accessibility_percentile` of the loaded values.  On a track
        # binned to several kb the log leaves the median bead reading 0.85 open,
        # so fibre compaction has almost nothing to act on; binary restores the
        # range.  Default stays "log" so existing configs are unchanged.
        self.accessibility_mode = "log"
        self.accessibility_percentile = 80.0

        # ---- nuclear forces ----
        # Lamina, nucleolar attraction and chromosome territories, from MultiMM.
        # All three read the shared nuclear frame and run at coarse levels only:
        # a single IB is far smaller than the shell width, so the terms carry no
        # gradient there.  Lamina needs a compartment track.
        self.use_lamina = False
        self.lamina_weight = 400.0  # MultiMM IBL_SCALE
        self.use_central_force = False
        self.central_weight = 20.0  # MultiMM CF_STRENGTH
        self.use_chromosomal_blocks = False
        self.chrom_block_kc = 0.3  # MultiMM CHB_KC
        self.chrom_block_weight = 1e-4  # MultiMM CHB_DE
        # Nuclear frame: 0.0 = auto = packing * mean(bond) * N^(1/3), MultiMM's
        # constant-density rule.  R1 = R2 * inner_fraction^(1/3).
        self.nucleus_radius = 0.0
        self.nucleus_packing_factor = 1.0
        self.nucleus_inner_fraction = 0.2

        # ---- overlapping-anchor handling ----
        # overlap_anchor_strict controls span computation in densification:
        #   False (default): subanchors tile the overlap region with non-degenerate
        #     genomic ranges (Python divergence).
        #   True: reference-parity - overlap clamps to 0, so MC-chain subanchors
        #     between overlapping anchors are placed at a single boundary point
        #     (matches LooperSolver.cpp:1829-1831).
        # drop_zero_length_subanchors is an independent output-filtering toggle:
        #   False (default): every densified subanchor appears in the BeadOut output,
        #     even if start == end.
        #   True: subanchor BeadOut entries with start == end are filtered out of
        #     the output (the MC chain still contains them; only the externally
        #     visible bead list drops them). Useful with strict mode to suppress
        #     the collapsed-overlap zero-length noise.
        self.overlap_anchor_strict = False
        self.drop_zero_length_subanchors = False

        # ---- dynamic loop density ----
        # When False (default), every arc gets exactly self.loop_density subanchors.
        # When True, subanchor count for arc i is round(span_bp / target_bp_per_subanchor),
        # clamped to [min_subanchors_per_arc, max_subanchors_per_arc].  Aims to keep
        # roughly equal genomic distance between beads instead of equal beads per arc.
        # If the arc span is small relative to the target the count drops toward
        # min_subanchors_per_arc (0 → adjacent anchors get no subanchors).
        # The contact-heatmap binning and densification stay in sync - both use the
        # same per-arc counts, so use_subanchor_heatmap remains compatible.
        self.use_dynamic_loop_density = False
        self.target_bp_per_subanchor = 5000  # 5 kb per bead at default density
        self.min_subanchors_per_arc = 0  # allow very short arcs to skip subanchors
        self.max_subanchors_per_arc = 50  # cap to avoid runaway on huge gaps

        # ---- MC smooth ----
        self.max_temp_smooth = 20.0
        self.dt_temp_smooth = 0.99995
        self.jump_scale_smooth = 50.0
        self.jump_coef_smooth = 20.0
        self.mc_stop_improvement_smooth = 0.995
        self.mc_stop_successes_smooth = 5
        self.mc_stop_steps_smooth = 10000
        self.smooth_dist_weight = 1.0
        self.smooth_angle_weight = 1.0

    def load_ini(self, path: str) -> bool:
        """Load settings from an .ini file, overriding defaults in place."""
        cfg = configparser.ConfigParser()
        cfg.read(path)
        return self._load_from_parser(cfg)

    @classmethod
    def from_dict(cls, config: Mapping[str, Mapping[str, object]]) -> "Settings":
        """Build a Settings from a nested ``{section: {key: value}}`` mapping,
        mirroring the .ini layout — e.g.::

            Settings.from_dict({
                "data": {"data_dir": "data/GM12878/", "anchors": "...bed"},
                "excluded_volume": {"use_excluded_volume": True},
            })

        Section and key names are exactly those used in the .ini files. Any key
        not provided keeps its default. Values may be native Python types
        (bool/int/float/str); they are stringified into the same ConfigParser the
        .ini path uses, so parsing, type-coercion and unknown-key warnings are
        identical to ``load_ini``. Use this for notebooks, sweeps, and the
        validation harness instead of writing temporary .ini files.
        """
        cfg = configparser.ConfigParser()
        cfg.read_dict(
            {
                str(section): {str(k): str(v) for k, v in keys.items()}
                for section, keys in config.items()
            }
        )
        s = cls()
        s._load_from_parser(cfg)
        return s

    def _load_from_parser(self, cfg: configparser.ConfigParser) -> bool:
        """Apply settings from a populated ConfigParser, overriding defaults in
        place. Shared by `load_ini` (file) and `from_dict` (mapping)."""
        # Every (section, key) the loader below actually consults.  Used after
        # all reads to flag unknown keys in sections we own (see
        # `_warn_unknown_keys`).  Store keys as ConfigParser stores them
        # (optionxform-normalised) so lookups match the file's options exactly.
        consulted: dict[str, set[str]] = {}

        def get(section: str, key: str) -> str | None:
            consulted.setdefault(section, set()).add(cfg.optionxform(key))
            try:
                return cfg.get(section, key)
            except (configparser.NoSectionError, configparser.NoOptionError):
                return None

        def geti(section: str, key: str, default: int) -> int:
            v = get(section, key)
            return int(v) if v is not None else default

        def getf(section: str, key: str, default: float) -> float:
            v = get(section, key)
            return float(v) if v is not None else default

        def getb(section: str, key: str, default: bool) -> bool:
            v = get(section, key)
            if v is None:
                return default
            return v.strip().lower() in ("yes", "true", "1")

        def gets(section: str, key: str, default: str) -> str:
            v = get(section, key)
            return v.strip() if v is not None else default

        def getworkers(section: str, key: str, default: int) -> int:
            """Worker-count getter: ``auto`` -> all usable CPU cores, else an int."""
            v = get(section, key)
            if v is None:
                return default
            v = v.strip().lower()
            return _all_cores() if v == "auto" else int(v)

        def ignore(section: str, key: str) -> None:
            """Declare a key we deliberately don't read, so the unknown-key
            check below doesn't flag it (it's recognised, just unused)."""
            consulted.setdefault(section, set()).add(cfg.optionxform(key))

        # [main]
        self.output_level = geti("main", "output_level", self.output_level)
        self.log_file = gets("main", "log_file", self.log_file)
        self.random_walk = getb("main", "random_walk", self.random_walk)
        self.use_2d = getb("main", "use_2D", self.use_2d)
        self.loop_density = geti("main", "loop_density", self.loop_density)
        self.max_pet_length = geti("main", "max_pet_length", self.max_pet_length)
        self.long_pet_power = getf("main", "long_pet_power", self.long_pet_power)
        self.long_pet_scale = getf("main", "long_pet_scale", self.long_pet_scale)
        self.steps_lvl1 = geti("main", "steps_lvl1", self.steps_lvl1)
        self.steps_lvl2 = geti("main", "steps_lvl2", self.steps_lvl2)
        self.steps_arcs = geti("main", "steps_arcs", self.steps_arcs)
        self.steps_smooth = geti("main", "steps_smooth", self.steps_smooth)
        self.noise_lvl1 = getf("main", "noise_lvl1", self.noise_lvl1)
        self.noise_lvl2 = getf("main", "noise_lvl2", self.noise_lvl2)
        ignore("main", "noise_arcs")  # intentionally unused (see Settings class comment)
        self.noise_smooth = getf("main", "noise_smooth", self.noise_smooth)
        self.noise_ib = getf("main", "noise_ib", self.noise_ib)

        # [data]
        self.data_dir = gets("data", "data_dir", self.data_dir)
        self.data_anchors = gets("data", "anchors", self.data_anchors)
        self.data_pet_clusters = gets("data", "clusters", self.data_pet_clusters)
        self.data_singletons = gets("data", "singletons", self.data_singletons)
        self.data_singletons_inter = gets("data", "singletons_inter", self.data_singletons_inter)
        self.data_factors = gets("data", "factors", self.data_factors)
        self.data_split_singletons_by_chr = getb(
            "data", "split_singleton_files_by_chr", self.data_split_singletons_by_chr
        )
        self.data_centromeres = gets("data", "centromeres", self.data_centromeres)
        self.data_segment_split = gets("data", "segment_split", self.data_segment_split)
        self.ib_refine_scope = gets("simulation_ib", "refine_scope", self.ib_refine_scope)
        self.data_segment_heatmap = gets("data", "segment_heatmap", self.data_segment_heatmap)
        self.data_compartments = gets("data", "compartments", self.data_compartments)
        self.data_accessibility = gets("data", "accessibility", self.data_accessibility)
        self.data_phasing_track = gets("data", "phasing_track", self.data_phasing_track)

        # [template]
        self.template_segment = gets("template", "template_segment", self.template_segment)
        self.template_scale = getf("template", "template_scale", self.template_scale)
        self.dist_heatmap = gets("template", "dist_heatmap", self.dist_heatmap)
        self.dist_heatmap_scale = getf("template", "dist_heatmap_scale", self.dist_heatmap_scale)

        # [distance]
        self.genomic_dist_power = getf("distance", "genomic_dist_power", self.genomic_dist_power)
        self.genomic_dist_scale = getf("distance", "genomic_dist_scale", self.genomic_dist_scale)
        self.genomic_dist_base = getf("distance", "genomic_dist_base", self.genomic_dist_base)
        self.freq_dist_scale = getf("distance", "freq_dist_scale", self.freq_dist_scale)
        self.freq_dist_power = getf("distance", "freq_dist_power", self.freq_dist_power)
        self.freq_dist_scale_inter = getf(
            "distance", "freq_dist_scale_inter", self.freq_dist_scale_inter
        )
        self.freq_dist_power_inter = getf(
            "distance", "freq_dist_power_inter", self.freq_dist_power_inter
        )
        self.count_dist_a = getf("distance", "count_dist_a", self.count_dist_a)
        self.count_dist_scale = getf("distance", "count_dist_scale", self.count_dist_scale)
        self.count_dist_shift = getf("distance", "count_dist_shift", self.count_dist_shift)
        self.count_dist_base_level = getf(
            "distance", "count_dist_base_level", self.count_dist_base_level
        )
        self.use_separation_arc_target = getb(
            "distance", "use_separation_arc_target", self.use_separation_arc_target
        )
        self.arc_target_exponent = getf("distance", "arc_target_exponent", self.arc_target_exponent)
        self.arc_target_pivot_kb = getf("distance", "arc_target_pivot_kb", self.arc_target_pivot_kb)

        # [heatmaps]
        self.heatmap_inter_scaling = getf("heatmaps", "inter_scaling", self.heatmap_inter_scaling)
        self.heatmap_distance_stretching = getf(
            "heatmaps", "distance_heatmap_stretching", self.heatmap_distance_stretching
        )

        # [springs]
        self.spring_stretch = getf("springs", "stretch_constant", self.spring_stretch)
        self.spring_squeeze = getf("springs", "squeeze_constant", self.spring_squeeze)
        self.spring_angular = getf("springs", "angular_constant", self.spring_angular)
        self.spring_stretch_arcs = getf(
            "springs", "stretch_constant_arcs", self.spring_stretch_arcs
        )
        self.spring_squeeze_arcs = getf(
            "springs", "squeeze_constant_arcs", self.spring_squeeze_arcs
        )
        self.spring_stretch_ib = getf("springs", "stretch_constant_ib", self.spring_stretch_ib)
        self.spring_squeeze_ib = getf("springs", "squeeze_constant_ib", self.spring_squeeze_ib)

        # [motif_orientation]
        self.use_ctcf_motif = getb(
            "motif_orientation", "use_motif_orientation", self.use_ctcf_motif
        )
        self.motif_weight = getf("motif_orientation", "weight", self.motif_weight)
        self.motifs_symmetric = getb("motif_orientation", "symmetric_motifs", self.motifs_symmetric)

        # [anchor_heatmap]
        self.use_anchor_heatmap = getb(
            "anchor_heatmap", "use_anchor_heatmap", self.use_anchor_heatmap
        )
        self.anchor_heatmap_influence = getf(
            "anchor_heatmap", "heatmap_influence", self.anchor_heatmap_influence
        )

        # [subanchor_heatmap]
        self.use_subanchor_heatmap = getb(
            "subanchor_heatmap", "use_subanchor_heatmap", self.use_subanchor_heatmap
        )
        self.subanchor_heatmap_influence = getf(
            "subanchor_heatmap", "heatmap_influence", self.subanchor_heatmap_influence
        )
        self.subanchor_heatmap_dist_weight = getf(
            "subanchor_heatmap", "heatmap_dist_weight", self.subanchor_heatmap_dist_weight
        )
        self.subanchor_estimate_steps = geti(
            "subanchor_heatmap", "estimate_distances_steps", self.subanchor_estimate_steps
        )
        self.subanchor_estimate_replicates = geti(
            "subanchor_heatmap", "estimate_distances_replicates", self.subanchor_estimate_replicates
        )
        self.subanchor_batch_trials = getb(
            "subanchor_heatmap", "batch_trials", self.subanchor_batch_trials
        )
        self.subanchor_heat_min_reduction = getf(
            "subanchor_heatmap", "heat_min_reduction", self.subanchor_heat_min_reduction
        )
        self.heatmap_workers = getworkers(
            "subanchor_heatmap", "heatmap_workers", self.heatmap_workers
        )

        # [simulation_heatmap]
        self.max_temp_heatmap = getf(
            "simulation_heatmap", "max_temp_heatmap", self.max_temp_heatmap
        )
        self.dt_temp_heatmap = getf(
            "simulation_heatmap", "delta_temp_heatmap", self.dt_temp_heatmap
        )
        self.jump_scale_heatmap = getf(
            "simulation_heatmap", "jump_temp_scale_heatmap", self.jump_scale_heatmap
        )
        self.jump_coef_heatmap = getf(
            "simulation_heatmap", "jump_temp_coef_heatmap", self.jump_coef_heatmap
        )
        self.mc_stop_steps_heatmap = geti(
            "simulation_heatmap", "stop_condition_steps_heatmap", self.mc_stop_steps_heatmap
        )
        self.mc_stop_improvement_heatmap = getf(
            "simulation_heatmap",
            "stop_condition_improvement_threshold_heatmap",
            self.mc_stop_improvement_heatmap,
        )
        self.mc_stop_successes_heatmap = geti(
            "simulation_heatmap",
            "stop_condition_successes_threshold_heatmap",
            self.mc_stop_successes_heatmap,
        )

        # [simulation_backend]
        self.mc_heatmap_chains = geti(
            "simulation_backend", "heatmap_chains", self.mc_heatmap_chains
        )
        self.mc_smooth_chains = geti("simulation_backend", "smooth_chains", self.mc_smooth_chains)
        self.mc_executor_threaded_workers = getworkers(
            "simulation_backend", "ib_workers", self.mc_executor_threaded_workers
        )
        self.mc_executor_arcs = gets(
            "simulation_backend", "mc_executor_arcs", self.mc_executor_arcs
        )
        self.mc_executor_densify = gets(
            "simulation_backend", "mc_executor_densify", self.mc_executor_densify
        )
        self.mc_executor_estimate_dist = gets(
            "simulation_backend", "mc_executor_estimate_dist", self.mc_executor_estimate_dist
        )
        self.mc_executor_smooth = gets(
            "simulation_backend", "mc_executor_smooth", self.mc_executor_smooth
        )
        self.mc_executor_jax_bucket_shapes = getb(
            "simulation_backend",
            "mc_executor_jax_bucket_shapes",
            self.mc_executor_jax_bucket_shapes,
        )
        self.mc_executor_jax_batch_width_smooth = gets(
            "simulation_backend",
            "mc_executor_jax_batch_width_smooth",
            self.mc_executor_jax_batch_width_smooth,
        )
        self.mc_executor_jax_batch_width_arcs = gets(
            "simulation_backend",
            "mc_executor_jax_batch_width_arcs",
            self.mc_executor_jax_batch_width_arcs,
        )
        self.mc_executor_jax_arcs_kernel = gets(
            "simulation_backend",
            "mc_executor_jax_arcs_kernel",
            self.mc_executor_jax_arcs_kernel,
        )
        self.mc_executor_jax_smooth_kernel = gets(
            "simulation_backend",
            "mc_executor_jax_smooth_kernel",
            self.mc_executor_jax_smooth_kernel,
        )
        self.mc_executor_jax_estimate_kernel = gets(
            "simulation_backend",
            "mc_executor_jax_estimate_kernel",
            self.mc_executor_jax_estimate_kernel,
        )
        self.hybrid_polish_renoise = getf(
            "simulation_backend", "hybrid_polish_renoise", self.hybrid_polish_renoise
        )
        self.mc_multigpu_mode = gets("simulation_backend", "multigpu_mode", self.mc_multigpu_mode)

        # [simulation_arcs]
        self.max_temp = getf("simulation_arcs", "max_temp", self.max_temp)
        self.dt_temp = getf("simulation_arcs", "delta_temp", self.dt_temp)
        self.jump_scale = getf("simulation_arcs", "jump_temp_scale", self.jump_scale)
        self.jump_coef = getf("simulation_arcs", "jump_temp_coef", self.jump_coef)
        self.mc_stop_steps = geti("simulation_arcs", "stop_condition_steps", self.mc_stop_steps)
        self.mc_stop_improvement = getf(
            "simulation_arcs", "stop_condition_improvement_threshold", self.mc_stop_improvement
        )
        self.mc_stop_successes = geti(
            "simulation_arcs", "stop_condition_successes_threshold", self.mc_stop_successes
        )

        # [excluded_volume]
        self.use_excluded_volume = getb(
            "excluded_volume", "use_excluded_volume", self.use_excluded_volume
        )
        self.exclusion_weight = getf("excluded_volume", "weight", self.exclusion_weight)
        self.exclusion_apply_to_arcs = getb(
            "excluded_volume", "apply_to_arcs", self.exclusion_apply_to_arcs
        )
        self.exclusion_apply_to_smooth = getb(
            "excluded_volume", "apply_to_smooth", self.exclusion_apply_to_smooth
        )
        self.exclusion_apply_to_heatmap = getb(
            "excluded_volume", "apply_to_heatmap", self.exclusion_apply_to_heatmap
        )
        self.exclusion_apply_to_ib = getb(
            "excluded_volume", "apply_to_ib", self.exclusion_apply_to_ib
        )
        self.exclusion_skip_neighbors = geti(
            "excluded_volume", "skip_neighbors", self.exclusion_skip_neighbors
        )
        # Per-level radii.  Key naming: radius_<level>.  0 = auto.
        self.exclusion_radius_arcs = getf(
            "excluded_volume", "radius_arcs", self.exclusion_radius_arcs
        )
        self.arcs_repulsion_cutoff_factor = getf(
            "excluded_volume", "arcs_repulsion_cutoff_factor", self.arcs_repulsion_cutoff_factor
        )
        self.exclusion_radius_smooth = getf(
            "excluded_volume", "radius_smooth", self.exclusion_radius_smooth
        )
        self.exclusion_radius_heatmap = getf(
            "excluded_volume", "radius_heatmap", self.exclusion_radius_heatmap
        )
        self.exclusion_radius_ib = getf("excluded_volume", "radius_ib", self.exclusion_radius_ib)
        # Per-level auto-factor.  Used only when the matching radius is 0.
        self.exclusion_auto_factor_arcs = getf(
            "excluded_volume", "auto_factor_arcs", self.exclusion_auto_factor_arcs
        )
        self.exclusion_auto_factor_smooth = getf(
            "excluded_volume", "auto_factor_smooth", self.exclusion_auto_factor_smooth
        )
        self.exclusion_auto_factor_heatmap = getf(
            "excluded_volume", "auto_factor_heatmap", self.exclusion_auto_factor_heatmap
        )
        self.exclusion_auto_factor_ib = getf(
            "excluded_volume", "auto_factor_ib", self.exclusion_auto_factor_ib
        )
        self.use_genomic_floor = getb(
            "excluded_volume", "use_genomic_floor", self.use_genomic_floor
        )
        self.genomic_floor_factor = getf(
            "excluded_volume", "genomic_floor_factor", self.genomic_floor_factor
        )
        self.genomic_floor_exponent = getf(
            "excluded_volume", "genomic_floor_exponent", self.genomic_floor_exponent
        )
        self.genomic_floor_scale = getf(
            "excluded_volume", "genomic_floor_scale", self.genomic_floor_scale
        )
        self.genomic_floor_polish_temp = getf(
            "excluded_volume", "genomic_floor_polish_temp", self.genomic_floor_polish_temp
        )
        self.genomic_floor_weight = getf(
            "excluded_volume", "genomic_floor_weight", self.genomic_floor_weight
        )

        # [simulation_ib]
        self.use_ib_mc = getb("simulation_ib", "use_ib_mc", self.use_ib_mc)
        self.max_temp_ib = getf("simulation_ib", "max_temp", self.max_temp_ib)
        self.dt_temp_ib = getf("simulation_ib", "delta_temp", self.dt_temp_ib)
        self.jump_scale_ib = getf("simulation_ib", "jump_temp_scale", self.jump_scale_ib)
        self.jump_coef_ib = getf("simulation_ib", "jump_temp_coef", self.jump_coef_ib)
        self.mc_stop_steps_ib = geti("simulation_ib", "stop_condition_steps", self.mc_stop_steps_ib)
        self.use_ib_arcs = getb("simulation_ib", "use_ib_arcs", self.use_ib_arcs)
        self.ib_arcs_weight = getf("simulation_ib", "arcs_weight", self.ib_arcs_weight)
        self.mc_stop_improvement_ib = getf(
            "simulation_ib",
            "stop_condition_improvement_threshold",
            self.mc_stop_improvement_ib,
        )
        self.mc_stop_successes_ib = geti(
            "simulation_ib",
            "stop_condition_successes_threshold",
            self.mc_stop_successes_ib,
        )
        self.dist_weight_ib = getf("simulation_ib", "dist_weight", self.dist_weight_ib)

        # [confinement]
        self.use_confinement = getb("confinement", "use_confinement", self.use_confinement)
        self.confinement_weight = getf("confinement", "weight", self.confinement_weight)
        self.confinement_apply_to_arcs = getb(
            "confinement", "apply_to_arcs", self.confinement_apply_to_arcs
        )
        self.confinement_apply_to_smooth = getb(
            "confinement", "apply_to_smooth", self.confinement_apply_to_smooth
        )
        self.confinement_apply_to_ib = getb(
            "confinement", "apply_to_ib", self.confinement_apply_to_ib
        )
        self.confinement_radius_arcs = getf(
            "confinement", "radius_arcs", self.confinement_radius_arcs
        )
        self.confinement_radius_smooth = getf(
            "confinement", "radius_smooth", self.confinement_radius_smooth
        )
        self.confinement_radius_ib = getf("confinement", "radius_ib", self.confinement_radius_ib)
        self.confinement_packing_factor_arcs = getf(
            "confinement", "packing_factor_arcs", self.confinement_packing_factor_arcs
        )
        self.confinement_packing_factor_smooth = getf(
            "confinement", "packing_factor_smooth", self.confinement_packing_factor_smooth
        )
        self.confinement_packing_factor_ib = getf(
            "confinement", "packing_factor_ib", self.confinement_packing_factor_ib
        )

        # [boundary_stitch]
        self.use_boundary_stitch = getb(
            "boundary_stitch", "use_boundary_stitch", self.use_boundary_stitch
        )
        self.boundary_stitch_spring_weight = getf(
            "boundary_stitch", "spring_weight", self.boundary_stitch_spring_weight
        )
        self.boundary_stitch_ev_weight = getf(
            "boundary_stitch", "ev_weight", self.boundary_stitch_ev_weight
        )
        self.boundary_stitch_max_iter = geti(
            "boundary_stitch", "max_iter", self.boundary_stitch_max_iter
        )

        # [compartments]
        self.use_compartments = getb("compartments", "use_compartments", self.use_compartments)
        self.compartment_weight = getf("compartments", "weight", self.compartment_weight)
        self.compartment_energy_a = getf("compartments", "energy_a", self.compartment_energy_a)
        self.compartment_energy_b = getf("compartments", "energy_b", self.compartment_energy_b)
        self.compartment_apply_to_heatmap = getb(
            "compartments", "apply_to_heatmap", self.compartment_apply_to_heatmap
        )
        self.compartment_apply_to_ib = getb(
            "compartments", "apply_to_ib", self.compartment_apply_to_ib
        )
        self.compartment_apply_to_smooth = getb(
            "compartments", "apply_to_smooth", self.compartment_apply_to_smooth
        )
        self.compartment_radius_heatmap = getf(
            "compartments", "radius_heatmap", self.compartment_radius_heatmap
        )
        self.compartment_radius_ib = getf("compartments", "radius_ib", self.compartment_radius_ib)
        self.compartment_radius_smooth = getf(
            "compartments", "radius_smooth", self.compartment_radius_smooth
        )
        self.compartment_auto_factor_heatmap = getf(
            "compartments", "auto_factor_heatmap", self.compartment_auto_factor_heatmap
        )
        self.compartment_auto_factor_ib = getf(
            "compartments", "auto_factor_ib", self.compartment_auto_factor_ib
        )
        self.compartment_auto_factor_smooth = getf(
            "compartments", "auto_factor_smooth", self.compartment_auto_factor_smooth
        )

        # [accessibility]
        self.accessibility_mode = gets("accessibility", "mode", self.accessibility_mode)
        self.accessibility_percentile = getf(
            "accessibility", "percentile", self.accessibility_percentile
        )
        self.use_bridging = getb("accessibility", "use_bridging", self.use_bridging)
        self.bridging_weight = getf("accessibility", "bridging_weight", self.bridging_weight)
        self.bridging_apply_to_heatmap = getb(
            "accessibility", "apply_to_heatmap", self.bridging_apply_to_heatmap
        )
        self.bridging_apply_to_ib = getb("accessibility", "apply_to_ib", self.bridging_apply_to_ib)
        self.bridging_apply_to_smooth = getb(
            "accessibility", "apply_to_smooth", self.bridging_apply_to_smooth
        )
        self.bridging_radius_heatmap = getf(
            "accessibility", "radius_heatmap", self.bridging_radius_heatmap
        )
        self.bridging_radius_ib = getf("accessibility", "radius_ib", self.bridging_radius_ib)
        self.bridging_radius_smooth = getf(
            "accessibility", "radius_smooth", self.bridging_radius_smooth
        )
        self.bridging_auto_factor_heatmap = getf(
            "accessibility", "auto_factor_heatmap", self.bridging_auto_factor_heatmap
        )
        self.bridging_auto_factor_ib = getf(
            "accessibility", "auto_factor_ib", self.bridging_auto_factor_ib
        )
        self.bridging_auto_factor_smooth = getf(
            "accessibility", "auto_factor_smooth", self.bridging_auto_factor_smooth
        )
        self.use_fibre_compaction = getb(
            "accessibility", "use_fibre_compaction", self.use_fibre_compaction
        )
        self.fibre_compaction = getf("accessibility", "fibre_compaction", self.fibre_compaction)

        # [nucleus]
        self.use_lamina = getb("nucleus", "use_lamina", self.use_lamina)
        self.lamina_weight = getf("nucleus", "lamina_weight", self.lamina_weight)
        self.use_central_force = getb("nucleus", "use_central_force", self.use_central_force)
        self.central_weight = getf("nucleus", "central_weight", self.central_weight)
        self.use_chromosomal_blocks = getb(
            "nucleus", "use_chromosomal_blocks", self.use_chromosomal_blocks
        )
        self.chrom_block_kc = getf("nucleus", "chrom_block_kc", self.chrom_block_kc)
        self.chrom_block_weight = getf("nucleus", "chrom_block_weight", self.chrom_block_weight)
        self.nucleus_radius = getf("nucleus", "radius", self.nucleus_radius)
        self.nucleus_packing_factor = getf("nucleus", "packing_factor", self.nucleus_packing_factor)
        self.nucleus_inner_fraction = getf("nucleus", "inner_fraction", self.nucleus_inner_fraction)

        # [main] overlapping-anchor handling toggles (kept under [main] for simplicity).
        self.overlap_anchor_strict = getb(
            "main", "overlap_anchor_strict", self.overlap_anchor_strict
        )
        self.drop_zero_length_subanchors = getb(
            "main", "drop_zero_length_subanchors", self.drop_zero_length_subanchors
        )

        # [main] dynamic loop density toggles.
        self.use_dynamic_loop_density = getb(
            "main", "use_dynamic_loop_density", self.use_dynamic_loop_density
        )
        self.target_bp_per_subanchor = geti(
            "main", "target_bp_per_subanchor", self.target_bp_per_subanchor
        )
        self.min_subanchors_per_arc = geti(
            "main", "min_subanchors_per_arc", self.min_subanchors_per_arc
        )
        self.max_subanchors_per_arc = geti(
            "main", "max_subanchors_per_arc", self.max_subanchors_per_arc
        )

        # [simulation_arcs_smooth]
        self.smooth_dist_weight = getf(
            "simulation_arcs_smooth", "dist_weight", self.smooth_dist_weight
        )
        self.smooth_angle_weight = getf(
            "simulation_arcs_smooth", "angle_weight", self.smooth_angle_weight
        )
        self.max_temp_smooth = getf("simulation_arcs_smooth", "max_temp", self.max_temp_smooth)
        self.dt_temp_smooth = getf("simulation_arcs_smooth", "delta_temp", self.dt_temp_smooth)
        self.jump_scale_smooth = getf(
            "simulation_arcs_smooth", "jump_temp_scale", self.jump_scale_smooth
        )
        self.jump_coef_smooth = getf(
            "simulation_arcs_smooth", "jump_temp_coef", self.jump_coef_smooth
        )
        self.mc_stop_steps_smooth = geti(
            "simulation_arcs_smooth", "stop_condition_steps", self.mc_stop_steps_smooth
        )
        self.mc_stop_improvement_smooth = getf(
            "simulation_arcs_smooth",
            "stop_condition_improvement_threshold",
            self.mc_stop_improvement_smooth,
        )
        self.mc_stop_successes_smooth = geti(
            "simulation_arcs_smooth",
            "stop_condition_successes_threshold",
            self.mc_stop_successes_smooth,
        )

        self._warn_unknown_keys(cfg, consulted)
        return True

    @staticmethod
    def _warn_unknown_keys(cfg: configparser.ConfigParser, consulted: dict[str, set[str]]) -> None:
        """Warn about keys present in the file but never read by `load_ini`.

        A "mistype" is any option the loader never asks for: a misspelling
        (``mc_executor_jax_bukcet_shapes``), a wrong prefix
        (``jax_bucket_shapes`` instead of ``mc_executor_jax_bucket_shapes``), a
        stale/renamed name (``mc_executor_heat`` -> ``mc_executor_estimate_dist``),
        or a key put under the wrong section.  Any of these is silently ignored
        by ConfigParser, so the setting just doesn't take effect.

        Scoped to sections the loader actually consults, so reference-binary-only
        sections (e.g. ``[cuda]``) that the Python pipeline never reads don't
        produce false positives.
        """
        for section in cfg.sections():
            known = consulted.get(section)
            if not known:  # section we don't own; skip to avoid false positives
                continue
            for key in cfg.options(section):
                if key in known:
                    continue
                near = difflib.get_close_matches(key, known, n=1, cutoff=0.6)
                hint = f" (did you mean '{near[0]}'?)" if near else ""
                LOG.warning("[%s] unknown key '%s' is ignored%s", section, key, hint)

    def genomic_length_to_distance(self, length_bp: int) -> float:
        from gnome3d.util import genomic_length_to_distance

        return genomic_length_to_distance(
            length_bp, self.genomic_dist_base, self.genomic_dist_scale, self.genomic_dist_power
        )

    def freq_to_dist_heatmap(self, freq: float) -> float:
        from gnome3d.util import freq_to_dist_heatmap

        return freq_to_dist_heatmap(freq, self.freq_dist_scale, self.freq_dist_power)

    def freq_to_dist_heatmap_inter(self, freq: float) -> float:
        from gnome3d.util import freq_to_dist_heatmap_inter

        return freq_to_dist_heatmap_inter(
            freq, self.freq_dist_scale_inter, self.freq_dist_power_inter
        )

    def freq_to_distance(self, freq: int) -> float:
        from gnome3d.util import freq_to_distance

        return freq_to_distance(
            freq,
            self.count_dist_a,
            self.count_dist_scale,
            self.count_dist_shift,
            self.count_dist_base_level,
        )

    def arc_expected_distance(self, score: int, sep_bp: int) -> float:
        """The arc target for an arc of `score` PETs spanning `sep_bp`. The parity law when
        `use_separation_arc_target` is off."""
        from gnome3d.util import arc_target_with_separation

        base = self.freq_to_distance(score)
        if not self.use_separation_arc_target:
            return base
        return arc_target_with_separation(
            base, sep_bp, self.arc_target_pivot_kb, self.arc_target_exponent
        )

    def data_path(self, filename: str) -> str:
        """Resolve a data filename relative to data_dir."""
        if not filename:
            return ""
        p = Path(self.data_dir) / filename
        return str(p)
