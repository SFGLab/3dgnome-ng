"""
Configuration for 3dgnome-ng.

Mirrors Reference Settings class.  All defaults match Settings::init() in Settings.cpp.
"""

import configparser
from pathlib import Path


class Settings:
    # ---- output / misc ----
    output_level: int
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
    data_segment_heatmap: str

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
    # parallelises here.
    ib_workers: int
    # `mc_backend` selects which compute backend handles the MC hot paths.
    # 'numba' (default) is the production-tested CPU implementation.
    # 'jax' routes per-level MC calls to a JAX/CUDA kernel — typically 5-30x
    # faster than numba on smooth-MC at N>=2048.  Per-level apply flags below
    # control which levels actually use JAX when mc_backend='jax'; flags have
    # no effect when mc_backend='numba'.  Defaults reflect measured wins:
    #   smooth:  yes (JAX wins big — chain+EV+heat+orient+conf, all dense O(N))
    #   arcs:    no  (JAX loses — arc energy is SPARSE, numba's per-pair
    #                early-continue beats JAX's dense kernel at production N)
    #   heatmap: no  (per profile, heatmap is <0.1% of typical wall — chr-level
    #                only fires at N=3-23 where JAX overhead dominates.
    #                Enable for multi-chr workloads where segment-level
    #                heatmap-MC can hit larger N.)
    # mc_ib has no JAX implementation (also <1% of typical wall per profile),
    # so no apply flag.
    mc_backend: str
    mc_backend_apply_to_smooth: bool
    mc_backend_apply_to_arcs: bool
    mc_backend_apply_to_heatmap: bool

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
    exclusion_radius_smooth: float
    exclusion_radius_heatmap: float
    exclusion_radius_ib: float
    exclusion_auto_factor_arcs: float
    exclusion_auto_factor_smooth: float
    exclusion_auto_factor_heatmap: float
    exclusion_auto_factor_ib: float

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

    # ---- small-IB spring boost ----
    use_small_ib_boost: bool
    small_ib_threshold: int
    small_ib_spring_multiplier: float

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
        self.data_segment_heatmap = ""

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
        self.ib_workers = 1
        self.mc_backend = "numba"
        self.mc_backend_apply_to_smooth = True
        self.mc_backend_apply_to_arcs = False
        self.mc_backend_apply_to_heatmap = False

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
        # Per-level auto factor: used only when the matching radius is 0.0.
        # 0.5 means "EV kicks in once beads get closer than half the typical
        # bond distance at this level".
        self.exclusion_auto_factor_arcs = 0.5
        self.exclusion_auto_factor_smooth = 0.5
        self.exclusion_auto_factor_heatmap = 0.5
        self.exclusion_auto_factor_ib = 0.5

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

        # ---- small-IB spring boost ----
        # Multiplies spring constants when reconstructing IBs with few anchors,
        # to keep loosely-constrained chains from stretching out of the model.
        self.use_small_ib_boost = False
        self.small_ib_threshold = 10  # IBs with anchors < this are "small"
        self.small_ib_spring_multiplier = 5.0

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
        cfg = configparser.ConfigParser()
        cfg.read(path)

        def get(section: str, key: str) -> str | None:
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

        # [main]
        self.output_level = geti("main", "output_level", self.output_level)
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
        # noise_arcs intentionally ignored (see Settings class comment).
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
        self.data_segment_heatmap = gets("data", "segment_heatmap", self.data_segment_heatmap)

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
        self.ib_workers = geti("simulation_backend", "ib_workers", self.ib_workers)
        self.mc_backend = gets("simulation_backend", "mc_backend", self.mc_backend)
        self.mc_backend_apply_to_smooth = getb(
            "simulation_backend", "mc_backend_apply_to_smooth", self.mc_backend_apply_to_smooth
        )
        self.mc_backend_apply_to_arcs = getb(
            "simulation_backend", "mc_backend_apply_to_arcs", self.mc_backend_apply_to_arcs
        )
        self.mc_backend_apply_to_heatmap = getb(
            "simulation_backend", "mc_backend_apply_to_heatmap", self.mc_backend_apply_to_heatmap
        )

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

        # [simulation_ib]
        self.use_ib_mc = getb("simulation_ib", "use_ib_mc", self.use_ib_mc)
        self.max_temp_ib = getf("simulation_ib", "max_temp", self.max_temp_ib)
        self.dt_temp_ib = getf("simulation_ib", "delta_temp", self.dt_temp_ib)
        self.jump_scale_ib = getf("simulation_ib", "jump_temp_scale", self.jump_scale_ib)
        self.jump_coef_ib = getf("simulation_ib", "jump_temp_coef", self.jump_coef_ib)
        self.mc_stop_steps_ib = geti("simulation_ib", "stop_condition_steps", self.mc_stop_steps_ib)
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

        # [small_ib_boost]
        self.use_small_ib_boost = getb(
            "small_ib_boost", "use_small_ib_boost", self.use_small_ib_boost
        )
        self.small_ib_threshold = geti("small_ib_boost", "threshold", self.small_ib_threshold)
        self.small_ib_spring_multiplier = getf(
            "small_ib_boost", "spring_multiplier", self.small_ib_spring_multiplier
        )

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

        return True

    def genomic_length_to_distance(self, length_bp: int) -> float:
        from .util import genomic_length_to_distance

        return genomic_length_to_distance(
            length_bp, self.genomic_dist_base, self.genomic_dist_scale, self.genomic_dist_power
        )

    def freq_to_dist_heatmap(self, freq: float) -> float:
        from .util import freq_to_dist_heatmap

        return freq_to_dist_heatmap(freq, self.freq_dist_scale, self.freq_dist_power)

    def freq_to_dist_heatmap_inter(self, freq: float) -> float:
        from .util import freq_to_dist_heatmap_inter

        return freq_to_dist_heatmap_inter(
            freq, self.freq_dist_scale_inter, self.freq_dist_power_inter
        )

    def freq_to_distance(self, freq: int) -> float:
        from .util import freq_to_distance

        return freq_to_distance(
            freq,
            self.count_dist_a,
            self.count_dist_scale,
            self.count_dist_shift,
            self.count_dist_base_level,
        )

    def data_path(self, filename: str) -> str:
        """Resolve a data filename relative to data_dir."""
        if not filename:
            return ""
        p = Path(self.data_dir) / filename
        return str(p)
