"""The settings of a run. `SETTINGS.md` documents every key."""

from __future__ import annotations

import configparser
import difflib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from gnome3d import log

if TYPE_CHECKING:
    from gnome3d.polymer import PolymerLaw

LOG = log.get("settings")


def _all_cores() -> int:
    """Returns the usable CPU count, which follows cgroup and affinity limits on Linux."""
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
    data_factors: str
    factor_strength: str
    data_singletons: str
    data_singletons_inter: str
    data_centromeres: str
    data_segment_split: str
    ib_refine_scope: str
    data_compartments: str
    data_phasing_track: str
    data_anchor_activity: str
    data_anchor_contacts: str
    data_contact_map: str
    data_components: str

    # ---- template ----

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
    subanchor_heat_min_reduction: float
    heatmap_workers: int

    # ---- PET / arc length limits ----
    max_pet_length: int
    long_pet_power: float
    long_pet_scale: float

    # ---- heatmap parameters ----
    heatmap_inter_scaling: float
    heatmap_distance_stretching: float

    # ---- distance conversion ----

    # ---- spring constants ----
    spring_stretch: float
    spring_squeeze: float
    spring_angular: float
    spring_stretch_arcs: float
    arc_weight_exponent: float
    background_weight: float
    loop_dropout: bool
    loop_dropout_scale: float
    loci_pairs: str
    loop_contact: bool
    factory_weight: float
    gyration_weight: float
    gyration_min_bp: int
    arcs_repulsion: bool
    factory_radius: float
    background_range_bp: int
    spring_squeeze_arcs: float
    use_contact_background: bool
    contact_background_z: float
    contact_resolution: int

    # ---- simulation steps ----
    steps_lvl1: int
    steps_lvl2: int
    steps_arcs: int
    steps_smooth: int
    smooth_hard_wall: bool
    smooth_anchor_cap: float
    smooth_start: str
    smooth_prefetch: int
    smooth_jax_grid: bool
    smooth_jax_grid_min_beads: int

    # ---- noise coefficients ----
    noise_lvl1: float
    noise_lvl2: float
    noise_smooth: float

    # ---- MC heatmap ----
    max_temp_heatmap: float
    dt_temp_heatmap: float
    jump_scale_heatmap: float
    jump_coef_heatmap: float
    mc_stop_improvement_heatmap: float
    mc_stop_successes_heatmap: int
    mc_stop_steps_heatmap: int

    # ---- MC parallelism ----
    mc_heatmap_chains: int
    mc_smooth_chains: int
    mc_executor_threaded_workers: int
    mc_executor_arcs: str
    mc_executor_densify: str
    mc_executor_estimate_dist: str
    mc_executor_smooth: str
    mc_executor_jax_bucket_shapes: bool
    merge_smooth_launches: bool
    mc_executor_jax_batch_width_smooth: str
    mc_executor_jax_batch_width_arcs: str

    mc_multigpu_mode: str

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
    mc_neighbour_grid: bool
    exclusion_radius_arcs: float
    exclusion_arcs_chain_radius: str
    arcs_repulsion_cutoff_factor: float
    exclusion_radius_smooth: float
    exclusion_radius_heatmap: float
    exclusion_radius_ib: float
    exclusion_auto_factor_arcs: float
    exclusion_auto_factor_smooth: float
    exclusion_auto_factor_heatmap: float
    exclusion_auto_factor_ib: float

    # ---- IB-level MC pass (chain bonds + EV between IB centroids) ----
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
    heatmap_weight_ib: float
    noise_ib: float

    # ---- confinement ----
    use_confinement: bool
    confinement_weight: float
    confinement_weight_arcs: float
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

    # ---- cross block relaxation ----
    use_cross_block_relax: bool
    relax_ev_weight: float
    relax_ev_radius: float
    relax_temp: float
    relax_noise: float
    relax_bond_weight: float
    relax_within_blocks: bool

    # ---- holes between blocks ----
    use_hole_beads: bool
    holes_hard_wall: bool
    holes_flank_beads: int

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

    # ---- overlapping-anchor handling (densification) ----
    overlap_anchor_strict: bool
    drop_zero_length_subanchors: bool
    uniform_chain: bool

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
    polymer_exponent: float
    contact_half_saturation: float
    polymer: PolymerLaw | None
    arcs_solver: str
    arcs_solver_iters: int
    arcs_solver_tol: float
    arcs_start: str
    arcs_scope: str
    mc_stop_ratio_arcs: float
    smooth_dist_weight: float
    smooth_angle_weight: float

    def __init__(self) -> None:
        self._set_defaults()

    def _set_defaults(self) -> None:
        # ---- output / misc ----
        self.output_level = 0
        self.log_file = ""
        self.random_walk = False
        self.use_2d = False
        self.loop_density = 5

        # ---- data paths ----
        self.data_dir = ""
        self.data_anchors = ""
        self.data_pet_clusters = ""
        self.data_factors = ""
        self.factor_strength = ""
        self.data_singletons = ""
        self.data_singletons_inter = ""
        self.data_centromeres = ""
        self.data_segment_split = ""
        self.ib_refine_scope = "segment"
        self.data_compartments = ""
        self.data_phasing_track = ""
        self.data_anchor_activity = ""
        self.data_anchor_contacts = ""
        self.data_contact_map = ""
        self.data_components = ""

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
        self.subanchor_heat_min_reduction = 0.0
        self.heatmap_workers = 1

        # ---- PET / arc length limits ----
        self.max_pet_length = 1_000_000
        self.long_pet_power = 2.0
        self.long_pet_scale = 10.0

        # ---- heatmap parameters ----
        self.heatmap_inter_scaling = 1.0
        self.heatmap_distance_stretching = 2.0

        # ---- spring constants ----
        self.spring_stretch = 0.1
        self.spring_squeeze = 0.1
        self.spring_angular = 0.1
        self.spring_stretch_arcs = 1.0
        self.arc_weight_exponent = 0.0
        self.loop_dropout = False
        self.loop_dropout_scale = 1.0
        self.loci_pairs = "all"
        self.loop_contact = False
        self.factory_weight = 0.0
        self.gyration_weight = 0.0
        self.gyration_min_bp = 0
        self.arcs_repulsion = True
        self.factory_radius = 2.0
        self.background_weight = 0.0
        self.background_range_bp = 100_000
        self.spring_squeeze_arcs = 1.0
        self.use_contact_background = False
        self.contact_background_z = 0.0
        self.contact_resolution = 5000

        # ---- simulation steps ----
        self.steps_lvl1 = 2
        self.steps_lvl2 = 2
        self.steps_arcs = 5
        self.steps_smooth = 5

        # ---- noise coefficients ----
        self.noise_lvl1 = 1.0
        self.noise_lvl2 = 0.1
        self.noise_smooth = 0.5
        self.smooth_hard_wall = False
        self.smooth_anchor_cap = 0.0
        self.smooth_start = "line"
        self.smooth_prefetch = 1
        self.smooth_jax_grid = False
        self.smooth_jax_grid_min_beads = 4096

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
        self.merge_smooth_launches = True
        self.mc_executor_jax_batch_width_smooth = "auto"
        self.mc_executor_jax_batch_width_arcs = "auto"
        self.mc_multigpu_mode = "groups"

        # ---- MC arcs ----
        self.max_temp = 20.0
        self.dt_temp = 0.99995
        self.jump_scale = 50.0
        self.jump_coef = 20.0
        self.mc_stop_improvement = 0.995
        self.mc_stop_successes = 5
        self.mc_stop_steps = 10000

        # ---- excluded volume ----
        self.use_excluded_volume = False
        self.exclusion_weight = 0.5
        self.exclusion_apply_to_arcs = False
        self.exclusion_apply_to_smooth = True
        self.exclusion_apply_to_heatmap = False
        self.exclusion_apply_to_ib = True
        self.exclusion_skip_neighbors = 1
        self.mc_neighbour_grid = True
        self.exclusion_radius_arcs = 0.0
        self.exclusion_arcs_chain_radius = "none"
        self.exclusion_radius_smooth = 0.0
        self.exclusion_radius_heatmap = 0.0
        self.exclusion_radius_ib = 0.0
        self.arcs_repulsion_cutoff_factor = 0.0
        self.exclusion_auto_factor_arcs = 0.5
        self.exclusion_auto_factor_smooth = 0.5
        self.exclusion_auto_factor_heatmap = 0.5
        self.exclusion_auto_factor_ib = 0.5

        # ---- IB-level MC pass ----
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
        self.heatmap_weight_ib = 0.0
        self.noise_ib = 0.5

        # ---- confinement ----
        self.use_confinement = False
        self.confinement_weight = 0.5
        self.confinement_weight_arcs = 0.0
        self.confinement_apply_to_arcs = True
        self.confinement_apply_to_smooth = True
        self.confinement_apply_to_ib = True
        self.confinement_radius_arcs = 0.0
        self.confinement_radius_smooth = 0.0
        self.confinement_radius_ib = 0.0
        self.confinement_packing_factor_arcs = 1.5
        self.confinement_packing_factor_smooth = 1.5
        self.confinement_packing_factor_ib = 0.75

        # ---- boundary stitch ----
        self.use_boundary_stitch = False
        self.boundary_stitch_spring_weight = 1.0
        self.boundary_stitch_ev_weight = 1.0
        self.boundary_stitch_max_iter = 2000

        # ---- cross block relaxation ----
        self.use_cross_block_relax = False
        self.relax_ev_weight = 10.0
        self.relax_ev_radius = 0.0
        self.relax_temp = 0.1
        self.relax_noise = 0.5
        self.relax_bond_weight = 10.0
        self.relax_within_blocks = False
        self.relax_local_window = -1

        # ---- holes between blocks ----
        self.use_hole_beads = False
        self.holes_hard_wall = True
        self.holes_flank_beads = 0

        # ---- A/B compartments ----
        self.use_compartments = False
        self.compartment_weight = 1.0
        self.compartment_energy_a = 1.0
        self.compartment_energy_b = 2.0
        self.compartment_apply_to_heatmap = True
        self.compartment_apply_to_ib = True
        self.compartment_apply_to_smooth = True
        self.compartment_radius_heatmap = 0.0
        self.compartment_radius_ib = 0.0
        self.compartment_radius_smooth = 0.0
        self.compartment_auto_factor_heatmap = 1.5
        self.compartment_auto_factor_ib = 1.5
        self.compartment_auto_factor_smooth = 1.5

        # ---- overlapping-anchor handling ----
        self.overlap_anchor_strict = False
        self.drop_zero_length_subanchors = False
        self.uniform_chain = False

        # ---- dynamic loop density ----
        self.use_dynamic_loop_density = False
        self.target_bp_per_subanchor = 5000
        self.min_subanchors_per_arc = 0
        self.max_subanchors_per_arc = 50

        # ---- MC smooth ----
        self.max_temp_smooth = 20.0
        self.dt_temp_smooth = 0.99995
        self.jump_scale_smooth = 50.0
        self.jump_coef_smooth = 20.0
        self.mc_stop_improvement_smooth = 0.995
        self.mc_stop_successes_smooth = 5
        self.mc_stop_steps_smooth = 10000
        self.polymer_exponent = 0.0
        self.contact_half_saturation = 1.0
        self.polymer = None
        self.arcs_solver = "mc"
        self.arcs_solver_iters = 200
        self.arcs_solver_tol = 0.0
        self.arcs_start = "centroid"
        self.arcs_scope = "block"
        self.mc_stop_ratio_arcs = 0.9999
        self.smooth_dist_weight = 1.0
        self.smooth_angle_weight = 1.0

    def load_ini(self, path: str) -> bool:
        """Loads settings from an .ini file over the current values."""
        cfg = configparser.ConfigParser()
        cfg.read(path)
        return self._load_from_parser(cfg)

    @classmethod
    def from_dict(cls, config: Mapping[str, Mapping[str, object]]) -> Settings:
        """Builds a Settings from a nested `{section: {key: value}}` mapping with the section and
        key names of the .ini files.

            Settings.from_dict({
                "data": {"data_dir": "data/GM12878/", "anchors": "...bed"},
                "excluded_volume": {"use_excluded_volume": True},
            })

        A key that is absent keeps its default. Values are parsed as `load_ini` parses them.
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
        """Applies the settings of a ConfigParser over the current values."""
        # Every (section, key) the loader reads, in the spelling of ConfigParser, for the unknown
        # key check.
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
            """Returns a worker count, every usable core for `auto`."""
            v = get(section, key)
            if v is None:
                return default
            v = v.strip().lower()
            return _all_cores() if v == "auto" else int(v)

        def ignore(section: str, key: str) -> None:
            """Declares a key that is accepted and not read. The unknown key check passes it."""
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
        ignore("main", "noise_arcs")
        self.noise_smooth = getf("main", "noise_smooth", self.noise_smooth)
        self.noise_ib = getf("main", "noise_ib", self.noise_ib)

        # [data]
        self.data_dir = gets("data", "data_dir", self.data_dir)
        self.data_anchors = gets("data", "anchors", self.data_anchors)
        self.data_pet_clusters = gets("data", "clusters", self.data_pet_clusters)
        self.data_factors = gets("data", "factors", self.data_factors)
        self.factor_strength = gets("springs", "factor_strength", self.factor_strength)
        self.data_singletons = gets("data", "singletons", self.data_singletons)
        self.data_singletons_inter = gets("data", "singletons_inter", self.data_singletons_inter)
        self.data_centromeres = gets("data", "centromeres", self.data_centromeres)
        self.data_segment_split = gets("data", "segment_split", self.data_segment_split)
        self.ib_refine_scope = gets("simulation_ib", "refine_scope", self.ib_refine_scope)
        self.data_compartments = gets("data", "compartments", self.data_compartments)
        self.data_phasing_track = gets("data", "phasing_track", self.data_phasing_track)
        self.data_anchor_activity = gets("data", "anchor_activity", self.data_anchor_activity)
        self.data_anchor_contacts = gets("data", "anchor_contacts", self.data_anchor_contacts)
        self.data_contact_map = gets("data", "contact_map", self.data_contact_map)
        self.data_components = gets("data", "components", self.data_components)

        # [template]

        # [distance]

        # [heatmaps]
        self.heatmap_inter_scaling = getf("heatmaps", "inter_scaling", self.heatmap_inter_scaling)
        self.heatmap_distance_stretching = getf(
            "heatmaps", "distance_heatmap_stretching", self.heatmap_distance_stretching
        )

        # [springs]
        self.spring_stretch = getf("springs", "stretch_constant", self.spring_stretch)
        self.arc_weight_exponent = getf("springs", "arc_weight_exponent", self.arc_weight_exponent)
        self.loop_dropout = getb("springs", "loop_dropout", self.loop_dropout)
        self.loop_dropout_scale = getf("springs", "loop_dropout_scale", self.loop_dropout_scale)
        self.loci_pairs = gets("springs", "loci_pairs", self.loci_pairs)
        if self.loci_pairs not in ("all", "near", "octave", "near_octave", "segment"):
            raise ValueError(
                "[springs] loci_pairs must be all, near, octave, near_octave or segment, got"
                f" {self.loci_pairs!r}"
            )
        self.loop_contact = getb("springs", "loop_contact", self.loop_contact)
        self.factory_weight = getf("factories", "weight", self.factory_weight)
        self.factory_radius = getf("factories", "radius", self.factory_radius)
        self.spring_squeeze = getf("springs", "squeeze_constant", self.spring_squeeze)
        self.spring_angular = getf("springs", "angular_constant", self.spring_angular)
        self.spring_stretch_arcs = getf(
            "springs", "stretch_constant_arcs", self.spring_stretch_arcs
        )
        self.background_weight = getf("springs", "background_weight", self.background_weight)
        self.gyration_weight = getf("springs", "gyration_weight", self.gyration_weight)
        self.gyration_min_bp = geti("springs", "gyration_min_bp", self.gyration_min_bp)
        self.background_range_bp = geti("springs", "background_range_bp", self.background_range_bp)
        self.spring_squeeze_arcs = getf(
            "springs", "squeeze_constant_arcs", self.spring_squeeze_arcs
        )
        self.use_contact_background = getb(
            "springs", "use_contact_background", self.use_contact_background
        )
        self.contact_background_z = getf(
            "springs", "contact_background_z", self.contact_background_z
        )
        self.contact_resolution = geti("springs", "contact_resolution", self.contact_resolution)
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
        self.merge_smooth_launches = getb(
            "simulation_backend", "merge_smooth_launches", self.merge_smooth_launches
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
        self.mc_neighbour_grid = getb(
            "simulation_backend", "neighbour_grid", self.mc_neighbour_grid
        )
        self.exclusion_radius_arcs = getf(
            "excluded_volume", "radius_arcs", self.exclusion_radius_arcs
        )
        self.exclusion_arcs_chain_radius = gets(
            "excluded_volume", "arcs_chain_radius", self.exclusion_arcs_chain_radius
        )
        if self.exclusion_arcs_chain_radius not in ("none", "sum"):
            raise ValueError(
                "[excluded_volume] arcs_chain_radius must be none or sum, got"
                f" {self.exclusion_arcs_chain_radius!r}"
            )
        self.arcs_repulsion_cutoff_factor = getf(
            "excluded_volume", "arcs_repulsion_cutoff_factor", self.arcs_repulsion_cutoff_factor
        )
        self.arcs_repulsion = getb("excluded_volume", "arcs_repulsion", self.arcs_repulsion)
        self.exclusion_radius_smooth = getf(
            "excluded_volume", "radius_smooth", self.exclusion_radius_smooth
        )
        self.exclusion_radius_heatmap = getf(
            "excluded_volume", "radius_heatmap", self.exclusion_radius_heatmap
        )
        self.exclusion_radius_ib = getf("excluded_volume", "radius_ib", self.exclusion_radius_ib)
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
        self.heatmap_weight_ib = getf("simulation_ib", "heatmap_weight", self.heatmap_weight_ib)

        # [confinement]
        self.use_confinement = getb("confinement", "use_confinement", self.use_confinement)
        self.confinement_weight = getf("confinement", "weight", self.confinement_weight)
        self.confinement_weight_arcs = getf(
            "confinement", "weight_arcs", self.confinement_weight_arcs
        )
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

        # [relax]
        self.use_cross_block_relax = getb(
            "relax", "use_cross_block_relax", self.use_cross_block_relax
        )
        self.relax_ev_weight = getf("relax", "ev_weight", self.relax_ev_weight)
        self.relax_ev_radius = getf("relax", "ev_radius", self.relax_ev_radius)
        self.relax_temp = getf("relax", "temp", self.relax_temp)
        self.relax_noise = getf("relax", "noise", self.relax_noise)
        self.relax_bond_weight = getf("relax", "bond_weight", self.relax_bond_weight)
        self.relax_local_window = geti("relax", "local_window", self.relax_local_window)
        self.relax_within_blocks = getb("relax", "within_blocks", self.relax_within_blocks)

        # [holes]
        self.use_hole_beads = getb("holes", "use_hole_beads", self.use_hole_beads)
        self.holes_hard_wall = getb("holes", "hard_wall", self.holes_hard_wall)
        self.holes_flank_beads = geti("holes", "flank_beads", self.holes_flank_beads)

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

        # [main]
        self.overlap_anchor_strict = getb(
            "main", "overlap_anchor_strict", self.overlap_anchor_strict
        )
        self.drop_zero_length_subanchors = getb(
            "main", "drop_zero_length_subanchors", self.drop_zero_length_subanchors
        )
        self.uniform_chain = getb("main", "uniform_chain", self.uniform_chain)

        # [main]
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
        self.smooth_hard_wall = getb("simulation_arcs_smooth", "hard_wall", self.smooth_hard_wall)
        self.smooth_anchor_cap = getf(
            "simulation_arcs_smooth", "anchor_cap", self.smooth_anchor_cap
        )
        if self.uniform_chain and self.smooth_anchor_cap > 0.0:
            raise ValueError(
                "[main] uniform_chain is on with [simulation_arcs_smooth] anchor_cap above zero."
                " A node of the uniform chain moves with its two neighbours, which is one rule"
                " for the fixed beads of the smooth stage, and anchor_cap is another. Set"
                " anchor_cap to 0."
            )
        if self.loci_pairs != "all" and not self.uniform_chain:
            raise ValueError(
                "[springs] loci_pairs is set and [main] uniform_chain is off. The law between"
                " the loci is a rule of the uniform chain. Set loci_pairs to all or"
                " uniform_chain to yes."
            )
        if self.exclusion_arcs_chain_radius != "none" and not self.uniform_chain:
            raise ValueError(
                "[excluded_volume] arcs_chain_radius is set and [main] uniform_chain is off. The"
                " radius from the chain is stated on the nodes of the uniform chain. Set"
                " arcs_chain_radius to none or uniform_chain to yes."
            )
        self.smooth_start = gets("simulation_arcs_smooth", "start", self.smooth_start)
        self.smooth_prefetch = geti("simulation_arcs_smooth", "prefetch", self.smooth_prefetch)
        if self.smooth_prefetch < 1:
            raise ValueError("[simulation_arcs_smooth] prefetch must be at least 1")
        self.smooth_jax_grid = getb("simulation_arcs_smooth", "jax_grid", self.smooth_jax_grid)
        self.smooth_jax_grid_min_beads = geti(
            "simulation_arcs_smooth", "jax_grid_min_beads", self.smooth_jax_grid_min_beads
        )
        if self.smooth_start not in ("line", "coil", "maxent"):
            raise ValueError(
                "[simulation_arcs_smooth] start must be line, coil or maxent,"
                f" got {self.smooth_start!r}"
            )
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
        self.mc_stop_ratio_arcs = getf(
            "simulation_arcs", "stop_condition_ratio", self.mc_stop_ratio_arcs
        )
        self.polymer_exponent = getf("distance", "polymer_exponent", self.polymer_exponent)
        self.contact_half_saturation = getf(
            "distance", "contact_half_saturation", self.contact_half_saturation
        )
        self.arcs_solver = gets("simulation_arcs", "solver", self.arcs_solver)
        self.arcs_solver_iters = geti("simulation_arcs", "solver_iters", self.arcs_solver_iters)
        self.arcs_solver_tol = getf("simulation_arcs", "solver_tol", self.arcs_solver_tol)
        self.arcs_start = gets("simulation_arcs", "start", self.arcs_start)
        self.arcs_scope = gets("simulation_arcs", "scope", self.arcs_scope)
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
        """Warns about keys in the file that the loader does not read, such as a misspelt or
        renamed key or a key under the wrong section. Only sections the loader reads are checked.
        """
        for section in cfg.sections():
            known = consulted.get(section)
            if not known:
                continue
            for key in cfg.options(section):
                if key in known:
                    continue
                near = difflib.get_close_matches(key, known, n=1, cutoff=0.6)
                hint = f" (did you mean '{near[0]}'?)" if near else ""
                LOG.warning("[%s] unknown key '%s' is ignored%s", section, key, hint)

    def polymer_law(self) -> PolymerLaw:
        """Returns the distance law of the run. `build_state` attaches the fitted one. Before that
        a default is built from `polymer_exponent` or the fallback, with one warning.
        """
        if self.polymer is None:
            from gnome3d.polymer import FALLBACK_NU, PolymerLaw

            nu = float(self.polymer_exponent) if self.polymer_exponent > 0.0 else FALLBACK_NU
            log.get("settings").warning(
                "no fitted polymer law attached; using exponent %.3f %s",
                nu,
                "from the config" if self.polymer_exponent > 0.0 else "as the fallback",
            )
            self.polymer = PolymerLaw(
                nu=nu,
                s0_bp=int(self.target_bp_per_subanchor),
                q_half=float(self.contact_half_saturation),
            )
        return self.polymer

    def cluster_files(self) -> list[tuple[str, int, str]]:
        """Returns the cluster files as (path, factor index, factor name).

        `[data] clusters` holds one or more paths, comma separated and `factors` one name each.
        Factor 0 is CTCF, the set the orientation term reads.
        """
        paths = [p.strip() for p in str(self.data_pet_clusters).split(",") if p.strip()]
        names = [n.strip() for n in str(self.data_factors).split(",") if n.strip()]
        if names and len(names) != len(paths):
            raise ValueError(
                f"[data] factors names {len(names)} sets for {len(paths)} cluster files"
            )
        if not names:
            names = ["CTCF"] + [f"factor{i}" for i in range(1, len(paths))]
        return [(self.data_path(p), i, names[i]) for i, p in enumerate(paths)]

    def factor_strengths(self) -> dict[int, float]:
        """Returns the multiplier on loop strength per factor index from
        `[springs] factor_strength`, one value per cluster file. A factor without a value has 1.
        """
        vals = [v.strip() for v in str(self.factor_strength).split(",") if v.strip()]
        return {i: float(v) for i, v in enumerate(vals)}

    def genomic_length_to_distance(self, length_bp: int) -> float:
        """Returns the distance of two beads `sep_bp` apart with no contact, in beads."""
        return self.polymer_law().background(length_bp)

    def arc_expected_distance(self, score: int, sep_bp: int, factor: int = 0) -> float:
        """Returns the target of an arc of `score` PETs that spans `sep_bp`, in beads, by the
        strength fit of its factor.
        """
        return self.polymer_law().arc_distance(score, sep_bp, factor)

    def data_path(self, filename: str) -> str:
        """Returns the path of a data file under `data_dir`."""
        if not filename:
            return ""
        p = Path(self.data_dir) / filename
        return str(p)
