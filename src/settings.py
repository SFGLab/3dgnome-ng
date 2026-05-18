"""
src/settings.py - Configuration for 3dgnome-ng.

Mirrors C++ Settings class.  All defaults match Settings::init() in Settings.cpp.
"""

import configparser
from pathlib import Path


class Settings:
    def __init__(self):
        self._set_defaults()

    def _set_defaults(self):
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
        self.noise_arcs = 0.5
        self.noise_smooth = 0.5

        # ---- MC heatmap ----
        self.max_temp_heatmap = 20.0
        self.dt_temp_heatmap = 0.99995
        self.jump_scale_heatmap = 50.0
        self.jump_coef_heatmap = 20.0
        self.mc_stop_improvement_heatmap = 0.995
        self.mc_stop_successes_heatmap = 5
        self.mc_stop_steps_heatmap = 10000

        # ---- MC arcs ----
        self.max_temp = 20.0
        self.dt_temp = 0.99995
        self.jump_scale = 50.0
        self.jump_coef = 20.0
        self.mc_stop_improvement = 0.995
        self.mc_stop_successes = 5
        self.mc_stop_steps = 10000

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

        def get(section, key, default):
            try:
                return cfg.get(section, key)
            except (configparser.NoSectionError, configparser.NoOptionError):
                return None

        def geti(section, key, default):
            v = get(section, key, default)
            return int(v) if v is not None else default

        def getf(section, key, default):
            v = get(section, key, default)
            return float(v) if v is not None else default

        def getb(section, key, default):
            v = get(section, key, default)
            if v is None:
                return default
            return v.strip().lower() in ("yes", "true", "1")

        def gets(section, key, default):
            v = get(section, key, default)
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
        self.noise_arcs = getf("main", "noise_arcs", self.noise_arcs)
        self.noise_smooth = getf("main", "noise_smooth", self.noise_smooth)

        # [data]
        self.data_dir = gets("data", "data_dir", self.data_dir)
        self.data_anchors = gets("data", "anchors", self.data_anchors)
        self.data_pet_clusters = gets("data", "clusters", self.data_pet_clusters)
        self.data_singletons = gets("data", "singletons", self.data_singletons)
        self.data_singletons_inter = gets("data", "singletons_inter", self.data_singletons_inter)
        self.data_factors = gets("data", "factors", self.data_factors)
        self.data_split_singletons_by_chr = getb("data", "split_singleton_files_by_chr",
                                                 self.data_split_singletons_by_chr)
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
        self.freq_dist_scale_inter = getf("distance", "freq_dist_scale_inter", self.freq_dist_scale_inter)
        self.freq_dist_power_inter = getf("distance", "freq_dist_power_inter", self.freq_dist_power_inter)
        self.count_dist_a = getf("distance", "count_dist_a", self.count_dist_a)
        self.count_dist_scale = getf("distance", "count_dist_scale", self.count_dist_scale)
        self.count_dist_shift = getf("distance", "count_dist_shift", self.count_dist_shift)
        self.count_dist_base_level = getf("distance", "count_dist_base_level", self.count_dist_base_level)

        # [heatmaps]
        self.heatmap_inter_scaling = getf("heatmaps", "inter_scaling", self.heatmap_inter_scaling)
        self.heatmap_distance_stretching = getf("heatmaps", "distance_heatmap_stretching",
                                                self.heatmap_distance_stretching)

        # [springs]
        self.spring_stretch = getf("springs", "stretch_constant", self.spring_stretch)
        self.spring_squeeze = getf("springs", "squeeze_constant", self.spring_squeeze)
        self.spring_angular = getf("springs", "angular_constant", self.spring_angular)
        self.spring_stretch_arcs = getf("springs", "stretch_constant_arcs", self.spring_stretch_arcs)
        self.spring_squeeze_arcs = getf("springs", "squeeze_constant_arcs", self.spring_squeeze_arcs)

        # [motif_orientation]
        self.use_ctcf_motif = getb("motif_orientation", "use_motif_orientation", self.use_ctcf_motif)
        self.motif_weight = getf("motif_orientation", "weight", self.motif_weight)
        self.motifs_symmetric = getb("motif_orientation", "symmetric_motifs", self.motifs_symmetric)

        # [anchor_heatmap]
        self.use_anchor_heatmap = getb("anchor_heatmap", "use_anchor_heatmap", self.use_anchor_heatmap)
        self.anchor_heatmap_influence = getf("anchor_heatmap", "heatmap_influence", self.anchor_heatmap_influence)

        # [subanchor_heatmap]
        self.use_subanchor_heatmap = getb("subanchor_heatmap", "use_subanchor_heatmap", self.use_subanchor_heatmap)
        self.subanchor_heatmap_influence = getf("subanchor_heatmap", "heatmap_influence",
                                                self.subanchor_heatmap_influence)
        self.subanchor_heatmap_dist_weight = getf("subanchor_heatmap", "heatmap_dist_weight",
                                                  self.subanchor_heatmap_dist_weight)
        self.subanchor_estimate_steps = geti("subanchor_heatmap", "estimate_distances_steps",
                                             self.subanchor_estimate_steps)
        self.subanchor_estimate_replicates = geti("subanchor_heatmap", "estimate_distances_replicates",
                                                  self.subanchor_estimate_replicates)

        # [simulation_heatmap]
        self.max_temp_heatmap = getf("simulation_heatmap", "max_temp_heatmap", self.max_temp_heatmap)
        self.dt_temp_heatmap = getf("simulation_heatmap", "delta_temp_heatmap", self.dt_temp_heatmap)
        self.jump_scale_heatmap = getf("simulation_heatmap", "jump_temp_scale_heatmap", self.jump_scale_heatmap)
        self.jump_coef_heatmap = getf("simulation_heatmap", "jump_temp_coef_heatmap", self.jump_coef_heatmap)
        self.mc_stop_steps_heatmap = geti("simulation_heatmap", "stop_condition_steps_heatmap",
                                          self.mc_stop_steps_heatmap)
        self.mc_stop_improvement_heatmap = getf("simulation_heatmap", "stop_condition_improvement_threshold_heatmap",
                                                self.mc_stop_improvement_heatmap)
        self.mc_stop_successes_heatmap = geti("simulation_heatmap", "stop_condition_successes_threshold_heatmap",
                                              self.mc_stop_successes_heatmap)

        # [simulation_arcs]
        self.max_temp = getf("simulation_arcs", "max_temp", self.max_temp)
        self.dt_temp = getf("simulation_arcs", "delta_temp", self.dt_temp)
        self.jump_scale = getf("simulation_arcs", "jump_temp_scale", self.jump_scale)
        self.jump_coef = getf("simulation_arcs", "jump_temp_coef", self.jump_coef)
        self.mc_stop_steps = geti("simulation_arcs", "stop_condition_steps", self.mc_stop_steps)
        self.mc_stop_improvement = getf("simulation_arcs", "stop_condition_improvement_threshold",
                                        self.mc_stop_improvement)
        self.mc_stop_successes = geti("simulation_arcs", "stop_condition_successes_threshold", self.mc_stop_successes)

        # [simulation_arcs_smooth]
        self.smooth_dist_weight = getf("simulation_arcs_smooth", "dist_weight", self.smooth_dist_weight)
        self.smooth_angle_weight = getf("simulation_arcs_smooth", "angle_weight", self.smooth_angle_weight)
        self.max_temp_smooth = getf("simulation_arcs_smooth", "max_temp", self.max_temp_smooth)
        self.dt_temp_smooth = getf("simulation_arcs_smooth", "delta_temp", self.dt_temp_smooth)
        self.jump_scale_smooth = getf("simulation_arcs_smooth", "jump_temp_scale", self.jump_scale_smooth)
        self.jump_coef_smooth = getf("simulation_arcs_smooth", "jump_temp_coef", self.jump_coef_smooth)
        self.mc_stop_steps_smooth = geti("simulation_arcs_smooth", "stop_condition_steps", self.mc_stop_steps_smooth)
        self.mc_stop_improvement_smooth = getf("simulation_arcs_smooth", "stop_condition_improvement_threshold",
                                               self.mc_stop_improvement_smooth)
        self.mc_stop_successes_smooth = geti("simulation_arcs_smooth", "stop_condition_successes_threshold",
                                             self.mc_stop_successes_smooth)

        return True

    def genomic_length_to_distance(self, length_bp: int) -> float:
        from .energy import genomic_length_to_distance
        return genomic_length_to_distance(length_bp, self.genomic_dist_base,
                                          self.genomic_dist_scale, self.genomic_dist_power)

    def freq_to_dist_heatmap(self, freq: float) -> float:
        from .energy import freq_to_dist_heatmap
        return freq_to_dist_heatmap(freq, self.freq_dist_scale, self.freq_dist_power)

    def freq_to_dist_heatmap_inter(self, freq: float) -> float:
        from .energy import freq_to_dist_heatmap_inter
        return freq_to_dist_heatmap_inter(freq, self.freq_dist_scale_inter, self.freq_dist_power_inter)

    def freq_to_distance(self, freq: int) -> float:
        from .energy import freq_to_distance
        return freq_to_distance(freq, self.count_dist_a, self.count_dist_scale,
                                self.count_dist_shift, self.count_dist_base_level)

    def data_path(self, filename: str) -> str:
        """Resolve a data filename relative to data_dir."""
        if not filename:
            return ""
        p = Path(self.data_dir) / filename
        return str(p)
