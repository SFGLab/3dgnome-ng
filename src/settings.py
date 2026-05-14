"""All algorithm parameters mirroring cudaMMC `Settings.cpp`.

Every field carries a `# cudaMMC Settings.cpp:LINE` citation for the upstream
default and (where applicable) the `[section] key` parsed by `loadFromINI`
(`Settings.cpp:370-616`).  Default values match `Settings::init()`
(`Settings.cpp:142-256`) exactly.

The INI loader below mirrors `Settings::loadFromINI` key-by-key, including the
known upstream bug at `Settings.cpp:594-597` where `dist_weight` is written to
`weightAngleSmooth` and `angle_weight` to `weightDistSmooth` (swapped on
purpose – see AGENTS.md "Mirror upstream bugs verbatim").
"""

import configparser
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Settings:
    # ── [cuda] runtime ───────────────────────────────────────────────────────
    cuda_blocks_multiplier: int = 4          # cudaMMC Settings.cpp:155
    cuda_threads_per_block: int = 256        # cudaMMC Settings.cpp:156
    milestone_fails_threshold: int = 3       # cudaMMC Settings.cpp:154

    # ── [main] generic ───────────────────────────────────────────────────────
    output_level: int = 0                    # cudaMMC Settings.cpp:145
    random_walk: bool = False                # cudaMMC Settings.cpp:146
    use_input_cache: bool = True             # cudaMMC Settings.cpp:150
    use_2d: bool = False                     # cudaMMC Settings.cpp:148
    loop_density: int = 5                    # cudaMMC Settings.cpp:152

    max_pet_cluster_length: int = 1_000_000  # cudaMMC Settings.cpp:173
    long_pet_clusters_effect_power: float = 2.0   # cudaMMC Settings.cpp:174
    long_pet_clusters_effect_scale: float = 10.0  # cudaMMC Settings.cpp:175

    # ── [motif_orientation] ──────────────────────────────────────────────────
    use_ctcf_motif_orientation: bool = False     # cudaMMC Settings.cpp:158
    motifs_symmetric: bool = True                # cudaMMC Settings.cpp:159
    motif_orientation_weight: float = 1.0        # cudaMMC Settings.cpp:160

    # ── [subanchor_heatmap] ──────────────────────────────────────────────────
    use_subanchor_heatmap: bool = False                       # cudaMMC Settings.cpp:162
    subanchor_estimate_distances_replicates: int = 5          # cudaMMC Settings.cpp:163
    subanchor_estimate_distances_steps: int = 2               # cudaMMC Settings.cpp:164
    subanchor_heatmap_influence: float = 0.5                  # cudaMMC Settings.cpp:165
    subanchor_heatmap_dist_weight: float = 1.0                # cudaMMC Settings.cpp:166

    # ── [anchor_heatmap] ─────────────────────────────────────────────────────
    use_anchor_heatmap: bool = False                          # cudaMMC Settings.cpp:168
    anchor_heatmap_influence: float = 0.5                     # cudaMMC Settings.cpp:169
    anchor_heatmap_dist_weight: float = 1.0                   # cudaMMC Settings.cpp:170

    # ── [main] hierarchical structure sizes ─────────────────────────────────
    segment_size: int = 2_000_000          # cudaMMC Settings.cpp:172
    ib_random_walk_jumps: float = 10.0     # cudaMMC Settings.cpp:192

    # ── [main] simulation step counts (multi-restart) ───────────────────────
    simulation_steps_level_chr: int = 2        # cudaMMC Settings.cpp:179
    simulation_steps_level_segment: int = 2    # cudaMMC Settings.cpp:180
    simulation_steps_level_anchor: int = 5     # cudaMMC Settings.cpp:181
    simulation_steps_level_subanchor: int = 5  # cudaMMC Settings.cpp:182

    # ── [main] per-level noise coefficients ─────────────────────────────────
    noise_coefficient_level_chr: float = 1.0        # cudaMMC Settings.cpp:194
    noise_coefficient_level_segment: float = 0.1    # cudaMMC Settings.cpp:195
    noise_coefficient_level_anchor: float = 0.5     # cudaMMC Settings.cpp:196
    noise_coefficient_level_subanchor: float = 0.5  # cudaMMC Settings.cpp:197

    # ── [heatmaps] ───────────────────────────────────────────────────────────
    heatmap_inter_scaling: float = 1.0                 # cudaMMC Settings.cpp:199
    heatmap_distance_heatmap_stretching: float = 2.0   # cudaMMC Settings.cpp:200

    # ── [distance] frequency/count → distance ───────────────────────────────
    freq_dist_scale: float = 100.0          # cudaMMC Settings.cpp:202  (freqToDistHeatmapScale)
    freq_dist_power: float = -0.333         # cudaMMC Settings.cpp:203
    freq_dist_scale_inter: float = 100.0    # cudaMMC Settings.cpp:204
    freq_dist_power_inter: float = -1.0     # cudaMMC Settings.cpp:205

    count_dist_a: float = 0.5               # cudaMMC Settings.cpp:207
    count_dist_scale: float = 20.0          # cudaMMC Settings.cpp:208
    count_dist_shift: float = 1.0           # cudaMMC Settings.cpp:209
    count_dist_base_level: float = 0.01     # cudaMMC Settings.cpp:210

    genomic_dist_power: float = 0.5         # cudaMMC Settings.cpp:212
    genomic_dist_scale: float = 1.0         # cudaMMC Settings.cpp:213
    genomic_dist_base: float = 0.0          # cudaMMC Settings.cpp:214

    # ── [springs] chain & arcs spring constants (stretch ≠ squeeze) ─────────
    spring_constant_stretch: float = 0.1        # cudaMMC Settings.cpp:251
    spring_constant_squeeze: float = 0.1        # cudaMMC Settings.cpp:250
    spring_angular_constant: float = 0.1        # cudaMMC Settings.cpp:252
    spring_constant_stretch_arcs: float = 1.0   # cudaMMC Settings.cpp:255
    spring_constant_squeeze_arcs: float = 1.0   # cudaMMC Settings.cpp:254

    # ── [simulation_heatmap] MC parameters ──────────────────────────────────
    max_temp_heatmap: float = 20.0                     # cudaMMC Settings.cpp:216
    dt_temp_heatmap: float = 0.99995                   # cudaMMC Settings.cpp:217
    temp_jump_coef_heatmap: float = 20.0               # cudaMMC Settings.cpp:218
    temp_jump_scale_heatmap: float = 50.0              # cudaMMC Settings.cpp:219
    mc_stop_improvement_heatmap: float = 0.995         # cudaMMC Settings.cpp:220
    mc_stop_min_successes_heatmap: int = 5             # cudaMMC Settings.cpp:221
    mc_stop_steps_heatmap: int = 10000                 # cudaMMC Settings.cpp:222

    # ── [simulation_arcs] MC parameters ─────────────────────────────────────
    max_temp_arcs: float = 20.0              # cudaMMC Settings.cpp:232  (maxTemp)
    dt_temp_arcs: float = 0.99995            # cudaMMC Settings.cpp:233  (dtTemp)
    temp_jump_coef_arcs: float = 20.0        # cudaMMC Settings.cpp:234  (tempJumpCoef)
    temp_jump_scale_arcs: float = 50.0       # cudaMMC Settings.cpp:235  (tempJumpScale)
    mc_stop_improvement_arcs: float = 0.995  # cudaMMC Settings.cpp:236
    mc_stop_min_successes_arcs: int = 5      # cudaMMC Settings.cpp:237
    mc_stop_steps_arcs: int = 10000          # cudaMMC Settings.cpp:238

    # ── [simulation_arcs_smooth] MC parameters ──────────────────────────────
    # NOTE: cudaMMC Settings.cpp:594-597 reads INI keys with weight_dist and
    # weight_angle SWAPPED.  Defaults are 1.0/1.0 so the bug is dormant; the
    # INI loader below preserves the swap so any user override is bit-identical.
    weight_dist_smooth: float = 1.0          # cudaMMC Settings.cpp:241
    weight_angle_smooth: float = 1.0         # cudaMMC Settings.cpp:240
    max_temp_smooth: float = 20.0            # cudaMMC Settings.cpp:242
    dt_temp_smooth: float = 0.99995          # cudaMMC Settings.cpp:243
    temp_jump_coef_smooth: float = 20.0      # cudaMMC Settings.cpp:244
    temp_jump_scale_smooth: float = 50.0     # cudaMMC Settings.cpp:245
    mc_stop_improvement_smooth: float = 0.995  # cudaMMC Settings.cpp:246
    mc_stop_min_successes_smooth: int = 5    # cudaMMC Settings.cpp:247
    mc_stop_steps_smooth: int = 10000        # cudaMMC Settings.cpp:248

    # ── Python-only safety caps (not in cudaMMC) ──────────────────────────────
    # Opt-in upper bound on MC iterations / wall time per restart, mostly
    # useful when the cudaMMC stop condition (ratio > impr AND succ < min)
    # is satisfied on the ``ratio`` clause but not on the ``succ`` clause
    # (very small step sizes can trickle improvements forever).  0 = off,
    # matching cudaMMC behaviour exactly.  Setting either > 0 makes MC log
    # ``[<phase>] stopping early (iter/wall limit reached)`` and return.
    mc_max_iters_arcs: int = 0
    mc_max_iters_smooth: int = 0
    mc_max_seconds_arcs: float = 0.0
    mc_max_seconds_smooth: float = 0.0

    # ── [data] file paths ────────────────────────────────────────────────────
    data_directory: str = ""
    data_anchors: str = ""
    data_pet_clusters: str = ""
    data_singletons: str = ""
    data_singletons_inter: str = ""
    data_centromeres: str = ""
    data_segments_split: str = ""     # cudaMMC Settings.cpp:448  (segment_split)
    data_factors: str = ""
    data_segment_heatmap: str = ""

    # ── [density], [telomeres], [template] — dormant; flag-only ─────────────
    use_density: bool = False                    # cudaMMC Settings.cpp:187
    density_scale: float = 1.0                   # cudaMMC Settings.cpp:188
    density_influence: float = 0.95              # cudaMMC Settings.cpp:189
    density_weight: float = 1.0                  # cudaMMC Settings.cpp:190
    use_telomere_positions: bool = False
    template_segment: str = ""
    template_scale: float = 1.0                  # cudaMMC Settings.cpp:184
    dist_heatmap: str = ""
    dist_heatmap_scale: float = 1.0              # cudaMMC Settings.cpp:185

    # ── PET-count BEDPE parsing ─────────────────────────────────────────────
    # cudaMMC Cluster.cpp hard-codes column 7 (0-based) for PET count; legacy
    # Python file auto-detected col 6/7.  Make it explicit, default to upstream.
    bedpe_score_column: int = 7

    # ── Python-only convenience ──────────────────────────────────────────────
    device: str = "cuda"   # No cudaMMC counterpart; CUDA auto-selected upstream.

    # Debug instrumentation — Python-only.  When non-empty, ``LooperSolver``
    # dumps a CIF per IB after each MC stage (arc, densify, smooth) plus a
    # summary of arc target vs chain target distances.  Used to diagnose the
    # "tangled ball" failure mode where cudaMMC produces clean loop rosettes
    # (per-IB anchors collapse tight, subanchors stretch to chain spring
    # targets in smooth MC).  Set to a directory path; empty string = off.
    debug_dump_stages: str = ""

    # Debug: stop after processing this many interaction blocks (across all
    # chromosomes).  0 = no limit.  Lets us inspect one IB end-to-end in
    # seconds instead of waiting for the full chromosome to finish.
    debug_max_ibs: int = 0

    # ── back-compat alias properties (legacy modules still reference these) ─
    @property
    def diagonal_size(self) -> int:
        # Per-heatmap fallback.  All NEW score paths take diagonal_size from
        # the heatmap object (cudaMMC Heatmap.cpp:58 getDiagonalSize); this
        # fallback exists only for tests/tools.
        return 3

    @property
    def k_chain(self) -> float:
        return self.spring_constant_stretch

    @property
    def angular_k(self) -> float:
        return self.spring_angular_constant

    @property
    def k_spring(self) -> float:
        return self.spring_constant_stretch_arcs

    @property
    def k_spring_repulsion(self) -> float:
        # cudaMMC LooperSolver.cpp:1933: `sc += 1.0f / v.length();` — no coefficient.
        return 1.0

    @property
    def k_orient(self) -> float:
        return self.motif_orientation_weight

    @property
    def k_heatmap(self) -> float:
        # cudaMMC LooperSolver.cpp:2222-2223: `err += cerr*cerr;` — no coefficient.
        return 1.0

    # Phase-2 noise re-init noise size (cudaMMC LooperSolver.cpp:2767  literal 0.05).
    @property
    def noise_size_small(self) -> float:
        return 0.05

    # ── Phase-4-pending shim aliases (mc.py / solver.py legacy callers) ─────
    # The current mc.py still uses delta-tracking and milestone-style step
    # accounting; cudaMMC does neither (full-recompute every iter, milestone
    # = tempJumpScale*size).  These aliases keep the legacy MC loops runnable
    # until Phase 4 lands.
    @property
    def step_size_heatmap(self) -> float:
        # cudaMMC step size = avg_dist (data-driven, cpp:307,312).  Use 1.0 as
        # a neutral fallback; the heatmap-MC caller computes the proper value
        # from `heatmap.getAvg() * noiseCoefficient*` in Phase 5.
        return 1.0
    @property
    def step_size_arcs(self) -> float:
        return 1.0
    @property
    def step_size_smooth(self) -> float:
        return 1.0
    @property
    def milestone_steps_heatmap(self) -> int:
        # cudaMMC Settings.cpp:222  MCstopConditionStepsHeatmap (= 10000).
        return self.mc_stop_steps_heatmap
    @property
    def milestone_steps_arcs(self) -> int:
        # cudaMMC Settings.cpp:238  MCstopConditionSteps (= 10000).
        return self.mc_stop_steps_arcs
    @property
    def milestone_steps_smooth(self) -> int:
        # cudaMMC Settings.cpp:248  MCstopConditionStepsSmooth (= 10000).
        return self.mc_stop_steps_smooth
    @property
    def milestone_improvement_ratio(self) -> float:
        # Legacy single-field alias (was used by mc.py for ALL phases at once).
        # The cudaMMC-faithful code paths read the per-phase field directly:
        # `mc_stop_improvement_{heatmap,arcs,smooth}` (Settings.cpp:220,236,246).
        return self.mc_stop_improvement_arcs
    @property
    def min_successes_heatmap(self) -> int:
        return self.mc_stop_min_successes_heatmap
    @property
    def min_successes_arcs(self) -> int:
        return self.mc_stop_min_successes_arcs
    @property
    def min_successes_smooth(self) -> int:
        return self.mc_stop_min_successes_smooth

    # ── INI loader (mirrors Settings::loadFromINI key-by-key) ──────────────
    @classmethod
    def from_ini(cls, path: str) -> "Settings":
        cfg = configparser.ConfigParser()
        cfg.read(path)
        # cudaMMC resolves data paths against ``data_dir`` (Settings.cpp:438).
        # The shipped GM12878 config has ``data_dir = /Projects/GM12878/``
        # which is bogus on most machines, so we also fall back to the INI's
        # own directory — that's where the bundled BED/BEDPE files live.
        import os as _os
        ini_dir = _os.path.dirname(_os.path.abspath(path))

        def _resolve(p: str, data_dir: str) -> str:
            """Resolve a path that may be bare, relative, or absolute."""
            if not p:
                return p
            if _os.path.isabs(p) and _os.path.exists(p):
                return p
            # Try as-given (relative to CWD), then INI dir, then data_dir.
            for cand in (p,
                         _os.path.join(ini_dir, p),
                         _os.path.join(data_dir, p) if data_dir else None,
                         _os.path.join(ini_dir, _os.path.basename(p))):
                if cand and _os.path.exists(cand):
                    return cand
            return p   # leave unchanged; downstream will warn/skip

        s = cls()

        def gf(sec: str, key: str, default: float) -> float:
            try:
                return cfg.getfloat(sec, key)
            except (configparser.NoSectionError, configparser.NoOptionError):
                return default

        def gi(sec: str, key: str, default: int) -> int:
            try:
                return cfg.getint(sec, key)
            except (configparser.NoSectionError, configparser.NoOptionError):
                return default

        def gb(sec: str, key: str, default: bool) -> bool:
            try:
                return cfg.getboolean(sec, key)
            except (configparser.NoSectionError, configparser.NoOptionError):
                return default

        def gs(sec: str, key: str, default: str) -> str:
            try:
                return cfg.get(sec, key)
            except (configparser.NoSectionError, configparser.NoOptionError):
                return default

        # cudaMMC Settings.cpp:379-384  [cuda]
        s.cuda_blocks_multiplier = gi("cuda", "blocks_multiplier", s.cuda_blocks_multiplier)
        s.cuda_threads_per_block = gi("cuda", "num_threads", s.cuda_threads_per_block)
        s.milestone_fails_threshold = gi("cuda", "milestone_fails", s.milestone_fails_threshold)

        # cudaMMC Settings.cpp:386-394  [main]
        s.output_level = gi("main", "output_level", s.output_level)
        s.random_walk = gb("main", "random_walk", s.random_walk)
        s.use_input_cache = gb("main", "cache_input", s.use_input_cache)
        s.use_2d = gb("main", "use_2D", s.use_2d)
        s.loop_density = gi("main", "loop_density", s.loop_density)

        # cudaMMC Settings.cpp:396-401  [motif_orientation]
        s.use_ctcf_motif_orientation = gb("motif_orientation", "use_motif_orientation", s.use_ctcf_motif_orientation)
        s.motifs_symmetric = gb("motif_orientation", "symmetric_motifs", s.motifs_symmetric)
        s.motif_orientation_weight = gf("motif_orientation", "weight", s.motif_orientation_weight)

        # cudaMMC Settings.cpp:403-413  [subanchor_heatmap]
        s.use_subanchor_heatmap = gb("subanchor_heatmap", "use_subanchor_heatmap", s.use_subanchor_heatmap)
        s.subanchor_estimate_distances_replicates = gi("subanchor_heatmap", "estimate_distances_replicates", s.subanchor_estimate_distances_replicates)
        s.subanchor_estimate_distances_steps = gi("subanchor_heatmap", "estimate_distances_steps", s.subanchor_estimate_distances_steps)
        s.subanchor_heatmap_influence = gf("subanchor_heatmap", "heatmap_influence", s.subanchor_heatmap_influence)
        s.subanchor_heatmap_dist_weight = gf("subanchor_heatmap", "heatmap_dist_weight", s.subanchor_heatmap_dist_weight)

        # cudaMMC Settings.cpp:415-420  [anchor_heatmap]
        s.use_anchor_heatmap = gb("anchor_heatmap", "use_anchor_heatmap", s.use_anchor_heatmap)
        s.anchor_heatmap_influence = gf("anchor_heatmap", "heatmap_influence", s.anchor_heatmap_influence)

        # cudaMMC Settings.cpp:424-429  [main] PET-cluster params
        s.max_pet_cluster_length = gi("main", "max_pet_length", s.max_pet_cluster_length)
        s.long_pet_clusters_effect_power = gf("main", "long_pet_power", s.long_pet_clusters_effect_power)
        s.long_pet_clusters_effect_scale = gf("main", "long_pet_scale", s.long_pet_clusters_effect_scale)

        # cudaMMC Settings.cpp:438-448  [data] paths
        s.data_directory = gs("data", "data_dir", s.data_directory)
        s.data_anchors = gs("data", "anchors", s.data_anchors)
        s.data_pet_clusters = gs("data", "clusters", s.data_pet_clusters)
        s.data_singletons = gs("data", "singletons", s.data_singletons)
        s.data_singletons_inter = gs("data", "singletons_inter", s.data_singletons_inter)
        s.data_factors = gs("data", "factors", s.data_factors)
        s.data_centromeres = gs("data", "centromeres", s.data_centromeres)
        s.data_segments_split = gs("data", "segment_split", s.data_segments_split)

        # Resolve all data paths against the INI's directory (see _resolve above).
        s.data_anchors = _resolve(s.data_anchors, s.data_directory)
        s.data_pet_clusters = _resolve(s.data_pet_clusters, s.data_directory)
        s.data_singletons = _resolve(s.data_singletons, s.data_directory)
        s.data_singletons_inter = _resolve(s.data_singletons_inter, s.data_directory)
        s.data_centromeres = _resolve(s.data_centromeres, s.data_directory)
        s.data_segments_split = _resolve(s.data_segments_split, s.data_directory)

        # cudaMMC Settings.cpp:450-458  [template]
        s.data_segment_heatmap = gs("template", "segment_heatmap", s.data_segment_heatmap)
        s.template_segment = gs("template", "template_segment", s.template_segment)
        s.template_scale = gf("template", "template_scale", s.template_scale)
        s.dist_heatmap = gs("template", "dist_heatmap", s.dist_heatmap)
        s.dist_heatmap_scale = gf("template", "dist_heatmap_scale", s.dist_heatmap_scale)

        # cudaMMC Settings.cpp:460-466  [density]
        s.use_density = gb("density", "use_density", s.use_density)
        s.density_scale = gf("density", "density_scale", s.density_scale)
        s.density_influence = gf("density", "density_influence", s.density_influence)
        s.density_weight = gf("density", "density_weight", s.density_weight)

        # cudaMMC Settings.cpp:481-493  [main] steps + noise
        s.simulation_steps_level_chr = gi("main", "steps_lvl1", s.simulation_steps_level_chr)
        s.simulation_steps_level_segment = gi("main", "steps_lvl2", s.simulation_steps_level_segment)
        s.simulation_steps_level_anchor = gi("main", "steps_arcs", s.simulation_steps_level_anchor)
        s.simulation_steps_level_subanchor = gi("main", "steps_smooth", s.simulation_steps_level_subanchor)
        s.noise_coefficient_level_chr = gf("main", "noise_lvl1", s.noise_coefficient_level_chr)
        s.noise_coefficient_level_segment = gf("main", "noise_lvl2", s.noise_coefficient_level_segment)
        s.noise_coefficient_level_anchor = gf("main", "noise_arcs", s.noise_coefficient_level_anchor)
        s.noise_coefficient_level_subanchor = gf("main", "noise_smooth", s.noise_coefficient_level_subanchor)

        # cudaMMC Settings.cpp:495-499  [heatmaps]
        s.heatmap_inter_scaling = gf("heatmaps", "inter_scaling", s.heatmap_inter_scaling)
        s.heatmap_distance_heatmap_stretching = gf("heatmaps", "distance_heatmap_stretching", s.heatmap_distance_heatmap_stretching)

        # cudaMMC Settings.cpp:501-525  [distance]
        s.ib_random_walk_jumps = gf("distance", "ib_random_walk_jumps", s.ib_random_walk_jumps)
        s.freq_dist_scale = gf("distance", "freq_dist_scale", s.freq_dist_scale)
        s.freq_dist_power = gf("distance", "freq_dist_power", s.freq_dist_power)
        s.freq_dist_scale_inter = gf("distance", "freq_dist_scale_inter", s.freq_dist_scale_inter)
        s.freq_dist_power_inter = gf("distance", "freq_dist_power_inter", s.freq_dist_power_inter)
        s.count_dist_a = gf("distance", "count_dist_a", s.count_dist_a)
        s.count_dist_scale = gf("distance", "count_dist_scale", s.count_dist_scale)
        s.count_dist_shift = gf("distance", "count_dist_shift", s.count_dist_shift)
        s.count_dist_base_level = gf("distance", "count_dist_base_level", s.count_dist_base_level)
        s.genomic_dist_power = gf("distance", "genomic_dist_power", s.genomic_dist_power)
        s.genomic_dist_scale = gf("distance", "genomic_dist_scale", s.genomic_dist_scale)
        s.genomic_dist_base = gf("distance", "genomic_dist_base", s.genomic_dist_base)

        # cudaMMC Settings.cpp:527-538  [springs]
        s.spring_constant_stretch = gf("springs", "stretch_constant", s.spring_constant_stretch)
        s.spring_constant_squeeze = gf("springs", "squeeze_constant", s.spring_constant_squeeze)
        s.spring_angular_constant = gf("springs", "angular_constant", s.spring_angular_constant)
        s.spring_constant_stretch_arcs = gf("springs", "stretch_constant_arcs", s.spring_constant_stretch_arcs)
        s.spring_constant_squeeze_arcs = gf("springs", "squeeze_constant_arcs", s.spring_constant_squeeze_arcs)

        # cudaMMC Settings.cpp:540-555  [simulation_heatmap]
        s.max_temp_heatmap = gf("simulation_heatmap", "max_temp_heatmap", s.max_temp_heatmap)
        s.dt_temp_heatmap = gf("simulation_heatmap", "delta_temp_heatmap", s.dt_temp_heatmap)
        s.temp_jump_coef_heatmap = gf("simulation_heatmap", "jump_temp_coef_heatmap", s.temp_jump_coef_heatmap)
        s.temp_jump_scale_heatmap = gf("simulation_heatmap", "jump_temp_scale_heatmap", s.temp_jump_scale_heatmap)
        s.mc_stop_steps_heatmap = gi("simulation_heatmap", "stop_condition_steps_heatmap", s.mc_stop_steps_heatmap)
        s.mc_stop_improvement_heatmap = gf("simulation_heatmap", "stop_condition_improvement_threshold_heatmap", s.mc_stop_improvement_heatmap)
        s.mc_stop_min_successes_heatmap = gi("simulation_heatmap", "stop_condition_successes_threshold_heatmap", s.mc_stop_min_successes_heatmap)

        # cudaMMC Settings.cpp:577-589  [simulation_arcs]
        s.max_temp_arcs = gf("simulation_arcs", "max_temp", s.max_temp_arcs)
        s.dt_temp_arcs = gf("simulation_arcs", "delta_temp", s.dt_temp_arcs)
        s.temp_jump_coef_arcs = gf("simulation_arcs", "jump_temp_coef", s.temp_jump_coef_arcs)
        s.temp_jump_scale_arcs = gf("simulation_arcs", "jump_temp_scale", s.temp_jump_scale_arcs)
        s.mc_stop_steps_arcs = gi("simulation_arcs", "stop_condition_steps", s.mc_stop_steps_arcs)
        s.mc_stop_improvement_arcs = gf("simulation_arcs", "stop_condition_improvement_threshold", s.mc_stop_improvement_arcs)
        s.mc_stop_min_successes_arcs = gi("simulation_arcs", "stop_condition_successes_threshold", s.mc_stop_min_successes_arcs)

        # cudaMMC Settings.cpp:591-611  [simulation_arcs_smooth]
        # IMPORTANT: dist_weight → weight_angle_smooth and angle_weight →
        # weight_dist_smooth is the upstream swap bug (Settings.cpp:594-597).
        # We mirror it byte-for-byte per AGENTS.md prime directive.
        s.weight_angle_smooth = gf("simulation_arcs_smooth", "dist_weight", s.weight_angle_smooth)
        s.weight_dist_smooth = gf("simulation_arcs_smooth", "angle_weight", s.weight_dist_smooth)
        s.max_temp_smooth = gf("simulation_arcs_smooth", "max_temp", s.max_temp_smooth)
        s.dt_temp_smooth = gf("simulation_arcs_smooth", "delta_temp", s.dt_temp_smooth)
        s.temp_jump_coef_smooth = gf("simulation_arcs_smooth", "jump_temp_coef", s.temp_jump_coef_smooth)
        s.temp_jump_scale_smooth = gf("simulation_arcs_smooth", "jump_temp_scale", s.temp_jump_scale_smooth)
        s.mc_stop_steps_smooth = gi("simulation_arcs_smooth", "stop_condition_steps", s.mc_stop_steps_smooth)
        s.mc_stop_improvement_smooth = gf("simulation_arcs_smooth", "stop_condition_improvement_threshold", s.mc_stop_improvement_smooth)
        s.mc_stop_min_successes_smooth = gi("simulation_arcs_smooth", "stop_condition_successes_threshold", s.mc_stop_min_successes_smooth)

        # Python-only safety caps (off by default).  Either [main] or the
        # phase-specific section is honoured for convenience.
        s.mc_max_iters_arcs = gi("simulation_arcs", "max_iters", s.mc_max_iters_arcs)
        s.mc_max_seconds_arcs = gf("simulation_arcs", "max_seconds", s.mc_max_seconds_arcs)
        s.mc_max_iters_smooth = gi("simulation_arcs_smooth", "max_iters", s.mc_max_iters_smooth)
        s.mc_max_seconds_smooth = gf("simulation_arcs_smooth", "max_seconds", s.mc_max_seconds_smooth)

        # Python-only — accept either [misc] or [main]
        s.device = gs("misc", "device", s.device)

        # ── fail loudly on currently-unsupported features ────────────────
        # See AGENTS.md "Inter-chromosomal arcs and density features": raise
        # NotImplementedError rather than diverge silently.
        for flag_name, val in (
            ("use_density", s.use_density),
            ("use_telomere_positions", s.use_telomere_positions),
            # AUDIT §G11 (random_walk fast path) / §G12 (template_segment).
            # Both are Phase-5 stubs — flag them so silent divergence is impossible.
            ("random_walk", s.random_walk),
        ):
            if val:
                raise NotImplementedError(
                    f"Settings.{flag_name}=True is not implemented in 3dgnome-torch. "
                    f"Disable in INI or use the cudaMMC reference."
                )
        if s.template_segment:
            # cudaMMC LooperSolver.cpp:193-211 reads a template structure file
            # and seeds segment positions from it.  Not ported (AUDIT §G12).
            raise NotImplementedError(
                "Settings.template_segment is not implemented "
                "(cudaMMC LooperSolver.cpp:193-211)."
            )
        if s.use_anchor_heatmap:
            # calcAnchorExpectedDistancesHeatmap anchor-heatmap modulation
            # (cudaMMC LooperSolver.cpp:3886-3914) is not yet ported.
            raise NotImplementedError(
                "Settings.use_anchor_heatmap=True is not implemented "
                "(cudaMMC LooperSolver.cpp:3886-3914 modulation missing)."
            )

        return s
