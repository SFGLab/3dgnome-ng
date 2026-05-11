"""All algorithm parameters with defaults matching cudaMMC config.ini / Settings.cpp."""

import configparser
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Settings:
    # ── CUDA / MC runtime ────────────────────────────────────────────────────
    milestone_fails_threshold: int = 3

    # ── Distance conversions ─────────────────────────────────────────────────
    # Genomic length → spatial distance  d = base + scale * (len_kb)^power
    # cudaMMC Settings.cpp: power=0.5, scale=1.0, base=0.0
    genomic_dist_power: float = 0.5
    genomic_dist_scale: float = 1.0
    genomic_dist_base: float = 0.0

    # Hi-C freq → distance (heatmap phase)  d = scale * freq^power
    # cudaMMC Settings.cpp: freqToDistHeatmapScale=100.0, power=-0.333
    freq_dist_scale: float = 100.0
    freq_dist_power: float = -0.333

    # Hi-C freq → distance (per-contact intra)
    # cudaMMC Settings.cpp: freqToDistScale=25.0, power=-0.6
    freq_dist_scale_per_contact: float = 25.0
    freq_dist_power_per_contact: float = -0.6

    # Hi-C freq → distance (per-contact inter-chr)
    freq_dist_scale_inter: float = 120.0
    freq_dist_power_inter: float = -1.0

    # PET count → distance  d = base_level + scale / exp(a * (count + shift))
    # cudaMMC Settings.cpp: a=0.5, scale=20.0, shift=1.0, base=0.01
    count_dist_a: float = 0.5
    count_dist_scale: float = 20.0
    count_dist_shift: float = 1.0
    count_dist_base_level: float = 0.01

    # ── Heatmap spring constants ─────────────────────────────────────────────
    k_heatmap: float = 1.0
    diagonal_size: int = 3            # exclude |i-j| < diagonal_size pairs

    # ── Arc spring constants ─────────────────────────────────────────────────
    k_spring: float = 1.0
    k_spring_repulsion: float = 1.0   # repulsion coefficient for unknown pairs

    # ── Structural / smooth spring constants ─────────────────────────────────
    # cudaMMC Settings.cpp: springConstantSqueeze=springConstantStretch=0.1
    k_chain: float = 0.1              # linker-length harmonic spring (squeeze=stretch)
    angular_k: float = 0.1           # cubic angular penalty coefficient
    k_orient: float = 1.0            # CTCF orientation penalty weight
    noise_size_small: float = 0.05    # initial displacement noise for anchors

    # ── Monte Carlo - Heatmap phase ──────────────────────────────────────────
    max_temp_heatmap: float = 20.0
    dt_temp_heatmap: float = 0.999   # per outer-step; reaches T=2 in ~2300 steps
    temp_jump_scale_heatmap: float = 50.0
    temp_jump_coef_heatmap: float = 20.0
    step_size_heatmap: float = 1.5
    step_size_decay_heatmap: float = 0.999   # matches dt_temp_heatmap
    mc_inner_steps: int = 512         # N inner steps per warp/outer-step

    # ── Monte Carlo - Arcs phase ─────────────────────────────────────────────
    # cudaMMC Settings.cpp: maxTemp=20, dtTemp=0.99995, jumpCoef=20, jumpScale=50
    max_temp_arcs: float = 20.0
    dt_temp_arcs: float = 0.99995     # applied PER BEAD MOVE (not per outer step)
    temp_jump_scale_arcs: float = 50.0
    temp_jump_coef_arcs: float = 20.0
    step_size_arcs: float = 0.5
    step_size_decay_arcs: float = 0.9999
    min_successes_arcs: int = 5       # cudaMMC MCstopConditionMinSuccesses=5
    milestone_steps_arcs: int = 10000 # cudaMMC MCstopConditionSteps=10000
    milestone_improvement_ratio: float = 0.995  # cudaMMC MCstopConditionImprovement

    # ── Monte Carlo - Smooth phase ────────────────────────────────────────────
    # cudaMMC Settings.cpp: maxTempSmooth=20, dtTempSmooth=0.99995, coef=20, scale=50
    max_temp_smooth: float = 20.0
    dt_temp_smooth: float = 0.99995   # applied PER BEAD MOVE (not per outer step)
    temp_jump_scale_smooth: float = 50.0
    temp_jump_coef_smooth: float = 20.0
    step_size_smooth: float = 0.3
    step_size_decay_smooth: float = 0.9999
    min_successes_smooth: int = 5     # cudaMMC MCstopConditionMinSuccessesSmooth=5
    milestone_steps_smooth: int = 10000  # cudaMMC MCstopConditionStepsSmooth=10000

    # ── Misc ──────────────────────────────────────────────────────────────────
    use_2d: bool = False              # confine moves to XY plane (debug)
    device: str = "cuda"             # "cuda" or "cpu"

    # ── Hierarchical structure sizes ──────────────────────────────────────────
    # expected genomic span for each level (used in initial placement)
    segment_size: int = 2_000_000    # ~2 Mb per segment
    ib_size: int = 200_000           # ~200 kb per interaction block

    @classmethod
    def from_ini(cls, path: str) -> "Settings":
        cfg = configparser.ConfigParser()
        cfg.read(path)

        def g_float(section: str, key: str, default: float) -> float:
            try:
                return cfg.getfloat(section, key)
            except (configparser.NoSectionError, configparser.NoOptionError):
                return default

        def g_int(section: str, key: str, default: int) -> int:
            try:
                return cfg.getint(section, key)
            except (configparser.NoSectionError, configparser.NoOptionError):
                return default

        def g_bool(section: str, key: str, default: bool) -> bool:
            try:
                return cfg.getboolean(section, key)
            except (configparser.NoSectionError, configparser.NoOptionError):
                return default

        return cls(
            milestone_fails_threshold=g_int("cuda", "milestone_fails", 3),
            mc_inner_steps=g_int("cuda", "mc_inner_steps", 512),

            genomic_dist_power=g_float("distance", "genomic_dist_power", 0.5),
            genomic_dist_scale=g_float("distance", "genomic_dist_scale", 1.0),
            genomic_dist_base=g_float("distance", "genomic_dist_base", 0.0),

            freq_dist_scale=g_float("distance", "freq_dist_heatmap_scale", 100.0),
            freq_dist_power=g_float("distance", "freq_dist_heatmap_power", -0.333),

            freq_dist_scale_per_contact=g_float("distance", "freq_dist_scale", 25.0),
            freq_dist_power_per_contact=g_float("distance", "freq_dist_power", -0.6),

            freq_dist_scale_inter=g_float("distance", "freq_dist_scale_inter", 120.0),
            freq_dist_power_inter=g_float("distance", "freq_dist_power_inter", -1.0),

            count_dist_a=g_float("distance", "count_dist_a", 0.5),
            count_dist_scale=g_float("distance", "count_dist_scale", 20.0),
            count_dist_shift=g_float("distance", "count_dist_shift", 1.0),
            count_dist_base_level=g_float("distance", "count_dist_base_level", 0.01),

            min_successes_arcs=g_int("simulation_arcs", "min_successes", 5),
            milestone_steps_arcs=g_int("simulation_arcs", "milestone_steps", 10000),
            milestone_improvement_ratio=g_float("simulation_arcs", "improvement_ratio", 0.995),
            min_successes_smooth=g_int("simulation_arcs_smooth", "min_successes", 5),
            milestone_steps_smooth=g_int("simulation_arcs_smooth", "milestone_steps", 10000),

            use_2d=g_bool("misc", "use_2d", False),
            device=cfg.get("misc", "device", fallback="cuda"),
        )
