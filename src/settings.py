"""All algorithm parameters with defaults matching cudaMMC config.ini / Settings.cpp."""

import configparser
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Settings:
    # ── CUDA / MC runtime ────────────────────────────────────────────────────
    cuda_threads_per_block: int = 256
    cuda_blocks_multiplier: int = 4
    milestone_fails_threshold: int = 3

    # ── Distance conversions ─────────────────────────────────────────────────
    # Genomic length → spatial distance  d = base + scale * len^power
    genomic_dist_power: float = 0.75
    genomic_dist_scale: float = 0.5
    genomic_dist_base: float = 1.0

    # Hi-C freq → distance (heatmap phase)  d = scale * freq^power
    freq_dist_scale: float = 100.0
    freq_dist_power: float = -0.333

    # Hi-C freq → distance (per-contact intra)
    freq_dist_scale_per_contact: float = 25.0
    freq_dist_power_per_contact: float = -0.6

    # Hi-C freq → distance (per-contact inter-chr)
    freq_dist_scale_inter: float = 120.0
    freq_dist_power_inter: float = -1.0

    # PET count → distance  d = base_level + a * scale / (count + shift)
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
    k_chain: float = 1.0              # linker-length harmonic spring
    angular_k: float = 0.1           # cubic angular penalty coefficient
    k_orient: float = 1.0            # CTCF orientation penalty weight
    noise_size_small: float = 0.05    # initial displacement noise for anchors

    # ── Monte Carlo – Heatmap phase ──────────────────────────────────────────
    max_temp_heatmap: float = 20.0
    dt_temp_heatmap: float = 0.99995  # per outer-step multiplicative cooling
    temp_jump_scale_heatmap: float = 50.0
    temp_jump_coef_heatmap: float = 20.0
    step_size_heatmap: float = 1.5
    step_size_decay_heatmap: float = 0.95  # per outer-step
    mc_inner_steps: int = 512         # N inner steps per warp/outer-step

    # ── Monte Carlo – Arcs phase ─────────────────────────────────────────────
    max_temp_arcs: float = 10.0
    dt_temp_arcs: float = 0.9999
    temp_jump_scale_arcs: float = 20.0
    temp_jump_coef_arcs: float = 10.0
    step_size_arcs: float = 0.5
    step_size_decay_arcs: float = 0.99
    min_successes_arcs: int = 10
    improvement_threshold_arcs: float = 1e-4

    # ── Monte Carlo – Smooth phase ────────────────────────────────────────────
    max_temp_smooth: float = 5.0
    dt_temp_smooth: float = 0.9999
    temp_jump_scale_smooth: float = 10.0
    temp_jump_coef_smooth: float = 5.0
    step_size_smooth: float = 0.3
    step_size_decay_smooth: float = 0.99
    min_successes_smooth: int = 10
    improvement_threshold_smooth: float = 1e-4

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
            cuda_threads_per_block=g_int("cuda", "num_threads", 256),
            cuda_blocks_multiplier=g_int("cuda", "blocks_multiplier", 4),
            milestone_fails_threshold=g_int("cuda", "milestone_fails", 3),

            genomic_dist_power=g_float("distance", "genomic_dist_power", 0.75),
            genomic_dist_scale=g_float("distance", "genomic_dist_scale", 0.5),
            genomic_dist_base=g_float("distance", "genomic_dist_base", 1.0),

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

            use_2d=g_bool("misc", "use_2d", False),
            device=cfg.get("misc", "device", fallback="cuda"),
        )
