"""Distance conversion formulas (vectorised PyTorch versions and scalar Python versions)."""

import math
import torch


# ── Scalar helpers (used during tree building / CPU phases) ─────────────────

def genomic_length_to_distance(length: int, scale: float = 0.5,
                                power: float = 0.75, base: float = 1.0) -> float:
    """Map genomic span (bp) to an initial spatial distance estimate."""
    if length <= 0:
        return base
    return base + scale * (length ** power)


def freq_to_distance_heatmap(freq: float, scale: float = 100.0,
                              power: float = -0.333) -> float:
    """Hi-C normalised frequency → spatial distance (heatmap phase)."""
    if freq <= 0.0:
        return scale  # large distance when no contact
    return scale * (freq ** power)


def freq_to_distance_intra(freq: float, scale: float = 25.0,
                            power: float = -0.6) -> float:
    """Per-contact intra-chromosomal freq → distance (arcs phase)."""
    if freq <= 0.0:
        return scale
    return scale * (freq ** power)


def freq_to_distance_inter(freq: float, scale: float = 120.0,
                            power: float = -1.0) -> float:
    """Per-contact inter-chromosomal freq → distance."""
    if freq <= 0.0:
        return scale
    return scale * (freq ** power)


def count_to_distance(count: int, a: float = 0.5, scale: float = 20.0,
                      shift: float = 1.0, base_level: float = 0.01) -> float:
    """PET count → spatial distance for arc spring targets.

    d = base_level + a * scale / (count + shift)
    """
    return base_level + a * scale / (count + shift)


# ── Tensor versions (used in score functions) ────────────────────────────────

def genomic_length_to_distance_t(lengths: torch.Tensor, scale: float = 0.5,
                                  power: float = 0.75, base: float = 1.0
                                  ) -> torch.Tensor:
    lengths = lengths.float().clamp(min=0)
    return base + scale * (lengths.clamp(min=1) ** power)


def freq_to_distance_heatmap_t(freqs: torch.Tensor, scale: float = 100.0,
                                power: float = -0.333) -> torch.Tensor:
    safe = freqs.float().clamp(min=1e-9)
    return scale * (safe ** power)


def freq_to_distance_intra_t(freqs: torch.Tensor, scale: float = 25.0,
                              power: float = -0.6) -> torch.Tensor:
    safe = freqs.float().clamp(min=1e-9)
    return scale * (safe ** power)


def freq_to_distance_inter_t(freqs: torch.Tensor, scale: float = 120.0,
                              power: float = -1.0) -> torch.Tensor:
    safe = freqs.float().clamp(min=1e-9)
    return scale * (safe ** power)


def count_to_distance_t(counts: torch.Tensor, a: float = 0.5,
                         scale: float = 20.0, shift: float = 1.0,
                         base_level: float = 0.01) -> torch.Tensor:
    return base_level + a * scale / (counts.float() + shift)
