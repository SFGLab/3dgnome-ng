"""Distance conversion formulas (vectorised PyTorch versions and scalar Python versions)."""

import math
import torch


# ── Scalar helpers (used during tree building / CPU phases) ─────────────────

def genomic_length_to_distance(length: int, scale: float = 0.5,
                                power: float = 0.75, base: float = 1.0) -> float:
    """Map genomic span (bp) to a spatial distance estimate.

    Original cudaMMC uses length in kilobases: dist = base + scale * (length_kb)^power.
    Using raw bp would give values ~1000× too large.
    """
    if length <= 0:
        return base
    length_kb = length / 1000.0
    return base + scale * (length_kb ** power)


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


def count_to_distance(count: int, a: float = 0.2, scale: float = 1.8,
                      shift: float = 8.0, base_level: float = 0.2) -> float:
    """PET count → spatial distance for arc spring targets.

    d = base_level + scale / exp(a * (count + shift))
    Matches the original cudaMMC exponential formula.
    """
    return base_level + scale / math.exp(a * (count + shift))


# ── Tensor versions (used in score functions) ────────────────────────────────

def genomic_length_to_distance_t(lengths: torch.Tensor, scale: float = 0.5,
                                  power: float = 0.75, base: float = 1.0
                                  ) -> torch.Tensor:
    lengths_kb = lengths.float().clamp(min=0) / 1000.0
    return base + scale * (lengths_kb.clamp(min=1e-3) ** power)


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


def count_to_distance_t(counts: torch.Tensor, a: float = 0.2,
                         scale: float = 1.8, shift: float = 8.0,
                         base_level: float = 0.2) -> torch.Tensor:
    """PET count → spatial distance (tensor version).

    d = base_level + scale / exp(a * (count + shift))
    """
    return base_level + scale / torch.exp(a * (counts.float() + shift))
