"""The polymer law and the fit of its exponent on the run's contacts.

Mean distance grows with genomic separation as `d ~ s ^ nu` and contact probability falls as
`s ^ (-3 nu)`. The slope of log contact count against log separation is `-3 nu`.
`FALLBACK_NU` is used only when the input cannot supply an exponent and the run logs that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gnome3d.types import InteractionArc, SingletonContact

FALLBACK_NU = 0.285
"""The fractal globule value, and what three 4DN Hi-C cell lines measure at 20 kb to 1 Mb. Used
only when a run's own contacts cannot supply an exponent."""

_BAND_LO = 20_000
_BAND_HI = 1_000_000
_MIN_PAIRS = 5_000
_MIN_BINS = 8
_MIN_COUNT = 20
_SLOPE_RANGE = (-1.6, -0.5)


@dataclass(frozen=True)
class ContactFit:
    """The result of the exponent fit.

    `nu` is the exponent. When `ok` is false it is `FALLBACK_NU` and `reason` states the cause.
    `lo` and `hi` are the band in bp, `n_pairs` the intra chromosomal pairs inside it and `n_bins`
    the populated bins of the fit.
    """

    nu: float
    slope: float
    lo: int
    hi: int
    n_pairs: int
    n_bins: int
    ok: bool
    reason: str


def _refused(reason: str, lo: int = _BAND_LO, hi: int = _BAND_HI, n: int = 0) -> ContactFit:
    return ContactFit(FALLBACK_NU, float("nan"), lo, hi, n, 0, False, reason)


def fit_contact_exponent(contacts: list[SingletonContact]) -> ContactFit:
    """Fits the contact decay exponent on a run's singletons.

    Uses intra chromosomal pairs on the grid of the file. The contact probability at a grid
    separation is the counts there over the bin pairs the chromosomes offer at that separation.
    The band runs from 20 kb or twice the grid step, the larger, to 1 Mb. The fit is refused, with
    a reason, when the band holds too few pairs or separations or the slope is not a polymer decay.

    Parameters
    ----------
    contacts
        `(chr_a, pos_a, chr_b, pos_b, score)` records.
    """
    if not contacts:
        return _refused("no contacts")
    intra = [c for c in contacts if c[0] == c[2]]
    if not intra:
        return _refused("no intra chromosomal pairs")
    chrom = np.array([c[0] for c in intra])
    a = np.array([c[1] for c in intra], dtype=np.float64)
    b = np.array([c[3] for c in intra], dtype=np.float64)
    w = np.array([max(float(c[4]), 0.0) for c in intra], dtype=np.float64)
    sep = np.abs(b - a)
    pos = sep > 0
    if not pos.any():
        return _refused("no intra chromosomal pairs")
    # The grid step is the smallest separation that many pairs share. The smallest in the file can
    # be the end bin of a chromosome, which is cut short.
    vals, freq = np.unique(sep[pos], return_counts=True)
    step = float(vals[freq >= 0.1 * freq.max()][0])
    lo = int(max(_BAND_LO, 2 * step))
    hi = _BAND_HI
    if lo >= hi:
        return _refused(f"resolution too coarse for a {hi // 1000} kb band", lo, hi)
    band = (sep >= lo) & (sep <= hi)
    n_pairs = int(band.sum())
    if n_pairs < _MIN_PAIRS:
        return _refused(f"only {n_pairs:,} pairs in the band", lo, hi, n_pairs)
    k = np.rint(sep[band] / step).astype(np.int64)
    k_max = int(k.max())
    counts = np.bincount(k, weights=w[band], minlength=k_max + 1)
    offered = np.zeros(k_max + 1, dtype=np.float64)
    for name in np.unique(chrom):
        m = chrom == name
        n_bins = (
            int(np.rint((max(a[m].max(), b[m].max()) - min(a[m].min(), b[m].min())) / step)) + 1
        )
        kk = np.arange(k_max + 1)
        offered += np.maximum(n_bins - kk, 0)
    ks = np.arange(k_max + 1)
    keep = (counts > _MIN_COUNT) & (offered > 0) & (ks * step >= lo) & (ks * step <= hi)
    if int(keep.sum()) < _MIN_BINS:
        return _refused(f"only {int(keep.sum())} populated separations", lo, hi, n_pairs)
    prob = counts[keep] / offered[keep]
    slope = float(np.polyfit(np.log(ks[keep] * step), np.log(prob), 1)[0])
    if not (_SLOPE_RANGE[0] <= slope <= _SLOPE_RANGE[1]):
        return ContactFit(
            FALLBACK_NU,
            slope,
            lo,
            hi,
            n_pairs,
            int(keep.sum()),
            False,
            f"slope {slope:+.3f} is not a polymer decay, expected {_SLOPE_RANGE[0]} to "
            f"{_SLOPE_RANGE[1]}",
        )
    return ContactFit(-slope / 3.0, slope, lo, hi, n_pairs, int(keep.sum()), True, "")


_ARC_MIN = 200
_ARC_BINS_PER_DECADE = 4
_ARC_MIN_PER_BIN = 10


@dataclass(frozen=True)
class ArcStrengthFit:
    """The typical PET count of a loop against its span, fitted on the run's arcs.

    `strength(score, span)` is the PET count over the typical count at that span. When `ok` is
    false the typical count is one constant, the median.
    """

    log_a: float
    slope: float
    median: float
    ok: bool
    reason: str

    def typical(self, span_bp: int) -> float:
        if not self.ok:
            return self.median
        return float(np.exp(self.log_a + self.slope * np.log(max(abs(span_bp), 1))))

    def strength(self, score: int, span_bp: int) -> float:
        return float(score) / max(self.typical(span_bp), 1e-9)


def fit_arc_strength(arcs: list[InteractionArc]) -> ArcStrengthFit:
    """Fits the typical PET count against span on a run's arcs.

    A power law is fitted through the median count of each log span bin. The fit is refused, with
    the constant fallback, for too few arcs or too few populated bins.

    Parameters
    ----------
    arcs
        Arcs with their genomic span set.
    """
    span = np.array([abs(a.genomic_end - a.genomic_start) for a in arcs], dtype=np.float64)
    score = np.array([a.score for a in arcs], dtype=np.float64)
    keep = (span > 0) & (score > 0)
    span, score = span[keep], score[keep]
    med = float(np.median(score)) if score.size else 1.0
    if score.size < _ARC_MIN:
        return ArcStrengthFit(0.0, 0.0, med, False, f"only {score.size} arcs")
    ls = np.log10(span)
    edges = np.arange(
        np.floor(ls.min() * _ARC_BINS_PER_DECADE), np.ceil(ls.max() * _ARC_BINS_PER_DECADE) + 1
    )
    edges = edges / _ARC_BINS_PER_DECADE
    which = np.clip(np.searchsorted(edges, ls, side="right") - 1, 0, edges.size - 2)
    xs: list[float] = []
    ys: list[float] = []
    for b in range(edges.size - 1):
        m = which == b
        if int(m.sum()) < _ARC_MIN_PER_BIN:
            continue
        xs.append(float(np.median(np.log(span[m]))))
        ys.append(float(np.log(np.median(score[m]))))
    if len(xs) < 3:
        return ArcStrengthFit(0.0, 0.0, med, False, f"only {len(xs)} populated span bins")
    slope, log_a = np.polyfit(xs, ys, 1)
    return ArcStrengthFit(float(log_a), float(slope), med, True, "")


@dataclass(frozen=True)
class PolymerLaw:
    """One law for every distance, in bead units. One bead is the distance of two beads `s0_bp` apart.

    Parameters
    ----------
    nu
        The exponent, measured from the run's contacts or pinned.
    s0_bp
        The separation of one bead, the run's subanchor spacing.
    q_half
        The loop strength, in multiples of a typical loop at that span, that pulls a pair halfway
        from the background to one bead.
    arcs
        The fit of typical loop strength against span. None makes strength the PET count over its
        median.
    """

    nu: float
    s0_bp: int
    q_half: float = 1.0
    arcs: ArcStrengthFit | None = None
    arcs_by_factor: dict[int, ArcStrengthFit] = field(default_factory=dict)
    # A multiplier on the loop strength of a factor. 0 holds its loops at the background.
    strength_by_factor: dict[int, float] = field(default_factory=dict)

    def background(self, sep_bp: int) -> float:
        """Returns the distance of two beads `sep_bp` apart with no contact, at least one bead."""
        sep = max(abs(int(sep_bp)), 1)
        return max(1.0, (sep / max(int(self.s0_bp), 1)) ** self.nu)

    def radius_of_gyration(self, span_bp: int) -> float:
        """Returns the radius of gyration of a chain of `span_bp` whose pair distances follow the law.

        The mean square pair distance of a unit chain is 1 / ((2 nu + 1)(nu + 1)) and the squared
        radius of gyration is half of it.
        """
        nu = self.nu
        return self.background(span_bp) / (2.0 * (2.0 * nu + 1.0) * (nu + 1.0)) ** 0.5

    def confinement_radius(self, span_bp: int) -> float:
        """Returns the radius of the uniform sphere with the chain's radius of gyration, root five
        thirds of it.
        """
        return (5.0 / 3.0) ** 0.5 * self.radius_of_gyration(span_bp)

    def contact_distance(self, sep_bp: int, q: float) -> float:
        """Returns the distance of a pair with loop strength `q`, the background at 0, one bead at
        saturation and halfway at `q_half`.
        """
        bg = self.background(sep_bp)
        h = 1.0 / (1.0 + max(q, 0.0) / max(self.q_half, 1e-9))
        return 1.0 + (bg - 1.0) * h

    def arc_strength(self, score: int, sep_bp: int, factor: int = 0) -> float:
        """Returns the strength of an arc, its PET count over the typical count at its span, times
        the factor's multiplier. A factor without a fit uses the fit of factor 0.
        """
        span = abs(int(sep_bp))
        fit = self.arcs_by_factor.get(factor, self.arcs)
        q = fit.strength(score, span) if fit is not None else float(score)
        return q * self.strength_by_factor.get(factor, 1.0)

    def arc_distance(self, score: int, sep_bp: int, factor: int = 0) -> float:
        """Returns the target distance of an arc."""
        return self.contact_distance(abs(int(sep_bp)), self.arc_strength(score, sep_bp, factor))

    def heatmap_distance(self, freq: float, expected: float, sep_bp: int) -> float:
        """Returns the target of a heatmap cell, the background times observed over expected to the
        minus third. No contact is no target.
        """
        if freq <= 1e-12 or expected <= 1e-12:
            return 0.0
        return self.background(sep_bp) * (freq / expected) ** (-1.0 / 3.0)
