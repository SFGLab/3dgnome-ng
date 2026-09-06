"""The polymer law, and the contact decay fit that supplies its exponent.

Chromatin in a nucleus is crumpled. Mean spatial distance grows with genomic separation as a
power law, `d ~ s ^ nu`, and two loci meet with probability near `s ^ (-3 nu)`, since the
capture volume is fixed and the coil volume grows as the cube of its radius. So the exponent
distance grows with can be read off the run's own contact data: the slope of log contact count
against log separation is `-3 nu`. Nothing here is a constant taken from another dataset. The
one number that is, `FALLBACK_NU`, is used only when the input cannot supply its own, and the
run says so.

See [[project_unified_arc_target]] for why the arcs stage needed this, and
[[project_polymer_law]] for the design.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnome3d.types import InteractionArc, SingletonContact

FALLBACK_NU = 0.285
"""The fractal globule value, and what three 4DN Hi-C cell lines measure at 20 kb to 1 Mb. Used
only when a run's own contacts cannot supply an exponent."""

_BAND_LO = 20_000
_BAND_HI = 1_000_000
_N_BINS = 12
_MIN_PAIRS = 5_000
_MIN_BINS = 8
_MIN_COUNT = 20
_SLOPE_RANGE = (-1.6, -0.5)


@dataclass(frozen=True)
class ContactFit:
    """What the fit found, and whether it can be used.

    `nu` is the exponent distance grows with. When `ok` is false it is `FALLBACK_NU` and
    `reason` says why the data could not supply one. `lo` and `hi` are the band in bp, `n_pairs`
    the intra chromosomal pairs inside it and `n_bins` the populated bins the slope was fitted
    on.
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
    """Fit the contact decay exponent on a run's singletons.

    Intra chromosomal pairs only. The band starts at 20 kb or at twice the smallest separation
    the data resolves, whichever is larger, so a file binned at 25 kb is fitted from 50 kb, and
    runs to 1 Mb. Counts are binned in log separation and divided by bin width, so the bin
    layout does not shape the slope. The fit is refused, with the reason recorded, when the
    band holds too few pairs or bins, or when the slope is not a decay a polymer can produce.

    Parameters
    ----------
    contacts
        `(chr_a, pos_a, chr_b, pos_b, score)` records, as the singletons loader returns them.
    """
    if not contacts:
        return _refused("no contacts")
    a = np.array([c[1] for c in contacts if c[0] == c[2]], dtype=np.float64)
    b = np.array([c[3] for c in contacts if c[0] == c[2]], dtype=np.float64)
    sep = np.abs(b - a)
    sep = sep[sep > 0]
    if sep.size == 0:
        return _refused("no intra chromosomal pairs")
    lo = int(max(_BAND_LO, 2 * sep.min()))
    hi = _BAND_HI
    if lo >= hi:
        return _refused(f"resolution too coarse for a {hi // 1000} kb band", lo, hi)
    inside = sep[(sep >= lo) & (sep <= hi)]
    if inside.size < _MIN_PAIRS:
        return _refused(f"only {inside.size:,} pairs in the band", lo, hi, int(inside.size))
    edges = np.logspace(np.log10(lo), np.log10(hi), _N_BINS + 1)
    counts, _ = np.histogram(inside, edges)
    centres = np.sqrt(edges[:-1] * edges[1:])
    density = counts / np.diff(edges)
    keep = counts > _MIN_COUNT
    if int(keep.sum()) < _MIN_BINS:
        return _refused(f"only {int(keep.sum())} populated bins", lo, hi, int(inside.size))
    slope = float(np.polyfit(np.log(centres[keep]), np.log(density[keep]), 1)[0])
    if not (_SLOPE_RANGE[0] <= slope <= _SLOPE_RANGE[1]):
        return ContactFit(
            FALLBACK_NU,
            slope,
            lo,
            hi,
            int(inside.size),
            int(keep.sum()),
            False,
            f"slope {slope:+.3f} is not a polymer decay, expected {_SLOPE_RANGE[0]} to "
            f"{_SLOPE_RANGE[1]}",
        )
    return ContactFit(-slope / 3.0, slope, lo, hi, int(inside.size), int(keep.sum()), True, "")


_ARC_MIN = 200
_ARC_BINS_PER_DECADE = 4
_ARC_MIN_PER_BIN = 10


@dataclass(frozen=True)
class ArcStrengthFit:
    """How strong a loop of a given span typically is in this run's own arcs.

    `strength(score, span)` is the arc's PET count over the typical count at its span, which is
    observed over expected read off the arcs themselves rather than a curve from another
    dataset. When `ok` is false the typical count is a constant and strength follows the PET
    count alone, scaled by the median.
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
    """Fit the typical PET count against span on a run's arcs.

    Arcs are binned in log span, the median count per populated bin is taken, and a power law
    is fitted through the bin medians. Refused, with the constant fallback, when there are too
    few arcs or too few populated bins to fit a line through.

    Parameters
    ----------
    arcs
        Arcs with their genomic span set, as the loader returns them.
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
    """One law for every distance, in bead units.

    The bead is the distance two beads hold at the resolution the run declared, `s0_bp`, so the
    background there is one by definition and nothing is in a unit that means nothing.

    Parameters
    ----------
    nu
        The exponent distance grows with, measured from the run's contacts or pinned.
    s0_bp
        The separation that is one bead, the run's subanchor spacing.
    q_half
        The loop strength, in multiples of a typical loop at that span, at which a contact
        pulls its pair halfway from the background to touching.
    arcs
        The run's own typical loop strength against span. None makes strength the PET count
        over its median.
    """

    nu: float
    s0_bp: int
    q_half: float = 1.0
    arcs: ArcStrengthFit | None = None

    def background(self, sep_bp: int) -> float:
        """The distance two beads that far apart hold with nothing between them. Never under one
        bead, since two beads cannot be closer than touching."""
        sep = max(abs(int(sep_bp)), 1)
        return max(1.0, (sep / max(int(self.s0_bp), 1)) ** self.nu)

    def contact_distance(self, sep_bp: int, q: float) -> float:
        """The distance for a pair with loop strength `q`, from the background at no strength
        to one bead at saturation. Halfway at `q_half`."""
        bg = self.background(sep_bp)
        h = 1.0 / (1.0 + max(q, 0.0) / max(self.q_half, 1e-9))
        return 1.0 + (bg - 1.0) * h

    def arc_distance(self, score: int, sep_bp: int) -> float:
        """The target for an arc of `score` PETs spanning `sep_bp`."""
        span = abs(int(sep_bp))
        q = self.arcs.strength(score, span) if self.arcs is not None else float(score)
        return self.contact_distance(span, q)

    def heatmap_distance(self, freq: float, expected: float, sep_bp: int) -> float:
        """The target for a heatmap cell, the background at that separation scaled by observed
        over expected contact to the minus third, which is the fractal globule relation. No
        contact is no target."""
        if freq <= 1e-12 or expected <= 1e-12:
            return 0.0
        return self.background(sep_bp) * (freq / expected) ** (-1.0 / 3.0)
