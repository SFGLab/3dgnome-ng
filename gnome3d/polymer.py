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

from gnome3d.types import SingletonContact

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
