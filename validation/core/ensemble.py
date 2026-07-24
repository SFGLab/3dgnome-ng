"""Ensemble reconstruction and summary metrics shared by the validation studies."""

from __future__ import annotations

import numpy as np

from gnome3d.data import ContactData
from gnome3d.settings import Settings
from gnome3d.types import BeadOut, BedRegion
from validation import metrics


def run_ensemble(
    s: Settings,
    data: ContactData,
    chrs_list: list[str],
    region: BedRegion | None,
    n: int,
) -> list[list[BeadOut]]:
    """One ensemble via the public simulate API. Returns beads for the single region's chr."""
    from gnome3d.simulate import simulate

    structures = simulate(s, data, chrs_list, n, region=region)
    return [per_chr[chrs_list[0]] for per_chr in structures]


def to_arrays_list(ens: list) -> tuple[list, list]:
    """Coordinates and midpoints per structure in an ensemble."""
    from validation.metrics import structure

    cl, ml = [], []
    for beads in ens:
        c, m = structure.to_arrays(beads)
        cl.append(c)
        ml.append(m)
    return cl, ml


def summarize(
    ensemble: list[list[BeadOut]],
    contacts: list[tuple[int, int, float]],
    radius: float,
    skip: int,
    overlap_norm_bp: int = 5000,
) -> dict[str, float]:
    """Average the per-structure metrics across the full ensemble, plus ensemble diversity. Reports both
    the raw overlap_frac, which is bead-density-dependent, and overlap_frac_norm, the overlaps after
    coarse-graining to overlap_norm_bp bins so structures at different bead resolutions are
    comparable. Use the norm one for ref-vs-tuned overlap claims.

    For performance, the overlap, distance-scaling and contact-probability metrics all reduce the
    same upper-triangle pairwise distances, and the genomic separations are constant across the
    ensemble because mids are shared. So we compute the separations once and one pdist per structure,
    condensed in triu(n,1) order, then derive all three by reduction, instead of three full O(N²)
    _pairwise passes per structure. It is exact and all structures are kept, with no subsampling."""
    from scipy.spatial.distance import pdist

    from validation.metrics import structure

    coords0, mids0 = metrics.to_arrays(ensemble[0])
    n = len(coords0)
    iu, ju = np.triu_indices(n, k=1)
    sep = np.abs(mids0[iu] - mids0[ju]).astype(np.float64)  # constant across ensemble
    nonbond = (ju - iu) > skip  # exclude |i-j| <= skip as bonded neighbours, matches overlap_fraction

    rgs, bonds, overlaps, ov_norm, rhos, dscals, cprobs, extents = [], [], [], [], [], [], [], []
    coords_list = []
    for beads in ensemble:
        coords, mids = metrics.to_arrays(beads)
        coords_list.append(coords)
        rgs.append(metrics.radius_of_gyration(coords))
        bonds.append(float(np.median(metrics.bond_lengths(coords))))
        d = pdist(coords)  # condensed upper-tri distances in triu(n,1) order, computed once
        overlaps.append(float((d[nonbond] < radius).mean()) if nonbond.any() else 0.0)
        ov_norm.append(metrics.overlap_fraction_binned(coords, mids, overlap_norm_bp, skip_neighbors=skip)[0])
        extents.append(metrics.max_extent(coords))
        rho, _ = metrics.self_consistency(coords, mids, contacts)
        rhos.append(rho)
        dscals.append(structure._loglog_bins(sep, d, 20)[2])
        cprobs.append(structure._loglog_bins(sep, (d < radius).astype(np.float64), 20)[2])
    dab = metrics.dab_matrix(coords_list)
    nanmean = lambda xs: float(np.nanmean(xs)) if xs else float("nan")
    return {
        "n_beads": float(len(ensemble[0])),
        "rg": nanmean(rgs),
        "bond": nanmean(bonds),
        "overlap_frac": nanmean(overlaps),
        "overlap_frac_norm": nanmean(ov_norm),
        "max_extent": nanmean(extents),
        "selfconsistency_rho": nanmean(rhos),
        "dist_scaling_exp": nanmean(dscals),
        "contact_prob_exp": nanmean(cprobs),
        "diversity_dab": metrics.ensemble_diversity(dab),
    }
