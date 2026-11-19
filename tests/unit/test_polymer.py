"""Unit checks for the polymer law and the contact decay fit."""

from __future__ import annotations

import numpy as np

from gnome3d.polymer import (
    FALLBACK_NU,
    PolymerLaw,
    fit_arc_strength,
    fit_contact_exponent,
)
from gnome3d.settings import Settings
from gnome3d.types import InteractionArc


def contacts(
    slope: float, n: int, seed: int, binsize: int = 0
) -> list[tuple[str, int, str, int, int]]:
    """Returns intra chromosomal contacts whose separation density falls as `s ^ slope`, drawn
    by inverse transform over 10 kb to 5 Mb and snapped to `binsize` when given.
    """
    rng = np.random.default_rng(seed)
    lo, hi = 1e4, 5e6
    a = slope + 1.0  # the density s^slope integrates to s^(slope+1)
    u = rng.random(n)
    s = (lo**a + u * (hi**a - lo**a)) ** (1.0 / a)
    p1 = rng.integers(1_000_000, 200_000_000, n)
    if binsize:
        p1 = (p1 // binsize) * binsize
        s = np.maximum(binsize, (s // binsize) * binsize)
    return [("chr1", int(x), "chr1", int(x + y), 1) for x, y in zip(p1, s, strict=True)]


def test_recovers_a_known_slope() -> None:
    """Recovers the exponent from clean contacts."""
    for slope in (-0.86, -1.08, -0.6):
        f = fit_contact_exponent(contacts(slope, 400_000, 1))
        assert f.ok and abs(f.slope - slope) < 0.03, (
            f"slope {slope} comes back within 0.03" + ": " + f"got {f.slope:+.3f}, nu {f.nu:.3f}"
        )
    f = fit_contact_exponent(contacts(-0.86, 400_000, 2))
    assert abs(f.nu + f.slope / 3.0) < 1e-9, "nu is minus a third of the slope"


def test_refuses_what_is_not_a_decay() -> None:
    """Refuses data that is not a polymer decay."""
    flat = fit_contact_exponent(contacts(-0.1, 200_000, 3))
    assert not flat.ok, "a flat profile is refused: " + str(flat.reason)
    assert abs(flat.nu - FALLBACK_NU) < 1e-12, "and the fallback exponent is reported"
    up = fit_contact_exponent(contacts(0.2, 200_000, 4))
    assert not up.ok, "a rising profile is refused: " + str(up.reason)
    few = fit_contact_exponent(contacts(-0.86, 300, 5))
    assert not few.ok, "too few pairs is refused: " + str(few.reason)
    empty = fit_contact_exponent([])
    assert not empty.ok, "no contacts is refused: " + str(empty.reason)
    inter = fit_contact_exponent([("chr1", 10, "chr2", 500_000, 1)] * 10_000)
    assert not inter.ok, "inter chromosomal pairs do not count: " + str(inter.reason)


def test_band_follows_the_resolution() -> None:
    """The band starts where the data can resolve."""
    point = fit_contact_exponent(contacts(-0.86, 400_000, 6))
    binned = fit_contact_exponent(contacts(-0.86, 400_000, 7, binsize=25_000))
    assert point.lo == 20_000, f"point data starts at 20 kb: {point.lo}"
    assert binned.lo == 50_000, f"25 kb binned data starts at 50 kb: {binned.lo}"
    assert binned.ok and abs(binned.slope + 0.86) < 0.05, (
        f"binned data still recovers the slope: {binned.slope:+.3f}"
    )


def test_reports_what_it_used() -> None:
    f = fit_contact_exponent(contacts(-0.86, 400_000, 8))
    assert f.n_pairs > 100_000, f"it says how many pairs it used: {f.n_pairs:,}"
    assert f.lo < f.hi, f"and the band it fitted on: {f.lo // 1000} kb to {f.hi // 1000} kb"


def test_the_law() -> None:
    """One bead is the distance at the resolution of the run and the background there is one."""
    print("\n[law] background, contact and heatmap distances in bead units")
    law = PolymerLaw(nu=0.3, s0_bp=1000, q_half=1.0)
    assert abs(law.background(1000) - 1.0) < 1e-12, "the background at the resolution is one bead"
    assert abs(law.background(10) - 1.0) < 1e-12, "closer than a bead is still one bead"
    assert abs(law.background(10_000) - 10**0.3) < 1e-9, (
        f"ten times the separation is ten to the exponent further: {law.background(10_000):.4f}"
    )
    seps = [5_000, 50_000, 500_000]
    slope = float(np.polyfit(np.log(seps), np.log([law.background(x) for x in seps]), 1)[0])
    assert abs(slope - 0.3) < 1e-9, "and the background carries the exponent exactly"

    bg = law.background(100_000)
    assert abs(law.contact_distance(100_000, 0.0) - bg) < 1e-12, "no contact sits on the background"
    assert abs(law.contact_distance(100_000, 1.0) - (1.0 + 0.5 * (bg - 1.0))) < 1e-9, (
        "a contact at half saturation sits halfway to touching"
    )
    assert abs(law.contact_distance(100_000, 1e9) - 1.0) < 1e-6, (
        "a saturated contact sits at touching, one bead"
    )
    ds = [law.contact_distance(100_000, q) for q in (0.0, 0.5, 1.0, 2.0, 8.0)]
    assert all(a > b for a, b in zip(ds[:-1], ds[1:], strict=True)), "more contact is always closer"
    assert min(ds) >= 1.0 - 1e-12, "and never closer than one bead"

    assert abs(law.heatmap_distance(0.02, 0.02, 100_000) - bg) < 1e-12, (
        "the expected contact at a separation puts a heatmap pair on the background"
    )
    assert abs(law.heatmap_distance(0.04, 0.02, 100_000) / bg - 2 ** (-1.0 / 3.0)) < 1e-9, (
        "twice the expected contact is two to the minus third closer"
    )
    assert law.heatmap_distance(0.0, 0.02, 100_000) == 0.0, (
        "no contact in a heatmap cell is no target"
    )


def arcs(seed: int, n: int, slope: float = -0.5) -> list[InteractionArc]:
    """Returns arcs whose typical PET count falls as span to `slope`, with scatter."""
    rng = np.random.default_rng(seed)
    span = 10 ** rng.uniform(4.0, 6.0, n)
    typical = 20.0 * (span / 1e4) ** slope
    score = np.maximum(1, np.round(typical * rng.lognormal(0.0, 0.4, n))).astype(int)
    out: list[InteractionArc] = []
    for k in range(n):
        st = int(rng.integers(1_000_000, 100_000_000))
        out.append(
            InteractionArc(0, 1, int(score[k]), genomic_start=st, genomic_end=st + int(span[k]))
        )
    return out


def test_arc_strength() -> None:
    """Typical strength against span, from the run's own arcs."""
    f = fit_arc_strength(arcs(1, 5000))
    assert f.ok, "the fit is accepted on enough arcs: " + str(f.reason)
    q10 = f.strength(20, 10_000)
    q100 = f.strength(20, 100_000)
    assert q100 > q10 > 0.0, (
        f"the same PET count is a stronger loop at a longer span: {q10:.2f} at 10 kb, {q100:.2f} at 100 kb"
    )
    assert 0.7 < f.strength(int(20 * (5.0) ** -0.5), 50_000) < 1.4, (
        f"a typical arc has strength near one: {f.strength(int(20 * 5.0**-0.5), 50_000):.2f}"
    )
    few = fit_arc_strength(arcs(2, 20))
    assert not few.ok and few.strength(4, 50_000) > few.strength(2, 50_000), (
        "too few arcs is refused, and strength then follows the PET count alone"
    )


def test_settings_route_through_the_law() -> None:
    """Every distance call on Settings uses the attached law. A Settings without one builds a
    default from the fallback and warns.
    """
    print("\n[settings] every distance call answers from the law")
    s = Settings()
    assert s.polymer is None, "no law is attached until data is loaded"
    assert s.polymer_exponent == 0.0, "the exponent is measured unless pinned"
    assert s.contact_half_saturation == 1.0, "half saturation defaults to one typical loop"
    d = s.genomic_length_to_distance(1000)
    assert s.polymer is not None and abs(d - 1.0) < 1e-12, (
        "a call with no law attached still answers, from the fallback"
    )
    assert abs(s.polymer.nu - FALLBACK_NU) < 1e-12, "and the default carries the fallback exponent"

    s = Settings()
    s.polymer_exponent = 0.2
    s.genomic_length_to_distance(1000)
    assert abs(s.polymer.nu - 0.2) < 1e-12, "a pinned exponent is what the default carries"

    s = Settings()
    s.polymer = PolymerLaw(nu=0.25, s0_bp=1000, q_half=1.0)
    assert abs(s.genomic_length_to_distance(1000) - 1.0) < 1e-12, (
        "the chain law call is the background"
    )
    assert 1.0 <= s.arc_expected_distance(4, 50_000) < s.polymer.background(50_000), (
        "an arc target is a contact distance, closer than the background"
    )
    assert s.arc_expected_distance(40, 50_000) < s.arc_expected_distance(4, 50_000), (
        "a stronger arc is closer"
    )


def test_block_size() -> None:
    """The radius of gyration of a chain under the law has no free constant. At nu one half Rg
    squared is N over 6 bonds.
    """
    law = PolymerLaw(nu=0.5, s0_bp=1000, q_half=1.0)
    rg = law.radius_of_gyration(6_000)
    assert abs(rg - 1.0) < 1e-9, f"Gaussian chain of six bonds has Rg 1: {rg:.6f}"
    rg24 = law.radius_of_gyration(24_000)
    assert abs(rg24 - 2.0) < 1e-9, "Rg grows as the square root of span at nu one half"
    r = law.confinement_radius(6_000)
    assert abs(r - (5.0 / 3.0) ** 0.5) < 1e-9, (
        f"the sphere with that Rg has radius root five thirds of it: {r:.6f}"
    )
    law3 = PolymerLaw(nu=0.3, s0_bp=1000, q_half=1.0)
    rg_law = 1500.0**0.3 / (2.0 * 1.6 * 1.3) ** 0.5
    assert abs(law3.radius_of_gyration(1_500_000) - rg_law) < 1e-9, (
        "the general formula, S^nu over root 2 (2 nu + 1)(nu + 1)"
    )
    assert abs(law3.radius_of_gyration(10) - 1.0 / (2.0 * 1.6 * 1.3) ** 0.5) < 1e-9, (
        "a span under one bead is one bead's chain"
    )


def test_counts_not_rows() -> None:
    """The fit weights rows by their count. A deep map has every pixel and its row count per
    separation carries no decay.
    """
    rng = np.random.default_rng(3)
    rows: list[tuple[str, int, str, int, int]] = []
    n_bins = 400
    for i in range(n_bins):
        for j in range(i + 1, n_bins):
            sep = (j - i) * 25_000
            if sep > 1_500_000:
                continue
            mean = 2000.0 * (sep / 50_000.0) ** -0.9
            rows.append(("chr1", i * 25_000, "chr1", j * 25_000, int(max(1, rng.poisson(mean)))))
    fit = fit_contact_exponent(rows)
    assert fit.ok and abs(fit.slope + 0.9) < 0.1, (
        f"every pixel present, the decay is read from the counts: slope {fit.slope:.3f} ok={fit.ok} {fit.reason}"
    )


def test_grid_step_survives_a_truncated_end_bin() -> None:
    """The grid step is the separation most pairs share. The last bin of a chromosome is short
    and gives a smaller separation.
    """
    rng = np.random.default_rng(4)
    rows: list[tuple[str, int, str, int, int]] = []
    n_bins = 300
    mids = [i * 25_000 + 12_500 for i in range(n_bins)] + [n_bins * 25_000 + 6_000]
    for i in range(len(mids)):
        for j in range(i + 1, len(mids)):
            sep = mids[j] - mids[i]
            if sep > 1_500_000:
                continue
            mean = 2000.0 * (sep / 50_000.0) ** -0.9
            rows.append(("chr1", mids[i], "chr1", mids[j], int(max(1, rng.poisson(mean)))))
    fit = fit_contact_exponent(rows)
    assert fit.ok and fit.lo == 50_000 and abs(fit.slope + 0.9) < 0.1, (
        f"an off grid end bin does not move the grid step: band from {fit.lo // 1000} kb, slope {fit.slope:.3f} {fit.reason}"
    )
