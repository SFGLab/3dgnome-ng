"""Unit checks for the polymer law and the contact decay fit it rests on.

    python harness/test_polymer.py

The fit reads the exponent distance grows with, `nu`, off the run's own singletons: contact
probability falls as separation to the power `-3 nu`, so a log log slope of the contact count
against separation gives it. It must recover a known slope from clean data, refuse data that is
not a decay at all, which is what a ChIA-PET singletons file is since it is enrichment filtered
around CTCF, and choose its band from the data's own resolution.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.polymer import (  # noqa: E402
    FALLBACK_NU,
    PolymerLaw,
    fit_arc_strength,
    fit_contact_exponent,
)
from gnome3d.settings import Settings  # noqa: E402
from gnome3d.types import InteractionArc  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def contacts(
    slope: float, n: int, seed: int, binsize: int = 0
) -> list[tuple[str, int, str, int, int]]:
    """Intra chromosomal contacts whose separation density falls as `s ^ slope`, drawn by
    inverse transform over 10 kb to 5 Mb, snapped to `binsize` when given."""
    rng = np.random.default_rng(seed)
    lo, hi = 1e4, 5e6
    a = slope + 1.0  # density s^slope integrates to s^(slope+1)
    u = rng.random(n)
    s = (lo**a + u * (hi**a - lo**a)) ** (1.0 / a)
    p1 = rng.integers(1_000_000, 200_000_000, n)
    if binsize:
        p1 = (p1 // binsize) * binsize
        s = np.maximum(binsize, (s // binsize) * binsize)
    return [("chr1", int(x), "chr1", int(x + y), 1) for x, y in zip(p1, s, strict=True)]


def test_recovers_a_known_slope() -> None:
    print("\n[fit] recovers the exponent from clean contacts")
    for slope in (-0.86, -1.08, -0.6):
        f = fit_contact_exponent(contacts(slope, 400_000, 1))
        check(
            f"slope {slope} comes back within 0.03",
            f.ok and abs(f.slope - slope) < 0.03,
            f"got {f.slope:+.3f}, nu {f.nu:.3f}",
        )
    f = fit_contact_exponent(contacts(-0.86, 400_000, 2))
    check("nu is minus a third of the slope", abs(f.nu + f.slope / 3.0) < 1e-9)


def test_refuses_what_is_not_a_decay() -> None:
    print("\n[fit] refuses data that is not a polymer decay")
    flat = fit_contact_exponent(contacts(-0.1, 200_000, 3))
    check("a flat profile is refused", not flat.ok, flat.reason)
    check("and the fallback exponent is reported", abs(flat.nu - FALLBACK_NU) < 1e-12)
    up = fit_contact_exponent(contacts(0.2, 200_000, 4))
    check("a rising profile is refused", not up.ok, up.reason)
    few = fit_contact_exponent(contacts(-0.86, 300, 5))
    check("too few pairs is refused", not few.ok, few.reason)
    empty = fit_contact_exponent([])
    check("no contacts is refused", not empty.ok, empty.reason)
    inter = fit_contact_exponent([("chr1", 10, "chr2", 500_000, 1)] * 10_000)
    check("inter chromosomal pairs do not count", not inter.ok, inter.reason)


def test_band_follows_the_resolution() -> None:
    print("\n[fit] the band starts where the data can resolve")
    point = fit_contact_exponent(contacts(-0.86, 400_000, 6))
    binned = fit_contact_exponent(contacts(-0.86, 400_000, 7, binsize=25_000))
    check("point data starts at 20 kb", point.lo == 20_000, f"{point.lo}")
    check("25 kb binned data starts at 50 kb", binned.lo == 50_000, f"{binned.lo}")
    check(
        "binned data still recovers the slope",
        binned.ok and abs(binned.slope + 0.86) < 0.05,
        f"{binned.slope:+.3f}",
    )


def test_reports_what_it_used() -> None:
    f = fit_contact_exponent(contacts(-0.86, 400_000, 8))
    check("it says how many pairs it used", f.n_pairs > 100_000, f"{f.n_pairs:,}")
    check("and the band it fitted on", f.lo < f.hi, f"{f.lo // 1000} kb to {f.hi // 1000} kb")


def test_the_law() -> None:
    """One law for every distance. Its unit is the bead, which is the distance at the
    resolution the run declared, so the background at that separation is one by definition."""
    print("\n[law] background, contact and heatmap distances in bead units")
    law = PolymerLaw(nu=0.3, s0_bp=1000, q_half=1.0)
    check("the background at the resolution is one bead", abs(law.background(1000) - 1.0) < 1e-12)
    check("closer than a bead is still one bead", abs(law.background(10) - 1.0) < 1e-12)
    check(
        "ten times the separation is ten to the exponent further",
        abs(law.background(10_000) - 10**0.3) < 1e-9,
        f"{law.background(10_000):.4f}",
    )
    seps = [5_000, 50_000, 500_000]
    slope = float(np.polyfit(np.log(seps), np.log([law.background(x) for x in seps]), 1)[0])
    check("and the background carries the exponent exactly", abs(slope - 0.3) < 1e-9)

    bg = law.background(100_000)
    check("no contact sits on the background", abs(law.contact_distance(100_000, 0.0) - bg) < 1e-12)
    check(
        "a contact at half saturation sits halfway to touching",
        abs(law.contact_distance(100_000, 1.0) - (1.0 + 0.5 * (bg - 1.0))) < 1e-9,
    )
    check(
        "a saturated contact sits at touching, one bead",
        abs(law.contact_distance(100_000, 1e9) - 1.0) < 1e-6,
    )
    ds = [law.contact_distance(100_000, q) for q in (0.0, 0.5, 1.0, 2.0, 8.0)]
    check("more contact is always closer", all(a > b for a, b in zip(ds[:-1], ds[1:], strict=True)))
    check("and never closer than one bead", min(ds) >= 1.0 - 1e-12)

    check(
        "the expected contact at a separation puts a heatmap pair on the background",
        abs(law.heatmap_distance(0.02, 0.02, 100_000) - bg) < 1e-12,
    )
    check(
        "twice the expected contact is two to the minus third closer",
        abs(law.heatmap_distance(0.04, 0.02, 100_000) / bg - 2 ** (-1.0 / 3.0)) < 1e-9,
    )
    check(
        "no contact in a heatmap cell is no target", law.heatmap_distance(0.0, 0.02, 100_000) == 0.0
    )


def arcs(seed: int, n: int, slope: float = -0.5) -> list[InteractionArc]:
    """Arcs whose typical PET count falls as span to `slope`, with scatter."""
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
    """A loop's strength is its PET count over what a loop of its span typically has, which is
    observed over expected read off the run's own arcs rather than a curve from another
    dataset."""
    print("\n[arcs] typical strength against span, from the run's own arcs")
    f = fit_arc_strength(arcs(1, 5000))
    check("the fit is accepted on enough arcs", f.ok, f.reason)
    q10 = f.strength(20, 10_000)
    q100 = f.strength(20, 100_000)
    check(
        "the same PET count is a stronger loop at a longer span",
        q100 > q10 > 0.0,
        f"{q10:.2f} at 10 kb, {q100:.2f} at 100 kb",
    )
    check(
        "a typical arc has strength near one",
        0.7 < f.strength(int(20 * (5.0) ** -0.5), 50_000) < 1.4,
        f"{f.strength(int(20 * 5.0**-0.5), 50_000):.2f}",
    )
    few = fit_arc_strength(arcs(2, 20))
    check(
        "too few arcs is refused, and strength then follows the PET count alone",
        not few.ok and few.strength(4, 50_000) > few.strength(2, 50_000),
    )


def test_settings_route_through_the_law() -> None:
    """Every distance call on Settings answers from the attached law, and a Settings with no law
    attached builds a default from the named fallback and says so, so nothing answers from a
    constant silently."""
    print("\n[settings] every distance call answers from the law")
    s = Settings()
    check("no law is attached until data is loaded", s.polymer is None)
    check("the exponent is measured unless pinned", s.polymer_exponent == 0.0)
    check("half saturation defaults to one typical loop", s.contact_half_saturation == 1.0)
    d = s.genomic_length_to_distance(1000)
    check(
        "a call with no law attached still answers, from the fallback",
        s.polymer is not None and abs(d - 1.0) < 1e-12,
    )
    check("and the default carries the fallback exponent", abs(s.polymer.nu - FALLBACK_NU) < 1e-12)

    s = Settings()
    s.polymer_exponent = 0.2
    s.genomic_length_to_distance(1000)
    check("a pinned exponent is what the default carries", abs(s.polymer.nu - 0.2) < 1e-12)

    s = Settings()
    s.polymer = PolymerLaw(nu=0.25, s0_bp=1000, q_half=1.0)
    check(
        "the chain law call is the background",
        abs(s.genomic_length_to_distance(1000) - 1.0) < 1e-12,
    )
    check(
        "an arc target is a contact distance, closer than the background",
        1.0 <= s.arc_expected_distance(4, 50_000) < s.polymer.background(50_000),
    )
    check(
        "a stronger arc is closer",
        s.arc_expected_distance(40, 50_000) < s.arc_expected_distance(4, 50_000),
    )


def main() -> int:
    print("polymer law checks")
    test_the_law()
    test_arc_strength()
    test_settings_route_through_the_law()
    test_recovers_a_known_slope()
    test_refuses_what_is_not_a_decay()
    test_band_follows_the_resolution()
    test_reports_what_it_used()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
