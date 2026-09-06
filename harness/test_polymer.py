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

from gnome3d.polymer import FALLBACK_NU, fit_contact_exponent  # noqa: E402

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


def main() -> int:
    print("polymer law checks")
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
