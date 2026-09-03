"""Unit checks for the separation aware arc target.

    python harness/test_arc_target.py

The parity era arc target is `freq_to_distance(PET)`, a function of PET count alone, so a 1 Mb
arc with four PETs targets the same 0.36 as a 100 kb one. The separation aware law multiplies in
the polymer background above a pivot span, `target = freq_to_distance(PET) * max(1, s_kb / s0)^nu`.

Checks: closed form, unchanged below the pivot, monotone in span, PET ordering kept at every
span, and the flag off returns the parity law exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.settings import Settings  # noqa: E402
from gnome3d.util import arc_target_with_separation  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}{('  ' + detail) if detail else ''}")


def test_helper() -> None:
    print("\n[law] the pure helper")
    base, nu, pivot = 0.36, 0.285, 10.0
    check(
        "at the pivot the law is the PET law",
        abs(arc_target_with_separation(base, 10_000, pivot, nu) - base) < 1e-12,
    )
    check(
        "below the pivot the law is the PET law",
        abs(arc_target_with_separation(base, 2_000, pivot, nu) - base) < 1e-12,
    )
    want = base * (100.0**nu)
    got = arc_target_with_separation(base, 1_000_000, pivot, nu)
    check(
        "1 Mb arc is lifted by (s_kb / s0)^nu", abs(got - want) < 1e-12, f"{got:.4f} vs {want:.4f}"
    )
    spans = [5_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000]
    vals = [arc_target_with_separation(base, sp, pivot, nu) for sp in spans]
    check("monotone in span", all(b >= a for a, b in zip(vals[:-1], vals[1:], strict=True)))
    strong = arc_target_with_separation(0.20, 1_000_000, pivot, nu)
    weak = arc_target_with_separation(0.36, 1_000_000, pivot, nu)
    check("PET ordering kept at long span", strong < weak, f"strong {strong:.3f}, weak {weak:.3f}")
    check(
        "exponent zero is the PET law everywhere",
        abs(arc_target_with_separation(base, 5_000_000, pivot, 0.0) - base) < 1e-12,
    )


def test_settings() -> None:
    print("\n[settings] the Settings wrapper and the flag")
    s = Settings()
    pet = 4
    base = s.freq_to_distance(pet)
    check(
        "flag off: arc_expected_distance is freq_to_distance",
        abs(s.arc_expected_distance(pet, 1_000_000) - base) < 1e-15,
    )
    s.use_separation_arc_target = True
    s.arc_target_exponent = 0.285
    s.arc_target_pivot_kb = 10.0
    lifted = s.arc_expected_distance(pet, 1_000_000)
    check(
        "flag on: 1 Mb arc lifted by 100^0.285",
        abs(lifted - base * 100**0.285) < 1e-12,
        f"{base:.3f} -> {lifted:.3f}",
    )
    check("flag on: 5 kb arc unchanged", abs(s.arc_expected_distance(pet, 5_000) - base) < 1e-15)
    check(
        "defaults are off, 0.285, 10 kb",
        (
            Settings().use_separation_arc_target,
            Settings().arc_target_exponent,
            Settings().arc_target_pivot_kb,
        )
        == (False, 0.285, 10.0),
    )


def test_matrix() -> None:
    print("\n[matrix] the expected distance matrix uses the law")
    from gnome3d.pipeline.coarse.build import arc_expected_matrix

    s = Settings()
    mids = [0, 5_000, 200_000, 1_200_000]
    arcs = [(0, 1, 4), (0, 3, 4), (1, 2, 30)]  # (i, j, PET)
    off = arc_expected_matrix(s, mids, arcs)
    check(
        "flag off: -1 for arcless, 0 diagonal, PET law for arcs",
        off[0, 2] == -1.0 and off[1, 1] == 0.0 and abs(off[0, 1] - s.freq_to_distance(4)) < 1e-15,
    )
    s.use_separation_arc_target = True
    on = arc_expected_matrix(s, mids, arcs)
    check("flag on: short arc unchanged", abs(on[0, 1] - off[0, 1]) < 1e-15)
    check(
        "flag on: 1.2 Mb arc lifted",
        abs(on[0, 3] - off[0, 3] * 120**0.285) < 1e-12,
        f"{off[0, 3]:.3f} -> {on[0, 3]:.3f}",
    )
    check("symmetric", np.array_equal(on, on.T))


def main() -> int:
    print("separation aware arc target checks")
    test_helper()
    test_settings()
    test_matrix()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
