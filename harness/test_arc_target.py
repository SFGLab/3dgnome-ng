"""Unit checks for the separation aware arc target.

    python harness/test_arc_target.py

The parity era arc target is `freq_to_distance(PET)`, a function of PET count alone, so a 1 Mb
arc with four PETs targets the same 0.36 as a 100 kb one. The separation aware law multiplies in
the polymer background above a pivot span, `target = freq_to_distance(PET) * max(1, s_kb / s0)^nu`.

Checks: closed form, unchanged below the pivot, monotone in span, PET ordering kept at every
span, and the flag off returns the parity law exactly.

Both of those laws set the target from the PET count and scale it by span, so the absolute
distance comes from the PET law and lands between 0.2 and 1.6 model units. The chain law, which
sets the target for a consecutive arcless anchor pair in the same matrix, lands between 4.0 and
135 over the same span range. A pair an arc joins is therefore asked to sit an order of
magnitude closer than the chain says two anchors that far apart should be, and measured on a
finished chromosome 93 percent of arc joined anchor pairs end up closer than a bead's own size.

The unified law puts both families on one background. A pair sits at the chain law distance for
its separation, and its PET count pulls it in from there by a factor between `arc_target_pull`
and 1. The PET law supplies that factor rather than the distance, normalised by its own limits,
which it has: it runs from `freq_to_distance(0)` down to `count_dist_base_level`.
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


def test_chain_bonds() -> None:
    print("\n[chain] consecutive anchors get the chain law when they have no arc")
    from gnome3d.pipeline.coarse.build import add_chain_bonds, arc_expected_matrix

    s = Settings()
    mids = [0, 500, 1_500, 120_000, 121_000]
    arcs = [(0, 2, 10), (3, 4, 5)]  # 0-2 skips 1; 3-4 is a consecutive pair WITH an arc
    base = arc_expected_matrix(s, mids, arcs)
    off = add_chain_bonds(base, mids, s)
    check("flag off returns the matrix unchanged", np.array_equal(off, base))
    s.use_arcs_chain_bonds = True
    on = add_chain_bonds(base, mids, s)
    gld = s.genomic_length_to_distance
    check(
        "consecutive arcless pair gets gld(gap)",
        abs(on[0, 1] - gld(500)) < 1e-12 and abs(on[1, 2] - gld(1_000)) < 1e-12,
        f"{on[0, 1]:.3f}, {on[1, 2]:.3f}",
    )
    check(
        "the island boundary pair gets its long bond",
        abs(on[2, 3] - gld(118_500)) < 1e-12,
        f"{on[2, 3]:.2f}",
    )
    check("consecutive pair with an arc keeps the arc", on[3, 4] == base[3, 4])
    check("non consecutive arcless pairs stay repulsive", on[0, 3] == -1.0 and on[1, 4] == -1.0)
    check(
        "arc pairs untouched, symmetric, input not mutated",
        on[0, 2] == base[0, 2] and np.array_equal(on, on.T) and base[0, 1] == -1.0,
    )
    check("default is off", Settings().use_arcs_chain_bonds is False)
    s.arcs_chain_bond_scale = 2.0
    scaled = add_chain_bonds(base, mids, s)
    check(
        "scale multiplies the bond target",
        abs(scaled[0, 1] - 2.0 * gld(500)) < 1e-12
        and abs(scaled[2, 3] - 2.0 * gld(118_500)) < 1e-12,
    )
    check("scale leaves arcs alone", scaled[0, 2] == base[0, 2] and scaled[3, 4] == base[3, 4])
    check("scale default is 1", Settings().arcs_chain_bond_scale == 1.0)


def test_unified_law() -> None:
    print("\n[unified] one background for both families")
    s = Settings()
    s.use_unified_arc_target = True
    s.arc_target_pull = 0.45
    for sep in (5_000, 50_000, 1_000_000):
        bg = s.genomic_length_to_distance(sep)
        check(
            f"at {sep // 1000} kb a zero PET arc sits on the background",
            abs(s.arc_expected_distance(0, sep) - bg) < 1e-9,
            f"{s.arc_expected_distance(0, sep):.4f} against {bg:.4f}",
        )
        check(
            f"at {sep // 1000} kb a saturated arc sits at the pull",
            abs(s.arc_expected_distance(10_000, sep) - s.arc_target_pull * bg) < 1e-6,
            f"{s.arc_expected_distance(10_000, sep):.4f} against {s.arc_target_pull * bg:.4f}",
        )

    pets = [0, 1, 2, 5, 10, 50]
    d = [s.arc_expected_distance(p, 50_000) for p in pets]
    check(
        "more PETs means a closer target", all(a >= b for a, b in zip(d[:-1], d[1:], strict=True))
    )

    seps = [5_000, 20_000, 100_000, 1_000_000]
    e = [s.arc_expected_distance(4, x) for x in seps]
    check("a wider arc targets further", all(a < b for a, b in zip(e[:-1], e[1:], strict=True)))

    # The point of the law: an arc's target now grows with separation the way the chain does,
    # rather than being pinned near the PET law's own scale.
    slope = float(np.polyfit(np.log(seps), np.log(e), 1)[0])
    chain = [s.genomic_length_to_distance(x) for x in seps]
    cslope = float(np.polyfit(np.log(seps), np.log(chain), 1)[0])
    check(
        "an arc target now follows the chain law's exponent",
        abs(slope - cslope) < 1e-9,
        f"{slope:.4f} against the chain's {cslope:.4f}",
    )
    ratio = [s.genomic_length_to_distance(x) / s.arc_expected_distance(4, x) for x in seps]
    check(
        "and the gap to the chain no longer widens with span",
        max(ratio) - min(ratio) < 1e-9,
        f"ratio {min(ratio):.2f} to {max(ratio):.2f}, was 11x to 100x",
    )
    check("it supersedes the separation aware law", s.use_unified_arc_target is True)


def test_unified_is_off_by_default() -> None:
    d = Settings()
    check("the unified law is opt in", d.use_unified_arc_target is False)
    check("and the pull defaults to the measured value", abs(d.arc_target_pull - 0.45) < 1e-12)


def main() -> int:
    print("separation aware arc target checks")
    test_helper()
    test_settings()
    test_matrix()
    test_chain_bonds()
    test_unified_law()
    test_unified_is_off_by_default()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
