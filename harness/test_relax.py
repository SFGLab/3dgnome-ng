"""Unit checks for the cross block relaxation.

    python harness/test_relax.py

After the stitch nothing acts between beads of different blocks, so two coils can pass through
each other. The relaxation runs the smooth kernel over the whole chromosome with excluded volume
on every pair, anchors held fixed, so the coils re route around each other and the arcs and the
stitch are kept. The gate is the count of cross block bead pairs inside the excluded volume
radius, which must reach zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.pipeline.relax import cross_block_contacts, relax_blocks  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402
from gnome3d.types import BeadOut  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}{('  ' + detail) if detail else ''}")


def coil(
    start_bp: int, n: int, centre: np.ndarray, rng: np.random.Generator, bond: float
) -> list[BeadOut]:
    """A random walk of n beads confined to a box of half width 3, so the coil is dense, with
    anchors at both ends and bonds of length `bond`, around centre."""
    p = [np.zeros(3)]
    for _ in range(n - 1):
        st = rng.normal(size=3)
        st *= bond / np.linalg.norm(st)
        p.append(np.clip(p[-1] + st, -3.0, 3.0))
    q = np.array(p)
    q = q - q.mean(axis=0) + centre
    out: list[BeadOut] = []
    for k in range(n):
        s = start_bp + k * 1000
        kind = "anchor" if k in (0, n - 1) else "subanchor"
        out.append(BeadOut(s, s + 500, float(q[k, 0]), float(q[k, 1]), float(q[k, 2]), kind))  # type: ignore[arg-type]
    return out


def positions(bl: list[BeadOut]) -> np.ndarray:
    return np.array([[b.x, b.y, b.z] for b in bl])


def settings(**kw: object) -> Settings:
    s = Settings()
    s.use_cross_block_relax = True
    s.mc_executor_smooth = "serial"
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_gate_function() -> None:
    print("\n[gate] the cross block contact count")
    rng = np.random.default_rng(1)
    a = coil(0, 60, np.zeros(3), rng, 1.0)
    b = coil(200_000, 60, np.zeros(3), rng, 1.0)  # same centre: fully overlapping
    c = coil(400_000, 60, np.array([100.0, 0, 0]), rng, 1.0)  # far away
    n_ab, touched_ab = cross_block_contacts([a, b], 1.0)
    n_ac, touched_ac = cross_block_contacts([a, c], 1.0)
    check("overlapping coils register contacts", n_ab > 50, f"{n_ab} pairs, {touched_ab} beads")
    check("separated coils register none", n_ac == 0 and touched_ac == 0)


def test_relax_separates() -> None:
    print("\n[relax] two overlapping coils are pushed apart with anchors fixed")
    rng = np.random.default_rng(2)
    a = coil(0, 150, np.zeros(3), rng, 1.0)
    b = coil(200_000, 150, np.array([0.5, 0.0, 0.0]), rng, 1.0)
    before, _ = cross_block_contacts([a, b], 1.0)
    out = relax_blocks([a, b], settings())
    after, touched = cross_block_contacts(out, 1.0)
    check(
        "contacts inside the radius go to zero",
        after == 0,
        f"{before} -> {after} ({touched} beads touched)",
    )
    for name, x, y in (("first", a, out[0]), ("second", b, out[1])):
        pa = positions(x)
        pb = positions(y)
        anch = [i for i, bd in enumerate(x) if bd.kind == "anchor"]
        check(f"{name} coil: anchors did not move", np.allclose(pa[anch], pb[anch]))
        bonds_a = np.linalg.norm(np.diff(pa, axis=0), axis=1)
        bonds_b = np.linalg.norm(np.diff(pb, axis=0), axis=1)
        check(
            f"{name} coil: bonds kept within 50 percent",
            np.all(bonds_b < 1.5 * bonds_a) and np.all(bonds_b > 0.5 * bonds_a),
            f"max ratio {np.max(bonds_b / bonds_a):.2f}",
        )
    check(
        "bead count, ranges and kinds preserved",
        all(
            (p.start, p.end, p.kind) == (q.start, q.end, q.kind)
            for x, y in zip([a, b], out, strict=True)
            for p, q in zip(x, y, strict=True)
        ),
    )


def test_flag_off() -> None:
    print("\n[flag] off returns the input")
    rng = np.random.default_rng(3)
    a = coil(0, 20, np.zeros(3), rng, 1.0)
    b = coil(100_000, 20, np.zeros(3), rng, 1.0)
    s = settings(use_cross_block_relax=False)
    check("flag off returns the blocks unchanged", relax_blocks([a, b], s) == [a, b])
    check("default is off", Settings().use_cross_block_relax is False)


def test_it_skips_when_there_is_nothing_to_fix() -> None:
    """The pass costs the same however little work it has, so it has to be able to decline.

    Measured on a trio run: it took an hour and fifty five minutes per structure whatever the
    input, once to move two beads out of 129,457, because it anneals the whole chromosome until
    its own convergence test fires. The contact count it already computes for its log line is the
    natural gate.
    """
    rng = np.random.default_rng(7)
    s = settings()
    # Two blocks placed far apart: no bead of one is anywhere near a bead of the other.
    a = coil(0, 40, np.zeros(3), rng, 1.0)
    b = coil(200_000, 40, np.array([500.0, 0.0, 0.0]), rng, 1.0)
    s.relax_min_contact_fraction = 0.0
    ran = relax_blocks([a, b], s)
    s.relax_min_contact_fraction = 0.01
    skipped = relax_blocks([a, b], s)
    same_as_input = all(
        x.x == y.x and x.y == y.y and x.z == y.z
        for blk_in, blk_out in zip([a, b], skipped, strict=True)
        for x, y in zip(blk_in, blk_out, strict=True)
    )
    check(
        "with a threshold set it returns untouched blocks when nothing is touching",
        same_as_input,
        "",
    )
    moved_when_ungated = any(
        x.x != y.x or x.y != y.y or x.z != y.z
        for blk_in, blk_out in zip([a, b], ran, strict=True)
        for x, y in zip(blk_in, blk_out, strict=True)
    )
    check(
        "and without one it still runs, so the default is unchanged",
        moved_when_ungated,
        "",
    )


def test_it_still_runs_when_blocks_do_interpenetrate() -> None:
    """A threshold must not stop it doing the job it exists for."""
    rng = np.random.default_rng(8)
    s = settings()
    s.relax_min_contact_fraction = 0.01
    a = coil(0, 40, np.zeros(3), rng, 1.0)
    b = coil(200_000, 40, np.zeros(3), rng, 1.0)  # same centre: fully overlapping
    out = relax_blocks([a, b], s)
    moved = any(
        x.x != y.x or x.y != y.y or x.z != y.z
        for blk_in, blk_out in zip([a, b], out, strict=True)
        for x, y in zip(blk_in, blk_out, strict=True)
    )
    check("it still runs when the blocks are on top of each other", moved)


def test_the_default_is_off() -> None:
    check(
        "the threshold defaults to zero, so the pass runs as it always did",
        Settings().relax_min_contact_fraction == 0.0,
    )


def test_local_window_restricts_who_may_move() -> None:
    """Only beads near an offending contact need to move, and the round count follows that.

    Measured on a real 129,457 bead chromosome: rounds scale with the movable bead count, 114 at
    1,177 movable and 691 at 11,766, which extrapolates to the 6,201 a trio run actually took
    with all 117,660 subanchors movable. A few dozen beads touch anything, so almost all of that
    is proposing moves for beads with nothing to fix.

    The window runs along the chain and the chain crosses block boundaries, so a bead of an
    untouched block next to a touched one may move. What has to hold is that most of the
    structure is frozen.
    """
    rng = np.random.default_rng(11)
    # Coils are half width 3, so offsetting by 5 makes only the facing edges touch, which is
    # what a real chromosome looks like: a handful of beads in contact, not two blocks on top of
    # each other.
    a = coil(0, 60, np.zeros(3), rng, 1.0)
    b = coil(200_000, 60, np.array([5.0, 0.0, 0.0]), rng, 1.0)
    c = coil(400_000, 60, np.array([100.0, 0.0, 0.0]), rng, 1.0)  # far from both

    def moved_count(before: list[list[BeadOut]], after: list[list[BeadOut]]) -> int:
        """Beads the MC actually moved.

        The pass narrows coordinates to float32 for the kernel and widens them back, so every
        bead's value changes by about a float32 ulp whether or not it moved. Moves are half a
        bond, so a tolerance well above that rounding separates the two cleanly.
        """
        return sum(
            1
            for bi, bo in zip(before, after, strict=True)
            for x, y in zip(bi, bo, strict=True)
            if abs(x.x - y.x) > 1e-3 or abs(x.y - y.y) > 1e-3 or abs(x.z - y.z) > 1e-3
        )

    s = settings()
    s.relax_local_window = 2
    local = moved_count([a, b, c], relax_blocks([a, b, c], s))
    s.relax_local_window = -1
    everything = moved_count([a, b, c], relax_blocks([a, b, c], s))
    check(
        "a window moves far fewer beads than letting every subanchor move",
        0 < local < everything,
        f"{local} against {everything} of {3 * 60}",
    )


def test_local_window_is_off_by_default() -> None:
    check(
        "the window defaults to every subanchor, which is what it did before",
        Settings().relax_local_window < 0,
    )


def test_it_is_reproducible_from_a_given_rng_state() -> None:
    """The pass does not seed; it runs on whatever state the stages left, so a whole run is
    reproducible while an isolated call is only reproducible from the same state."""
    from gnome3d.mc.numba import seed_numba

    rng = np.random.default_rng(12)
    a = coil(0, 50, np.zeros(3), rng, 1.0)
    b = coil(200_000, 50, np.zeros(3), rng, 1.0)
    s = settings()
    seed_numba(5)
    np.random.seed(5)
    first = relax_blocks([a, b], s)
    seed_numba(5)
    np.random.seed(5)
    second = relax_blocks([a, b], s)
    same = all(
        x.x == y.x and x.y == y.y and x.z == y.z
        for bi, bo in zip(first, second, strict=True)
        for x, y in zip(bi, bo, strict=True)
    )
    check("from the same RNG state it gives the same structure", same)


def main() -> int:
    print("cross block relaxation checks")
    test_gate_function()
    test_relax_separates()
    test_flag_off()
    test_it_skips_when_there_is_nothing_to_fix()
    test_it_still_runs_when_blocks_do_interpenetrate()
    test_the_default_is_off()
    test_local_window_restricts_who_may_move()
    test_local_window_is_off_by_default()
    test_it_is_reproducible_from_a_given_rng_state()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
