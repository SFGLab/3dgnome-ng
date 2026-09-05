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


def main() -> int:
    print("cross block relaxation checks")
    test_gate_function()
    test_relax_separates()
    test_flag_off()
    test_it_skips_when_there_is_nothing_to_fix()
    test_it_still_runs_when_blocks_do_interpenetrate()
    test_the_default_is_off()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
