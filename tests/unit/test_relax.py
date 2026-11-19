"""Unit checks for the cross block relaxation.

    python tests/unit/test_relax.py

The pass runs the smooth kernel over the whole chromosome with excluded volume on every pair
and the anchors fixed. The count of cross block bead pairs inside the excluded volume radius
must reach zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    """A dense random walk of n beads in a box of half width 3 around `centre`, with anchors at
    both ends and bonds of length `bond`.
    """
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
    b = coil(200_000, 60, np.zeros(3), rng, 1.0)  # same centre, full overlap
    c = coil(400_000, 60, np.array([100.0, 0, 0]), rng, 1.0)  # far away
    n_ab, touched_ab = cross_block_contacts([a, b], 1.0)
    n_ac, touched_ac = cross_block_contacts([a, c], 1.0)
    check("overlapping coils register contacts", n_ab > 50, f"{n_ab} pairs, {touched_ab} beads")
    check("separated coils register none", n_ac == 0 and touched_ac == 0)


def test_relax_separates() -> None:
    print("\n[relax] two overlapping coils are pushed apart with anchors fixed")
    from gnome3d.mc.numba import seed_numba

    rng = np.random.default_rng(2)
    a = coil(0, 150, np.zeros(3), rng, 1.0)
    b = coil(200_000, 150, np.array([0.5, 0.0, 0.0]), rng, 1.0)
    before, _ = cross_block_contacts([a, b], 1.0)
    # The pass uses the RNG state of the process, which is seeded here. The bond ratio is 1.47
    # against a limit of 1.5.
    seed_numba(5)
    np.random.seed(5)
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


def test_local_window_restricts_who_may_move() -> None:
    """With a window only the beads near a contact move. The window follows the chain across
    block boundaries. Most of the structure must stay fixed.
    """
    rng = np.random.default_rng(11)
    # The coils have half width 3. At an offset of 5 only the facing edges touch.
    a = coil(0, 60, np.zeros(3), rng, 1.0)
    b = coil(200_000, 60, np.array([5.0, 0.0, 0.0]), rng, 1.0)
    c = coil(400_000, 60, np.array([100.0, 0.0, 0.0]), rng, 1.0)  # far from both

    def moved_count(before: list[list[BeadOut]], after: list[list[BeadOut]]) -> int:
        """Returns the number of beads the MC moved. The float32 round trip changes every bead
        by about one ulp, which the tolerance ignores.
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
    """The pass does not seed. An isolated call repeats only from the same RNG state."""
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


def test_within_blocks_clears_inner_overlaps() -> None:
    """Overlaps inside a block move only with `within_blocks`."""
    rng = np.random.default_rng(5)
    a = coil(0, 80, np.zeros(3), rng, 1.0)
    far = coil(500_000, 20, np.array([100.0, 0.0, 0.0]), rng, 1.0)
    pa = positions(a)
    d = np.linalg.norm(pa[:, None] - pa[None, :], axis=2)
    iu = np.triu_indices(len(a), 2)
    before = int((d[iu] < 0.7).sum())
    s = settings(relax_local_window=1)
    off = positions(relax_blocks([a, far], s)[0])
    s.relax_within_blocks = True
    on = positions(relax_blocks([a, far], s)[0])

    def inner(q: np.ndarray) -> int:
        dd = np.linalg.norm(q[:, None] - q[None, :], axis=2)
        return int((dd[iu] < 0.7).sum())

    check(
        "off leaves a lone dense block as it was", inner(off) == before, f"{inner(off)} of {before}"
    )
    check("on clears most inner overlaps", inner(on) < 0.3 * before, f"{inner(on)} of {before}")
    alone = positions(relax_blocks([a], s)[0])
    check(
        "on clears them in a lone block too",
        inner(alone) < 0.3 * before,
        f"{inner(alone)} of {before}",
    )


def main() -> int:
    print("cross block relaxation checks")
    test_gate_function()
    test_relax_separates()
    test_flag_off()
    test_local_window_restricts_who_may_move()
    test_within_blocks_clears_inner_overlaps()
    test_local_window_is_off_by_default()
    test_it_is_reproducible_from_a_given_rng_state()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
