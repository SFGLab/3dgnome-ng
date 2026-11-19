"""Unit checks for the cross block relaxation."""

from __future__ import annotations

import numpy as np

from gnome3d.pipeline.relax import cross_block_contacts, relax_blocks
from gnome3d.settings import Settings
from gnome3d.types import BeadOut


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
    """The cross block contact count."""
    rng = np.random.default_rng(1)
    a = coil(0, 60, np.zeros(3), rng, 1.0)
    b = coil(200_000, 60, np.zeros(3), rng, 1.0)  # same centre, full overlap
    c = coil(400_000, 60, np.array([100.0, 0, 0]), rng, 1.0)  # far away
    n_ab, touched_ab = cross_block_contacts([a, b], 1.0)
    n_ac, touched_ac = cross_block_contacts([a, c], 1.0)
    assert n_ab > 50, f"overlapping coils register contacts: {n_ab} pairs, {touched_ab} beads"
    assert n_ac == 0 and touched_ac == 0, "separated coils register none"


def test_relax_separates() -> None:
    """Two overlapping coils are pushed apart with anchors fixed."""
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
    assert after == 0, (
        f"contacts inside the radius go to zero: {before} -> {after} ({touched} beads touched)"
    )
    for name, x, y in (("first", a, out[0]), ("second", b, out[1])):
        pa = positions(x)
        pb = positions(y)
        anch = [i for i, bd in enumerate(x) if bd.kind == "anchor"]
        assert np.allclose(pa[anch], pb[anch]), f"{name} coil: anchors did not move"
        bonds_a = np.linalg.norm(np.diff(pa, axis=0), axis=1)
        bonds_b = np.linalg.norm(np.diff(pb, axis=0), axis=1)
        assert np.all(bonds_b < 1.5 * bonds_a) and np.all(bonds_b > 0.5 * bonds_a), (
            f"{name} coil: bonds kept within 50 percent"
            + ": "
            + f"max ratio {np.max(bonds_b / bonds_a):.2f}"
        )
    assert all(
        (p.start, p.end, p.kind) == (q.start, q.end, q.kind)
        for x, y in zip([a, b], out, strict=True)
        for p, q in zip(x, y, strict=True)
    ), "bead count, ranges and kinds preserved"


def test_flag_off() -> None:
    """Off returns the input."""
    rng = np.random.default_rng(3)
    a = coil(0, 20, np.zeros(3), rng, 1.0)
    b = coil(100_000, 20, np.zeros(3), rng, 1.0)
    s = settings(use_cross_block_relax=False)
    assert relax_blocks([a, b], s) == [a, b], "flag off returns the blocks unchanged"
    assert Settings().use_cross_block_relax is False, "default is off"


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
    assert 0 < local < everything, (
        f"a window moves far fewer beads than letting every subanchor move: {local} against {everything} of {3 * 60}"
    )


def test_local_window_is_off_by_default() -> None:
    assert Settings().relax_local_window < 0, (
        "the window defaults to every subanchor, which is what it did before"
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
    assert same, "from the same RNG state it gives the same structure"


def test_within_blocks_clears_inner_overlaps() -> None:
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

    assert inner(off) == before, (
        f"off leaves a lone dense block as it was: {inner(off)} of {before}"
    )
    assert inner(on) < 0.3 * before, f"on clears most inner overlaps: {inner(on)} of {before}"
    alone = positions(relax_blocks([a], s)[0])
    assert inner(alone) < 0.3 * before, (
        f"on clears them in a lone block too: {inner(alone)} of {before}"
    )
