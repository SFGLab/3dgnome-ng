"""Unit checks for the excluded volume cell grid."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from gnome3d.mc.numba.cells import (
    BUF,
    build_grid,
    cell_of,
    grid_shape,
    local_excl_cells,
    relink,
)
from gnome3d.mc.numba.terms import (
    _local_excl_nb,
    _wall_local_cells_nb,
    _wall_local_nb,
)

# The scan and the grid add the same pairs in the same order. Both kernels are compiled with
# fastmath and on x86-64 the compiler adds the pairs of the scan in another order. The two
# numbers then differ in the last bits.
ROUNDING = 1e-12


def same(got: float, want: float) -> bool:
    return abs(got - want) <= ROUNDING * max(1.0, abs(want))


def compare(
    pos: np.ndarray, r0: float, weight: float, skip: int, buf_size: int = BUF
) -> tuple[int, float, int]:
    """Runs every bead through both paths. Returns the beads that differ by more than rounding,
    the worst difference and the overflows.
    """
    lo, dim, c = grid_shape(pos, r0)
    head, nxt, where = build_grid(pos, lo, dim, c)
    buf = np.empty(buf_size, dtype=np.int32)
    bad = 0
    worst = 0.0
    over = 0
    for p in range(pos.shape[0]):
        want = _local_excl_nb(pos, p, r0, weight, skip)
        got = local_excl_cells(pos, p, r0, weight, skip, lo, dim, c, head, nxt, buf)
        if got < 0.0:
            over += 1
            continue
        if not same(got, want):
            bad += 1
        worst = max(worst, abs(got - want))
    return bad, worst, over


def test_random() -> None:
    """The grid returns the scan's number to rounding."""
    rng = np.random.default_rng(0)
    for name, pos in (
        ("uniform cloud", rng.normal(0, 8, (2000, 3))),
        ("dense clump", rng.normal(0, 1.2, (1500, 3))),
        (
            "flat sheet",
            np.column_stack(
                [rng.normal(0, 8, 1200), rng.normal(0, 8, 1200), rng.normal(0, 0.05, 1200)]
            ),
        ),
        ("one line", np.column_stack([np.arange(1000.0), np.zeros(1000), np.zeros(1000)])),
    ):
        bad, worst, over = compare(np.ascontiguousarray(pos), 2.0, 10.0, 1)
        assert bad == 0 and over == 0, (
            f"{name}: equal to rounding for every bead"
            + ": "
            + f"{bad} differ, worst {worst:.2e}, {over} overflow"
        )


def test_edges() -> None:
    """The awkward configurations."""
    single = np.zeros((1, 3))
    bad, _, _ = compare(single, 2.0, 10.0, 1)
    assert bad == 0, "a single bead"
    same = np.zeros((300, 3))
    bad, worst, over = compare(same, 2.0, 10.0, 1)
    assert bad == 0 and over == 0, f"every bead at one point: {bad} differ, {over} overflow"
    rng = np.random.default_rng(3)
    far = np.vstack([rng.normal(0, 1, (200, 3)), rng.normal(500, 1, (200, 3))])
    bad, _, _ = compare(np.ascontiguousarray(far), 2.0, 10.0, 1)
    assert bad == 0, "two clumps far apart"
    line = np.column_stack([np.arange(500.0) * 0.01, np.zeros(500), np.zeros(500)])
    for skip in (0, 1, 5):
        bad, _, _ = compare(line, 2.0, 10.0, skip)
        assert bad == 0, f"skip_neighbors {skip}"


def test_overflow_is_safe() -> None:
    """A buffer too small reports itself instead of being wrong."""
    rng = np.random.default_rng(5)
    pos = np.ascontiguousarray(rng.normal(0, 1.0, (2000, 3)))
    bad, _, over = compare(pos, 2.0, 10.0, 1, buf_size=64)
    assert bad == 0 and over > 0, (
        f"overflow is signalled, never a wrong number: {over} beads overflowed, {bad} wrong"
    )


def test_real_structure() -> None:
    cif = Path.home() / "Desktop" / "stitch_viz" / "dchainboth" / "chr1_1_60000000_s1.cif"
    if not cif.exists():
        print("\n[real] skipped, no structure at " + str(cif))
        return
    print("\n[real] a finished chromosome region")
    rows = [ln.split() for ln in open(cif) if ln.startswith("ATOM")]
    pos = np.array([[float(r[10]), float(r[11]), float(r[12])] for r in rows], dtype=np.float64)
    starts = np.array([int(r[16]) for r in rows])
    pos = np.ascontiguousarray(pos[np.argsort(starts)])
    bond = float(np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    sub = np.ascontiguousarray(pos[:8000])
    bad, worst, over = compare(sub, 1.5 * bond, 10.0, 1)
    assert bad == 0 and over == 0, (
        f"equal to rounding on 8000 real beads: {bad} differ, worst {worst:.2e}, {over} overflow"
    )


def test_moves_keep_it_exact() -> None:
    """The grid stays right while beads move."""
    rng = np.random.default_rng(11)
    n = 1500
    pos = np.ascontiguousarray(rng.normal(0, 4, (n, 3)))
    r0, weight, skip = 2.0, 10.0, 1
    lo, dim, c = grid_shape(pos, r0)
    head, nxt, where = build_grid(pos, lo, dim, c)
    buf = np.empty(BUF, dtype=np.int32)
    bad = 0
    for step in range(4000):
        i = int(rng.integers(0, n))
        pos[i] += rng.uniform(-1.0, 1.0, 3)
        relink(head, nxt, where, i, cell_of(pos[i, 0], pos[i, 1], pos[i, 2], lo, dim, c))
        if step % 200 == 0:
            p = int(rng.integers(0, n))
            want = _local_excl_nb(pos, p, r0, weight, skip)
            got = local_excl_cells(pos, p, r0, weight, skip, lo, dim, c, head, nxt, buf)
            if not same(got, want):
                bad += 1
    assert bad == 0, f"still equal to rounding after 4000 moves: {bad} of 20 spot checks differ"
    seen = np.zeros(n, dtype=np.int64)
    for k in range(head.shape[0]):
        i = head[k]
        while i != -1:
            seen[i] += 1
            i = nxt[i]
    assert int(seen.min()) == 1 and int(seen.max()) == 1, "every bead is linked exactly once"
    assert all(
        where[i] == cell_of(pos[i, 0], pos[i, 1], pos[i, 2], lo, dim, c) for i in range(n)
    ), "every bead is in the cell its position says"


def test_beads_outside_the_box() -> None:
    """A bead beyond the box of the grid is in an edge cell and a query from beyond the box
    searches the edge cells.
    """
    print("\n[outside] beads beyond the box the grid was built on")
    rng = np.random.default_rng(5)
    n = 1200
    pos = np.ascontiguousarray(rng.normal(0, 3, (n, 3)))
    r0, weight, skip = 2.0, 10.0, 1
    lo, dim, c = grid_shape(pos, r0)
    head, nxt, where = build_grid(pos, lo, dim, c)
    buf = np.empty(BUF, dtype=np.int32)
    pos *= 3.0
    for i in range(n):
        relink(head, nxt, where, i, cell_of(pos[i, 0], pos[i, 1], pos[i, 2], lo, dim, c))
    outside = int(((pos < lo - c) | (pos > lo + dim * c + c)).any(axis=1).sum())
    bad = wall_bad = 0
    depth_worst = 0.0
    for p in range(n):
        want = _local_excl_nb(pos, p, r0, weight, skip)
        got = local_excl_cells(pos, p, r0, weight, skip, lo, dim, c, head, nxt, buf)
        bad += int(not same(got, want))
        cnt0, dep0 = _wall_local_nb(pos, p, r0, skip)
        cnt1, dep1 = _wall_local_cells_nb(pos, p, r0, skip, lo, dim, c, head, nxt)
        wall_bad += int(cnt0 != cnt1)
        depth_worst = max(depth_worst, abs(dep0 - dep1))
    assert outside > n // 4, f"the test has beads well outside the box: {outside} of {n}"
    assert bad == 0, (
        f"the excluded volume is equal to rounding out there: {bad} of {n} beads differ"
    )
    assert wall_bad == 0, f"the wall counts the same pairs out there: {wall_bad} of {n} differ"
    assert depth_worst < 1e-9, f"the wall depth agrees to rounding: worst {depth_worst:.2e}"
