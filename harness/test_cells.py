"""Unit checks for the excluded volume cell grid.

    python harness/test_cells.py

The grid exists to make the excluded volume term visit the beads that are near instead of all
of them. It is a pure optimisation, so the one property that matters is that it returns exactly
what the full scan returns, bit for bit, on every bead of a real structure and on the awkward
cases: a bead alone in its cell, a bead at the corner of the grid, everything piled at one
point, and a candidate list too long for the buffer.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.mc.numba.cells import (  # noqa: E402
    BUF,
    build_grid,
    cell_of,
    grid_shape,
    local_excl_cells,
    relink,
)
from gnome3d.mc.numba.terms import _local_excl_nb  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}{('  ' + detail) if detail else ''}")


def compare(
    pos: np.ndarray, r0: float, weight: float, skip: int, buf_size: int = BUF
) -> tuple[int, float, int]:
    """Every bead through both paths. Returns (mismatches, worst difference, overflows)."""
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
        if got != want:
            bad += 1
            worst = max(worst, abs(got - want))
    return bad, worst, over


def test_random() -> None:
    print("\n[random] the grid returns the scan's own number")
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
        check(
            f"{name}: identical for every bead",
            bad == 0 and over == 0,
            f"{bad} differ, worst {worst:.2e}, {over} overflow",
        )


def test_edges() -> None:
    print("\n[edges] the awkward configurations")
    single = np.zeros((1, 3))
    bad, _, _ = compare(single, 2.0, 10.0, 1)
    check("a single bead", bad == 0)
    same = np.zeros((300, 3))
    bad, worst, over = compare(same, 2.0, 10.0, 1)
    check("every bead at one point", bad == 0 and over == 0, f"{bad} differ, {over} overflow")
    rng = np.random.default_rng(3)
    far = np.vstack([rng.normal(0, 1, (200, 3)), rng.normal(500, 1, (200, 3))])
    bad, _, _ = compare(np.ascontiguousarray(far), 2.0, 10.0, 1)
    check("two clumps far apart", bad == 0)
    line = np.column_stack([np.arange(500.0) * 0.01, np.zeros(500), np.zeros(500)])
    for skip in (0, 1, 5):
        bad, _, _ = compare(line, 2.0, 10.0, skip)
        check(f"skip_neighbors {skip}", bad == 0)


def test_overflow_is_safe() -> None:
    print("\n[overflow] a buffer too small reports itself instead of being wrong")
    rng = np.random.default_rng(5)
    pos = np.ascontiguousarray(rng.normal(0, 1.0, (2000, 3)))
    bad, _, over = compare(pos, 2.0, 10.0, 1, buf_size=64)
    check(
        "overflow is signalled, never a wrong number",
        bad == 0 and over > 0,
        f"{over} beads overflowed, {bad} wrong",
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
    check(
        "identical on 8000 real beads",
        bad == 0 and over == 0,
        f"{bad} differ, worst {worst:.2e}, {over} overflow",
    )


def test_moves_keep_it_exact() -> None:
    """The grid is updated in place as beads move, which is the whole point of the linked list,
    so it has to stay exact over a long run of moves rather than only when freshly built."""
    print("\n[moves] the grid stays exact while beads move")
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
            if got != want:
                bad += 1
    check("still identical after 4000 moves", bad == 0, f"{bad} of 20 spot checks differ")
    seen = np.zeros(n, dtype=np.int64)
    for k in range(head.shape[0]):
        i = head[k]
        while i != -1:
            seen[i] += 1
            i = nxt[i]
    check("every bead is linked exactly once", int(seen.min()) == 1 and int(seen.max()) == 1)
    check(
        "every bead is in the cell its position says",
        all(where[i] == cell_of(pos[i, 0], pos[i, 1], pos[i, 2], lo, dim, c) for i in range(n)),
    )


def main() -> int:
    print("excluded volume cell grid checks")
    test_random()
    test_edges()
    test_overflow_is_safe()
    test_moves_keep_it_exact()
    test_real_structure()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
