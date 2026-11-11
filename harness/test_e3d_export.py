"""Unit checks for the export of gnome3d models into the enhancer3d layout,
`playground/rnapii/e3d_export.py`.

    python harness/test_e3d_export.py

enhancer3d reads a uniform 1 kb chain. A grid position takes its nearest bead's coordinates
when that bead's midpoint is within 5 kb and is empty otherwise, so a gene far from every bead
drops out instead of reading a distance it does not have.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "playground" / "rnapii"))

from e3d_export import subsample, to_grid, write_model  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}   {detail}")


def chain() -> tuple[np.ndarray, np.ndarray]:
    """Beads every 2 kb over 0-100 kb and 120-200 kb, a 20 kb gap between them."""
    mids = np.concatenate([np.arange(0, 100_001, 2000), np.arange(120_000, 200_001, 2000)])
    pos = np.stack([mids / 1000.0, np.zeros(len(mids)), np.zeros(len(mids))], axis=1)
    return pos, mids.astype(np.int64)


def test_grid_gap() -> None:
    pos, mids = chain()
    grid, positions, covered = to_grid(pos, mids, 0, 200_000, res=1000, max_gap=5000)
    empty = positions[np.isnan(grid[:, 0])]
    check(
        "only positions more than 5 kb from every bead are empty",
        empty.min() == 106_000 and empty.max() == 114_000 and len(empty) == 9,
        f"{empty.min()}..{empty.max()} n={len(empty)}",
    )
    check(
        "the covered fraction counts the rest", np.isclose(covered, 1 - 9 / 201), f"{covered:.4f}"
    )
    k = int(np.where(positions == 51_000)[0][0])
    check(
        "a position takes its nearest bead",
        np.isclose(grid[k, 0], 50.0) or np.isclose(grid[k, 0], 52.0),
    )


def test_write_read_back(tmp: Path) -> None:
    pos, mids = chain()
    grid, positions, _ = to_grid(pos, mids, 0, 200_000)
    write_model(tmp / "m0", grid, positions)
    rows = (tmp / "m0.txt").read_text().splitlines()
    atoms = [
        ln.split() for ln in (tmp / "m0.cif").read_text().splitlines() if ln.startswith("ATOM")
    ]
    check(
        "the txt holds the count then one row per position",
        int(rows[0]) == len(positions)
        and int(rows[1].split()[-1]) == 0
        and int(rows[2].split()[-1]) == 1000,
    )
    xs = np.array([float(a[10]) for a in atoms])
    check(
        "the cif carries x in token 10 with empty positions as nan",
        len(atoms) == len(positions) and np.allclose(xs, grid[:, 0], equal_nan=True),
    )


def test_subsample(tmp: Path) -> None:
    src = tmp / "E"
    src.mkdir()
    for i in range(100):
        (src / f"loops_chr1_{i}.hcm.smooth.cif").write_text("x")
        (src / f"loops_chr1_{i}.hcm.smooth.txt").write_text("x")
    a, b = tmp / "a", tmp / "b"
    subsample(src, a, n=10, seed=20261111)
    subsample(src, b, n=10, seed=20261111)
    got_a = sorted(p.name for p in a.glob("*.cif"))
    check(
        "ten models with their txt, the same ten for the same seed",
        len(got_a) == 10
        and got_a == sorted(p.name for p in b.glob("*.cif"))
        and all((a / n.replace(".cif", ".txt")).exists() for n in got_a),
    )


def main() -> int:
    test_grid_gap()
    with tempfile.TemporaryDirectory() as d:
        test_write_read_back(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_subsample(Path(d))
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
