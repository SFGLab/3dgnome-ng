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

from e3d_export import common_genes, hcm_level3, subsample, to_grid, write_model  # noqa: E402

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


def test_hcm_level3(tmp: Path) -> None:
    # previous format: n n_arcs root n_factors, factor names, then per cluster
    # genomic_pos start end x y z n_children children..., then 8 ints per arc
    rows = [
        "0 0 300 0 0 0 1 1",
        "0 0 300 0 0 0 1 2",
        "0 0 300 0 0 0 3 3 4 5",
        "50 0 100 1.0 0 0 0",
        "150 100 200 2.0 0 0 0",
        "250 200 300 3.0 0 0 0",
    ]
    f = tmp / "loops_chr1_0.hcm"
    f.write_text("6 1 0 1 CTCF\n" + "\n".join(rows) + "\n3 5 9 0 50 250 0 0\n")
    pos, start, end = hcm_level3(f)
    check(
        "the third level of the tree is the leaf beads with their spans",
        pos[:, 0].tolist() == [1.0, 2.0, 3.0]
        and start.tolist() == [0, 100, 200]
        and end.tolist() == [100, 200, 300],
        str(pos[:, 0].tolist()),
    )


def test_subsample_hcm(tmp: Path) -> None:
    src = tmp / "H"
    src.mkdir()
    for i in range(20):
        (src / f"loops_chr1_{i}.hcm").write_text("x")
    dest = tmp / "h10"
    got = subsample(src, dest, n=10, seed=1, pattern="loops_*.hcm")
    again = subsample(src, dest, n=10, seed=1, pattern="loops_*.hcm")
    check(
        "hcm models subsample without companions and a rerun is a no op",
        len(got) == 10 and got == again and not list(dest.glob("*.txt")),
    )


def test_common_genes(tmp: Path) -> None:
    pos, mids = chain()
    for k, gap in enumerate((5000, 2000)):
        grid, positions, _ = to_grid(pos, mids, 0, 200_000, max_gap=gap)
        write_model(tmp / f"arm{k}" / "loops_chr1_0.hcm.smooth", grid, positions) if (
            tmp / f"arm{k}"
        ).mkdir() is None else None
    genes = tmp / "genes.tsv"
    genes.write_text(
        "gene_id\tchrom\tstart\tend\tstrand\nA\tchr1\t50000\t60000\t+\nB\tchr1\t100000\t104000\t-\nC\tchr1\t110000\t111000\t+\n"
    )
    out = tmp / "common.tsv"
    n = common_genes(genes, out, [tmp / "arm0", tmp / "arm1"])
    kept = [ln.split("\t")[0] for ln in out.read_text().splitlines()[1:]]
    # B's TSS (end, minus strand) at 104,000 is 4 kb from a bead: covered at 5 kb, not at 2 kb
    check("only genes covered in every arm are kept", kept == ["A"] and n == 1, str(kept))


def main() -> int:
    test_grid_gap()
    with tempfile.TemporaryDirectory() as d:
        test_write_read_back(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_subsample(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_hcm_level3(Path(d))
        test_subsample_hcm(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_common_genes(Path(d))
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
