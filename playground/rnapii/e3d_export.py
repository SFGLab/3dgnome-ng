"""gnome3d models in the layout the enhancer3d cross line analysis reads, and arm E at ten models.

enhancer3d (`enhancer3d/models.py` `load_ensemble`) reads a `.cif` per model with x, y and z in
tokens 10 to 12 of each ATOM line, and one `.txt` beside the first whose rows are the chain's
positions, a uniform grid. gnome3d beads are irregular, so every 1 kb grid position takes the
coordinates of the bead whose midpoint is nearest, when that midpoint is within 5 kb, and is
nan otherwise. enhancer3d drops a pair whose distance is nan, so a gene more than 5 kb from any
bead leaves the statistics rather than reading a far bead's position. The same rule applies to
every arm, E included, since E's chain is already uniform at 1 kb and every position is covered.

    python playground/rnapii/e3d_export.py <arm_dir> <out_dir> <chrom> [--min-covered 0.5]
    python playground/rnapii/e3d_export.py --subsample <E_dir> <out_dir> [--n 10] [--seed 20261111]
    python playground/rnapii/e3d_export.py --hcm-to-cif <hcm_dir> <out_dir> [--n 10] [--seed 20261111]

Arm E's models are cudaMMC `.hcm` trees. `--hcm-to-cif` links a seeded ten of them under
`<out_dir>/hcm` and writes each one's third level, anchors and loop beads, as a gnome3d cif in
`<out_dir>`, which the battery, the pile-up and this exporter then read like any other arm.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def to_grid(
    pos: np.ndarray, mids: np.ndarray, start: int, end: int, res: int = 1000, max_gap: int = 5000
) -> tuple[np.ndarray, np.ndarray, float]:
    """Coordinates on the grid `start, start + res, ...` up to `end`, the grid, and the covered
    fraction. `mids` must be sorted; a position further than `max_gap` from every midpoint is
    nan."""
    positions = np.arange(start, end + 1, res, dtype=np.int64)
    k = np.clip(np.searchsorted(mids, positions), 1, len(mids) - 1)
    left, right = mids[k - 1], mids[k]
    nearest = np.where(positions - left <= right - positions, k - 1, k)
    gap = np.abs(mids[nearest] - positions)
    grid = pos[nearest].astype(np.float64)
    grid[gap > max_gap] = np.nan
    covered = float(np.mean(gap <= max_gap))
    return grid, positions, covered


def write_model(stem: Path, grid: np.ndarray, positions: np.ndarray) -> None:
    """`<stem>.cif` and `<stem>.txt` in the cudaMMC smooth layout, nan for an empty position."""
    lines = ["data_gnome3d_export", "loop_", "_atom_site.group_PDB"]
    for i, (x, y, z) in enumerate(grid, start=1):
        lines.append(f"ATOM {i} C CA . GLY A 1 {i} ? {x:.3f} {y:.3f} {z:.3f} 1.00 0.0 ? {i} GLY A CA 1")
    Path(f"{stem}.cif").write_text("\n".join(lines) + "\n")
    rows = [f"{len(positions)}"] + [
        f"{x:f} {y:f} {z:f} {p:d}" for (x, y, z), p in zip(grid, positions.tolist(), strict=True)
    ]
    Path(f"{stem}.txt").write_text("\n".join(rows) + "\n")


def subsample(src: Path, dest: Path, n: int, seed: int, pattern: str = "*.cif") -> list[Path]:
    """Symlinks in `dest` to `n` of the files in `src` matching `pattern`, chosen by `seed`, each
    with its `.txt` when one exists. A rerun with the same arguments changes nothing."""
    import os

    files = sorted(src.glob(pattern))
    if len(files) < n:
        raise SystemExit(f"{src}: {len(files)} files match {pattern}, fewer than {n}")
    pick = sorted(np.random.default_rng(seed).choice(len(files), size=n, replace=False).tolist())
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for i in pick:
        main_file = files[i]
        for f in (main_file, main_file.with_suffix(".txt")):
            link = dest / f.name
            if f.exists() and not os.path.lexists(link):
                link.symlink_to(f.resolve())
        out.append(dest / main_file.name)
    return out


def hcm_level3(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions, starts and ends of the third tree level of a cudaMMC `.hcm`, the anchors and
    loop beads, in tree order. Reads both file formats, the rule of enhancer3d's `read_hcm`."""
    tok = path.read_text().split()
    prev = tok[2] != "-1"
    n = int(tok[0])
    n_fac = int(tok[3]) if prev else int(tok[1])
    k = 4 + n_fac
    start = np.empty(n, np.int64)
    end = np.empty(n, np.int64)
    xyz = np.empty((n, 3), np.float64)
    children: list[list[int]] = []
    for i in range(n):
        off = 1 if prev else 0  # the previous format stores genomic_pos first, the new one orientation third
        start[i], end[i] = int(tok[k + off]), int(tok[k + off + 1])
        xyz[i] = (float(tok[k + 3]), float(tok[k + 4]), float(tok[k + 5]))
        nc = int(tok[k + 6])
        children.append([int(x) for x in tok[k + 7 : k + 7 + nc]])
        k += 7 + nc
    root = int(tok[2]) if prev else int(tok[k + 1])
    cur = [root]
    for _ in range(3):
        cur = [c for x in cur for c in (children[x] or [x])]
    idx = np.asarray(cur)
    return xyz[idx], start[idx], end[idx]


def hcm_to_cif(src: Path, dest: Path) -> None:
    """One `.hcm` as a gnome3d cif of its third level, every bead a subanchor since a cudaMMC
    tree carries no bead kind the battery's block inference could use."""
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from gnome3d.io import write_cif
    from gnome3d.types import BeadOut

    pos, start, end = hcm_level3(src)
    write_cif(str(dest), [BeadOut(int(a), int(b), *map(float, p), "subanchor") for p, a, b in zip(pos, start, end, strict=True)])


def main() -> None:
    args = sys.argv[1:]
    n = int(args[args.index("--n") + 1]) if "--n" in args else 10
    seed = int(args[args.index("--seed") + 1]) if "--seed" in args else 20261111
    if args and args[0] == "--subsample":
        for p in subsample(Path(args[1]), Path(args[2]), n, seed):
            print(p.name)
        return
    if args and args[0] == "--hcm-to-cif":
        dest = Path(args[2])
        picked = subsample(Path(args[1]), dest / "hcm", n, seed, pattern="loops_*.hcm")
        for i, h in enumerate(picked):
            out = dest / f"{h.stem}.cif"
            if not out.exists():
                hcm_to_cif(h, out)
            print(f"{h.name} -> {out.name}")
        return
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from playground.validation_battery import load

    floor = float(args[args.index("--min-covered") + 1]) if "--min-covered" in args else 0.5
    arm, out, chrom = Path(args[0]), Path(args[1]), args[2]
    out.mkdir(parents=True, exist_ok=True)
    cifs = sorted(arm.glob("*.cif"))
    models = [load(c) for c in cifs]
    start = min(int(m[2].min()) for m in models) // 1000 * 1000
    end = max(int(m[2].max()) for m in models)
    for i, (c, (pos, _, mid, _)) in enumerate(zip(cifs, models, strict=True)):
        o = np.argsort(mid)
        grid, positions, covered = to_grid(pos[o], mid[o], start, end)
        print(f"  {c.name}: {covered:.3f} of {len(positions)} positions within 5 kb of a bead")
        if covered < floor:
            raise SystemExit(f"{c}: covered fraction {covered:.3f} under {floor}, refusing")
        write_model(out / f"loops_{chrom}_{i}.hcm.smooth", grid, positions)


if __name__ == "__main__":
    main()
