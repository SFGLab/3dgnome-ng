"""Run MultiMM on one of our arms' inputs and collect its ensemble for the validation battery.

    python playground/multimm_arm.py LOOPS.bedpe REGION OURS_DIR OUT_ROOT [--compartments TRACK]
                                     [--platform OpenCL] [--multimm MultiMM] [--n 5]

The loops file is our seven column clusters bedpe, which is the format MultiMM reads. The bead
count is taken from the first cif in OURS_DIR so the two arms model the region at the same
resolution, and MultiMM's modelling level preset is left unset because it would reset that
count. With `--compartments` the signed eigenvector track is written as a bed of A and B rows
and MultiMM's compartment block force is switched on; without it the run is loops only, which
is the like for like comparison with our arcs.

MultiMM's ensemble members share one deterministic Hilbert start and one minimisation, so the
minimised structures are one structure repeated and only the molecular dynamics after it, whose
velocities are seeded per member, differ. Both are collected, into `<OUT_ROOT>_min` and
`<OUT_ROOT>_md`, since the minimised one is what MultiMM reports and the dynamics is the only
ensemble it makes.

MultiMM's beads carry no genomic coordinates. Each collected cif gives bead k of n the span
`[lo + (hi - lo) k / n, lo + (hi - lo) (k + 1) / n)` and marks every bead a subanchor, so the
battery bins it like any other structure and its overlap count lands in the subanchor column.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnome3d.io import write_cif  # noqa: E402
from gnome3d.types import BeadOut  # noqa: E402


def read_multimm_cif(path: Path) -> np.ndarray:
    """Positions from a MultiMM mmCIF, in file order, located by the `_atom_site` header."""
    cols: list[str] = []
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        if line.startswith("_atom_site."):
            cols.append(line.split(".", 1)[1].strip())
        elif line.startswith("ATOM"):
            f = line.split()
            rows.append([float(f[cols.index(c)]) for c in ("Cartn_x", "Cartn_y", "Cartn_z")])
    return np.array(rows, dtype=np.float64).reshape(-1, 3)


def uniform_beads(xyz: np.ndarray, lo: int, hi: int) -> list[BeadOut]:
    """One subanchor bead per row, spans tiling `[lo, hi)` in equal shares."""
    n = len(xyz)
    edges = [lo + ((hi - lo) * k) // n for k in range(n + 1)]
    return [
        BeadOut(edges[k], edges[k + 1], float(x), float(y), float(z), "subanchor")
        for k, (x, y, z) in enumerate(xyz)
    ]


def compartment_bed(track: Path, chrom: str, lo: int, hi: int) -> list[tuple[str, int, int, str]]:
    """A and B rows from a signed bedGraph, clipped to what MultiMM's region filter keeps.

    MultiMM keeps a row only if its start is above `lo` and its end below `hi`, so a row that
    straddles either edge is clipped to one base inside it rather than lost. Rows with a zero
    or missing value are unassigned and dropped.
    """
    out: list[tuple[str, int, int, str]] = []
    for line in track.read_text().splitlines():
        f = line.split()
        if len(f) < 4 or f[0] != chrom:
            continue
        try:
            v = float(f[3])
        except ValueError:
            continue
        if not np.isfinite(v) or v == 0.0:
            continue
        s, e = max(int(f[1]), lo + 1), min(int(f[2]), hi - 1)
        if s >= e:
            continue
        out.append((chrom, s, e, "A" if v > 0 else "B"))
    return out


def phase_by_loops(
    rows: list[tuple[str, int, int, str]], loops: Path, chrom: str, lo: int, hi: int
) -> list[tuple[str, int, int, str]]:
    """Orient the labels so the side holding more loop anchors is A.

    The eigenvector sign is arbitrary and the pipeline phases it by anchor counts when no
    phasing signal is given. MultiMM takes the labels as written, so the bed has to be phased
    here the same way. Each loop contributes both anchors, counted at their midpoints.
    """
    edges = sorted((s, e, lab) for _, s, e, lab in rows)
    starts = np.array([s for s, _, _ in edges], dtype=np.int64)
    ends = np.array([e for _, e, _ in edges], dtype=np.int64)
    labels = [lab for _, _, lab in edges]
    count = {"A": 0, "B": 0}
    for line in loops.read_text().splitlines():
        f = line.split()
        if len(f) < 7 or f[0] != chrom or f[3] != chrom:
            continue
        for a, b in ((int(f[1]), int(f[2])), (int(f[4]), int(f[5]))):
            m = (a + b) // 2
            if not lo <= m < hi:
                continue
            k = int(np.searchsorted(starts, m, side="right")) - 1
            if k >= 0 and m < ends[k]:
                count[labels[k]] += 1
    if count["B"] > count["A"]:
        return [(c, s, e, "B" if lab == "A" else "A") for c, s, e, lab in rows]
    return rows


def write_compartment_bed(rows: list[tuple[str, int, int, str]], path: Path) -> None:
    path.write_text("".join(f"{c}\t{s}\t{e}\t{lab}\n" for c, s, e, lab in rows))


def write_multimm_config(
    path: Path,
    *,
    loops: Path,
    out_path: Path,
    chrom: str,
    lo: int,
    hi: int,
    n_beads: int,
    n_ensemble: int,
    platform: str,
    compartments: Path | None,
) -> None:
    """MultiMM's `[Main]` config for one region at our bead count.

    Molecular dynamics is on with the step count MultiMM's region preset uses, and plots are
    off since they need a display. Keys are written by hand to keep their case.
    """
    lines = [
        "[Main]",
        f"PLATFORM = {platform}",
        f"LOOPS_PATH = {loops}",
        f"OUT_PATH = {out_path}",
        f"N_BEADS = {n_beads}",
        f"CHROM = {chrom}",
        f"LOC_START = {lo}",
        f"LOC_END = {hi}",
        "GENERATE_ENSEMBLE = True",
        f"N_ENSEMBLE = {n_ensemble}",
        "SIM_RUN_MD = True",
        "SIM_N_STEPS = 10000",
        "SAVE_PLOTS = False",
        f"COB_USE_COMPARTMENT_BLOCKS = {'True' if compartments else 'False'}",
    ]
    if compartments:
        lines.append(f"COMPARTMENT_PATH = {compartments}")
    path.write_text("\n".join(lines) + "\n")


def collect_ensemble(
    out_root: Path, n_ensemble: int, lo: int, hi: int, arm_min: Path, arm_md: Path
) -> int:
    """Relabel each member's minimised and after dynamics cif into the two arm directories.

    MultiMM writes member i to `<out_root>_i`. Returns how many members had a model directory.
    """
    arm_min.mkdir(parents=True, exist_ok=True)
    arm_md.mkdir(parents=True, exist_ok=True)
    found = 0
    for i in range(1, n_ensemble + 1):
        model = Path(f"{out_root}_{i}") / "model"
        if not model.is_dir():
            continue
        found += 1
        for src, dst in (
            (model / "MultiMM_minimized.cif", arm_min / f"member_{i}.cif"),
            (model / "MultiMM_afterMD.cif", arm_md / f"member_{i}.cif"),
        ):
            if src.is_file():
                write_cif(str(dst), uniform_beads(read_multimm_cif(src), lo, hi))
    return found


def count_beads(cif: Path) -> int:
    return sum(1 for ln in cif.read_text().splitlines() if ln.startswith("ATOM"))


def _flag(name: str, default: str) -> str:
    if name not in sys.argv:
        return default
    k = sys.argv.index(name)
    v = sys.argv[k + 1]
    del sys.argv[k : k + 2]
    return v


def main() -> None:
    compartments = _flag("--compartments", "")
    platform = _flag("--platform", "OpenCL")
    multimm = _flag("--multimm", "MultiMM")
    n_ensemble = int(_flag("--n", "5"))
    loops, region, ours, out_root = (Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]))
    chrom, span = region.split(":")
    lo, hi = (int(v) for v in span.split("-"))
    cifs = sorted(ours.glob("*.cif"))
    if not cifs:
        raise SystemExit(f"no cif in {ours} to take the bead count from")
    n_beads = count_beads(cifs[0])
    out_root.parent.mkdir(parents=True, exist_ok=True)
    bed: Path | None = None
    if compartments:
        bed = out_root.parent / f"{out_root.name}_compartments.bed"
        rows = phase_by_loops(compartment_bed(Path(compartments), chrom, lo, hi), loops, chrom, lo, hi)
        write_compartment_bed(rows, bed)
        print(f"compartment bed: {len(rows)} rows, A {sum(r[3] == 'A' for r in rows)} B {sum(r[3] == 'B' for r in rows)}")
    cfg = out_root.parent / f"{out_root.name}.ini"
    write_multimm_config(
        cfg,
        loops=loops.resolve(),
        out_path=out_root.resolve(),
        chrom=chrom,
        lo=lo,
        hi=hi,
        n_beads=n_beads,
        n_ensemble=n_ensemble,
        platform=platform,
        compartments=bed.resolve() if bed else None,
    )
    print(f"MultiMM {region} at {n_beads} beads from {cifs[0].name}, {n_ensemble} members, {platform}", flush=True)
    log = out_root.parent / f"{out_root.name}.log"
    with open(log, "w") as fh:
        rc = subprocess.run([multimm, "-c", str(cfg)], stdout=fh, stderr=subprocess.STDOUT).returncode
    arm_min = out_root.parent / f"{out_root.name}_min"
    arm_md = out_root.parent / f"{out_root.name}_md"
    found = collect_ensemble(out_root, n_ensemble, lo, hi, arm_min, arm_md)
    print(f"MultiMM exit {rc}, {found} of {n_ensemble} members collected into {arm_min.name} and {arm_md.name}")
    if rc != 0 or found != n_ensemble:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
