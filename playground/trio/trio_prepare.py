"""Spike. Turn fetched trio files into the per sample layout the pipeline expects.

Reads data/_trio/<SAMPLE>/ and writes data/<SAMPLE>/ plus data/_hic/<SAMPLE>/, which is what
slurm/ensemble/setup_cell.sh validates and what validation tracks searches. After this runs,
the existing tooling takes over unchanged.

    python playground/trio/trio_prepare.py --samples HG00512
    python -m validation tracks --cell HG00512 --skip-compartments --skip-signal
    python playground/trio/trio_segments.py --samples HG00512
    python slurm/ensemble/prep_singletons.py --mcool ... --chroms chr1-chr22,chrX --out ...

The loop input is the high quality set, PET3+ loops whose anchors both overlap a CTCF binding
site. Anchors are the distinct anchor intervals of those loops, which is how the GM12878
anchor file in this repo relates to its own cluster file.
"""

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import trio_orient  # noqa: E402
import trio_samples  # noqa: E402

CENTROMERES = Path("data/GM12878/hg38_centromeres.bed")


def read_loops(path: Path) -> list[tuple[str, int, int, str, int, int, int]]:
    out = []
    for n, line in enumerate(path.open(), 1):
        f = line.split()
        if len(f) != 7:
            raise SystemExit(f"{path}:{n} has {len(f)} columns, expected 7")
        out.append((f[0], int(f[1]), int(f[2]), f[3], int(f[4]), int(f[5]), int(f[6])))
    return out


def write_clusters(loops: list, dest: Path) -> None:
    with dest.open("w") as fh:
        for c1, s1, e1, c2, s2, e2, pet in loops:
            fh.write(f"{c1}\t{s1}\t{e1}\t{c2}\t{s2}\t{e2}\t{pet}\n")


def anchor_union(loops: list) -> list[tuple[str, int, int]]:
    """Distinct anchor intervals over both ends, in genomic order."""
    seen: set[tuple[str, int, int]] = set()
    for c1, s1, e1, c2, s2, e2, _ in loops:
        seen.add((c1, s1, e1))
        seen.add((c2, s2, e2))
    return sorted(seen, key=lambda a: (a[0], a[1], a[2]))


def _coord(text: str) -> tuple[int, bool]:
    """Parse a BED coordinate, tolerating the scientific notation some peak files carry.

    Three lines across the nine samples are written as floats, so 150994094 appears as
    1.51e+08 and lands about 6 kb off. Coerced rather than dropped, and counted so the
    coercion is visible.
    """
    try:
        return int(text), False
    except ValueError:
        return int(float(text)), True


def peak_overlap(anchors: list[tuple[str, int, int]], peaks: Path) -> float:
    """Fraction of anchors overlapping a called peak.

    The high quality loops are defined by both anchors sitting on a CTCF site from the family's
    peak union, so this should be near total. A sample that reads low was filtered against a
    different reference than its own family, which is otherwise invisible.
    """
    by: dict[str, list[tuple[int, int]]] = defaultdict(list)
    coerced = 0
    for line in peaks.open():
        f = line.split()
        if len(f) >= 3:
            start, a = _coord(f[1])
            end, b = _coord(f[2])
            coerced += a or b
            by[f[0]].append((start, end))
    if coerced:
        print(f"[prepare]   {coerced} peak lines in {peaks.name} had float coordinates, coerced")
    for c in by:
        by[c].sort()
    hit = 0
    for c, s, e in anchors:
        for ps, pe in by.get(c, []):
            if ps >= e:
                break
            if pe > s:
                hit += 1
                break
    return hit / max(len(anchors), 1)


def convert_hic(hic: Path, mcool: Path) -> None:
    """juicer .hic -> mcool, then give chromosomes the chr prefix the loops use.

    The .hic names chromosomes 1, 2, 3 while every other input says chr1. Left alone that
    mismatch produces empty results rather than an error.
    """
    import cooler
    from hic2cool import hic2cool_convert

    mcool.parent.mkdir(parents=True, exist_ok=True)
    if not mcool.is_file():
        print(f"[prepare] converting {hic.name} -> {mcool.name}", flush=True)
        hic2cool_convert(str(hic), str(mcool), 0)
    for res in cooler.fileops.list_coolers(str(mcool)):
        clr = cooler.Cooler(f"{mcool}::{res}")
        names = list(clr.chromnames)
        mapping = {n: f"chr{n}" for n in names if not n.startswith("chr") and n != "ALL"}
        if mapping:
            cooler.rename_chroms(clr, mapping)
    clr = cooler.Cooler(f"{mcool}::{cooler.fileops.list_coolers(str(mcool))[0]}")
    print(f"[prepare] {mcool.name} chroms now {list(clr.chromnames)[:4]}...")


def write_atomic(dest: Path, lines: list[str]) -> None:
    """Write through a temporary file so an interrupted run never leaves a partial output.

    A half written anchors file would still parse, and the run using it would simply model less
    of the chromosome without reporting anything.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    with tmp.open("w") as fh:
        fh.writelines(lines)
    tmp.replace(dest)


def prepare(
    sample: trio_samples.Sample, raw_root: Path, data_root: Path, skip_hic: bool, force: bool
) -> None:
    raw = raw_root / sample.name
    out = data_root / sample.name
    out.mkdir(parents=True, exist_ok=True)
    name = sample.name
    clusters = out / f"{name}_clusters_3+.bedpe"
    oriented = out / f"{name}_anchors_3+_oriented.bed"

    if not force and clusters.is_file() and oriented.is_file() and clusters.stat().st_size:
        n_anchors = sum(1 for _ in oriented.open())
        print(f"[prepare:{name}] have clusters and {n_anchors} oriented anchors, skipping")
    else:
        # trio_downsample.py writes a matched file for any sample whose subsampling missed its
        # family's target. Preferring it here keeps that step out of the fetched data.
        matched = raw / f"{name}_hq.matched.BE3"
        source = matched if matched.is_file() else raw / f"{name}_hq.BE3"
        if matched.is_file():
            print(f"[prepare:{name}] using the depth matched loop set {matched.name}")
        loops = read_loops(source)
        pets = [x[6] for x in loops]
        print(f"[prepare:{name}] {len(loops)} loops, PET {min(pets)} to {max(pets)}")
        write_atomic(
            clusters,
            [f"{c1}\t{s1}\t{e1}\t{c2}\t{s2}\t{e2}\t{pet}\n" for c1, s1, e1, c2, s2, e2, pet in loops],
        )

        anchors = anchor_union(loops)
        motifs = trio_orient.load_motifs(Path(trio_orient.MOTIFS))
        counts = {"L": 0, "R": 0, "N": 0}
        rows: list[str] = []
        for c, s, e in anchors:
            o = trio_orient.orient(c, s, e, motifs)
            counts[o] += 1
            rows.append(f"{c}\t{s}\t{e}\t{o}\n")
        write_atomic(oriented, rows)
        total = len(anchors)
        print(
            f"[prepare:{name}] {total} anchors  "
            f"L={counts['L']} R={counts['R']} N={counts['N']} ({100 * counts['N'] / total:.1f}% N)"
        )
        peaks = raw / f"{name}_peaks.broadPeak"
        if peaks.is_file():
            print(
                f"[prepare:{name}] anchors on a called peak: "
                f"{100 * peak_overlap(anchors, peaks):.1f}%"
            )

    if not (out / CENTROMERES.name).is_file():
        shutil.copy(CENTROMERES, out / CENTROMERES.name)

    if not skip_hic:
        convert_hic(raw / f"{name}_allres.hic", data_root / "_hic" / name / f"{name}.mcool")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/_trio")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--samples")
    ap.add_argument("--skip-hic", action="store_true", help="text inputs only, no mcool")
    ap.add_argument("--force", action="store_true", help="rebuild outputs that already exist")
    args = ap.parse_args()
    for s in trio_samples.resolve(args.samples):
        prepare(s, Path(args.raw), Path(args.data_root), args.skip_hic, args.force)


if __name__ == "__main__":
    main()
