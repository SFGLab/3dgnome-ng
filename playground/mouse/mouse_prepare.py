"""Spike. Turn the lab's mouse brain Hi-C into the per condition layout the pipeline reads.

Three conditions on mm10, each a pooled Juicer map with HiCCUPS loops beside it: E18.5 wild
type, P60 wild type and P60 TCF7L2 knockout. There is no ChIA-PET, so the loops are the arcs
and the Hi-C map is the singletons, which is the arrangement the cell line runs use when a Hi-C
file is present.

Loops. A HiCCUPS row is the two anchor bins plus an observed count. The observed count is the
arc score, since the arc law fits a loop's strength against the typical count at its span on
the run's own arcs, so any consistent count works. Chromosomes are written with the chr prefix
the pipeline's region parser expects.

Anchors. The distinct anchor intervals of the loops, as on the cell lines and the trios. The
orientation column comes from a scan of the JASPAR MA0139.1 CTCF matrix over the mm10 sequence
of the chromosomes being modelled. An anchor takes the strand of its best scoring hit, and N
where nothing scores above the threshold or the chromosome was not scanned. The strand to
letter mapping is the one trio_orient recovered from the GM12878 anchors, plus is R and minus
is L.

The motif term treats N as R rather than as absent, so a chromosome that was not scanned must
not be modelled with the motif term on. The regions here are on chr10 and chr19 and both are
scanned.

    python playground/mouse/mouse_prepare.py --src <dir with loops, fasta, pfm> --out data
    python playground/mouse/mouse_prepare.py --src <dir> --validate
"""

from __future__ import annotations

import argparse
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np

CONDITIONS = {
    "E18_WT": "e18_pulled_merged_loops.bedpe",
    "P60_WT": "adult_merged_loops_hiccups.bedpe",
    "P60_KO": "ko_5kb10kb_kr_loops.bedpe",
}
SCAN_CHROMS = ("chr10", "chr19")
STRAND_TO_ORIENTATION = {"+": "R", "-": "L"}
# JASPAR's genome tracks keep hits at p < 1e-4, which on the hg38 MA0139.1 track is 264 hits
# per Mb of chr1. The relative score with that hit density on mm10 chr10 is 0.835, measured by
# --validate, so the scan here keeps the same hits the track would.
MIN_RELATIVE_SCORE = 0.835
PSEUDOCOUNT = 0.8

Loop = tuple[str, int, int, str, int, int, int]


def read_hiccups(path: Path) -> list[Loop]:
    """HiCCUPS rows as (chrA, sA, eA, chrB, sB, eB, count), intra chromosomal only."""
    out: list[Loop] = []
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split()
            if len(f) < 12:
                continue
            c1, c2 = _chr(f[0]), _chr(f[3])
            if c1 != c2:
                continue
            s1, e1, s2, e2 = int(f[1]), int(f[2]), int(f[4]), int(f[5])
            if s2 < s1:
                s1, e1, s2, e2 = s2, e2, s1, e1
            count = max(1, int(round(float(f[11]))))
            out.append((c1, s1, e1, c2, s2, e2, count))
    out.sort()
    return out


def _chr(name: str) -> str:
    return name if name.startswith("chr") else "chr" + name


def anchor_union(loops: list[Loop]) -> list[tuple[str, int, int]]:
    seen: set[tuple[str, int, int]] = set()
    for c1, s1, e1, c2, s2, e2, _ in loops:
        seen.add((c1, s1, e1))
        seen.add((c2, s2, e2))
    return sorted(seen)


# --- CTCF motif scan -----------------------------------------------------------------------


def read_pfm(path: Path) -> np.ndarray:
    """JASPAR pfm as a (4, w) count matrix in ACGT order."""
    rows = [
        [float(x) for x in line.split()]
        for line in path.read_text().splitlines()
        if line and not line.startswith(">")
    ]
    m = np.array(rows, dtype=np.float64)
    assert m.shape[0] == 4, m.shape
    return m


def log_odds(pfm: np.ndarray) -> np.ndarray:
    """Log odds against a uniform background, JASPAR style pseudocount split by base."""
    p = (pfm + PSEUDOCOUNT / 4.0) / (pfm.sum(axis=0, keepdims=True) + PSEUDOCOUNT)
    return np.log2(p / 0.25)


def read_fasta_gz(path: Path) -> np.ndarray:
    """One chromosome as int8 codes, A C G T as 0 to 3 and anything else as 4."""
    parts: list[str] = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if not line.startswith(">"):
                parts.append(line.strip().upper())
    seq = "".join(parts).encode()
    table = np.full(256, 4, dtype=np.int8)
    for i, b in enumerate(b"ACGT"):
        table[b] = i
    return table[np.frombuffer(seq, dtype=np.uint8)]


def scan(codes: np.ndarray, lo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best strand score at every start position and which strand gave it, as relative scores.

    A window holding an unknown base scores minus infinity.
    """
    w = lo.shape[1]
    n = codes.shape[0] - w + 1
    lo5 = np.vstack([lo, np.full((1, w), -np.inf)])  # row 4 is the unknown base
    fwd = np.zeros(n)
    rev = np.zeros(n)
    rc = lo[::-1, ::-1]  # reverse complement: complement flips ACGT to TGCA, then reverse
    rc5 = np.vstack([rc, np.full((1, w), -np.inf)])
    for k in range(w):
        col = codes[k : k + n]
        fwd += lo5[col, k]
        rev += rc5[col, k]
    best = np.maximum(fwd, rev)
    strand = np.where(fwd >= rev, 1, -1).astype(np.int8)
    smax = lo.max(axis=0).sum()
    smin = lo.min(axis=0).sum()
    rel = (best - smin) / (smax - smin)
    return rel, strand


def orient_anchors(
    anchors: list[tuple[str, int, int]],
    rel_by_chr: dict[str, np.ndarray],
    strand_by_chr: dict[str, np.ndarray],
    width: int,
) -> list[str]:
    out: list[str] = []
    for c, s, e in anchors:
        rel = rel_by_chr.get(c)
        if rel is None:
            out.append("N")
            continue
        lo_i = max(0, s - width + 1)
        hi_i = min(rel.shape[0], e)
        if hi_i <= lo_i:
            out.append("N")
            continue
        win = rel[lo_i:hi_i]
        k = int(np.argmax(win))
        if win[k] < MIN_RELATIVE_SCORE:
            out.append("N")
            continue
        out.append(STRAND_TO_ORIENTATION["+" if strand_by_chr[c][lo_i + k] > 0 else "-"])
    return out


def scan_chromosomes(src: Path, pfm: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    lo = log_odds(pfm)
    rel_by: dict[str, np.ndarray] = {}
    strand_by: dict[str, np.ndarray] = {}
    for c in SCAN_CHROMS:
        fa = src / f"mm10_{c}.fa.gz"
        if not fa.is_file():
            print(f"[prepare] no sequence for {c}, its anchors get N")
            continue
        codes = read_fasta_gz(fa)
        rel, strand = scan(codes, lo)
        rel_by[c] = rel
        strand_by[c] = strand
        hits = int((rel >= MIN_RELATIVE_SCORE).sum())
        print(f"[prepare] {c}: {codes.shape[0] / 1e6:.1f} Mb, {hits} hits, {hits / (codes.shape[0] / 1e6):.0f} per Mb")
    return rel_by, strand_by, lo.shape[1]


# --- validation ------------------------------------------------------------------------------


def validate(src: Path) -> None:
    """Two checks on the scanner. The CTCF consensus scores at the top, and the hit density at
    the threshold is of the order of the JASPAR hg38 track's, which is the same matrix at
    p < 1e-4."""
    pfm = read_pfm(src / "MA0139.1.pfm")
    lo = log_odds(pfm)
    consensus = "".join("ACGT"[i] for i in pfm.argmax(axis=0))
    codes = np.array(["ACGT".index(b) for b in consensus], dtype=np.int8)
    rel, _ = scan(np.concatenate([codes, codes]), lo)
    print(f"[validate] consensus {consensus} relative score {rel[0]:.3f} (expect 1.000)")
    rel_by, _, _ = scan_chromosomes(src, pfm)
    track = Path("playground/jaspar_MA0139.1_hg38.tsv.gz")
    if track.is_file():
        n = 0
        span = 0
        with gzip.open(track, "rt") as fh:
            for line in fh:
                f = line.split()
                if f and f[0] == "chr1":
                    n += 1
                    span = max(span, int(f[2]))
        print(f"[validate] JASPAR hg38 chr1 track: {n / (span / 1e6):.0f} hits per Mb")


# --- main ------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="directory holding the loops, fasta and pfm")
    ap.add_argument("--out", default="data")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    src = Path(args.src)
    if args.validate:
        validate(src)
        return
    pfm = read_pfm(src / "MA0139.1.pfm")
    rel_by, strand_by, width = scan_chromosomes(src, pfm)
    for cond, fname in CONDITIONS.items():
        loops = read_hiccups(src / fname)
        anchors = anchor_union(loops)
        orn = orient_anchors(anchors, rel_by, strand_by, width)
        d = Path(args.out) / cond
        d.mkdir(parents=True, exist_ok=True)
        with (d / f"{cond}_clusters_3+.bedpe").open("w") as fh:
            for c1, s1, e1, c2, s2, e2, n in loops:
                fh.write(f"{c1}\t{s1}\t{e1}\t{c2}\t{s2}\t{e2}\t{n}\n")
        with (d / f"{cond}_anchors_3+_oriented.bed").open("w") as fh:
            for (c, s, e), o in zip(anchors, orn, strict=True):
                fh.write(f"{c}\t{s}\t{e}\t{o}\n")
        by: dict[str, list[str]] = defaultdict(list)
        for (c, _, _), o in zip(anchors, orn, strict=True):
            by[c].append(o)
        summary = ", ".join(
            f"{c} L{v.count('L')}/R{v.count('R')}/N{v.count('N')}" for c, v in sorted(by.items()) if c in SCAN_CHROMS
        )
        print(f"[prepare] {cond}: {len(loops)} loops, {len(anchors)} anchors; {summary}")


if __name__ == "__main__":
    main()
