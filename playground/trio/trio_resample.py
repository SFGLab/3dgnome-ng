"""Draw every sample of a family down to the family's sequencing depth from the full
libraries, so a trio comparison is not a depth comparison and depends on no one else's draw.

The providers' `downsampling/` folder was meant to do this and did not: for RNAPOL2 it cut
HG00514 to a tenth of its parents' loop count although its library is the deepest of the
three. This script starts from the merged libraries instead, the PET 1 and up cluster files
under `Loops/<pop>/<factor>/<sample>/merged/`, which are the rawest form the folder holds.

Per family, in order. The family's peak union is the three members' merged peak files merged
together. A cluster is kept when it spans at most `--max-span` and its anchors sit on that
union, both of them for CTCF and at least one for RNAPOL2, which is the rule the providers'
`wyniki_*` sets obey: every CTCF loop there has both ends on the family's CTCF peaks, every
RNAPOL2 loop has at least one end on the family's RNAPOL2 peaks and only 60 to 75 percent
both, and no loop of either spans over 1 Mb. Depth is the library's intra chromosomal PET total
over every cluster, peak or not. Each kept cluster's PET count is thinned by a binomial draw
at the family minimum over its own depth, and clusters left with three or more PETs are the
loop set. Thinning counts is the cluster level image of drawing reads, exact up to the
clusters a shallower run would have split, and a cluster under three PETs can only fall, so
only the three and up clusters are held in memory while the rest are streamed for the total.

    python playground/trio/trio_resample.py --factor RNAPOL2 --dry-run
    python playground/trio/trio_resample.py --factor RNAPOL2 --validate
    python playground/trio/trio_resample.py --factor CTCF

Output is `data/_trio/<S>/<S>[_<factor>]_hq.resampled.BE3`, which trio_prepare.py prefers
over the providers' set and over trio_downsample.py's draw. `--validate` reports, for each
sample, what fraction of the providers' filtered loops appear in the peak filtered merged set
at three or more PETs, which is what the peak rule has to reproduce.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import trio_samples  # noqa: E402

SEED = 20260920
MAX_SPAN = 1_000_000
# How many of a loop's two ends must sit on the family's peaks.
ENDS_ON_PEAKS = {"CTCF": 2, "RNAPOL2": 1}
Interval = tuple[int, int]
Cluster = tuple[str, int, int, str, int, int, int]


def _coord(text: str) -> int:
    """A BED coordinate, tolerating the scientific notation a few peak lines carry."""
    try:
        return int(text)
    except ValueError:
        return int(float(text))


def peak_union(paths: list[Path]) -> dict[str, list[Interval]]:
    """Merged intervals per chromosome over every peak file."""
    by: dict[str, list[Interval]] = defaultdict(list)
    for path in paths:
        for line in path.open():
            f = line.split()
            if len(f) >= 3:
                by[f[0]].append((_coord(f[1]), _coord(f[2])))
    out: dict[str, list[Interval]] = {}
    for c, lst in by.items():
        lst.sort()
        merged: list[list[int]] = []
        for s, e in lst:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        out[c] = [(s, e) for s, e in merged]
    return out


class PeakIndex:
    """Overlap queries on merged, non overlapping intervals per chromosome."""

    def __init__(self, union: dict[str, list[Interval]]) -> None:
        self.starts = {c: [s for s, _ in lst] for c, lst in union.items()}
        self.ends = {c: [e for _, e in lst] for c, lst in union.items()}

    def overlaps(self, c: str, s: int, e: int) -> bool:
        starts = self.starts.get(c)
        if not starts:
            return False
        k = bisect.bisect_right(starts, e - 1) - 1
        return k >= 0 and self.ends[c][k] > s


def scan(
    path: Path, peaks: PeakIndex, ends_needed: int, max_span: int
) -> tuple[int, int, list[Cluster]]:
    """Stream a cluster file. Returns the intra chromosomal PET total, the count of clusters
    at three or more PETs before any filter, and the passing clusters at three or more."""
    depth = 0
    n3 = 0
    kept: list[Cluster] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        for line in fh:
            f = line.split()
            if len(f) < 7 or f[0] != f[3]:
                continue
            n = int(f[6])
            depth += n
            if n < 3:
                continue
            n3 += 1
            s1, e1, s2, e2 = int(f[1]), int(f[2]), int(f[4]), int(f[5])
            if s2 - e1 > max_span:
                continue
            on = int(peaks.overlaps(f[0], s1, e1)) + int(peaks.overlaps(f[0], s2, e2))
            if on >= ends_needed:
                kept.append((f[0], s1, e1, f[3], s2, e2, n))
    return depth, n3, kept


def thin(kept: list[Cluster], p: float, rng: np.random.Generator) -> list[Cluster]:
    """Every cluster's PET count drawn from Binomial(n, p), those still at three or more kept."""
    if p >= 1.0:
        return list(kept)
    counts = np.array([c[6] for c in kept], dtype=np.int64)
    drawn = rng.binomial(counts, p)
    return [(*c[:6], int(d)) for c, d in zip(kept, drawn, strict=True) if d >= 3]


def read_loops(path: Path) -> list[Cluster]:
    out: list[Cluster] = []
    for line in path.open():
        f = line.split()
        if len(f) >= 7:
            out.append((f[0], int(f[1]), int(f[2]), f[3], int(f[4]), int(f[5]), int(f[6])))
    return out


def fraction_recovered(reference: list[Cluster], candidate: list[Cluster]) -> float:
    """Fraction of `reference` loops whose two anchors each overlap an anchor of one candidate
    loop. The providers re clustered subsampled reads, so coordinates differ by a few bases
    and exact matching would undercount."""
    by_chr: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    for c, s1, e1, _, s2, e2, _ in candidate:
        by_chr[c].append((s1, e1, s2, e2))
    for lst in by_chr.values():
        lst.sort()
    starts = {c: [x[0] for x in lst] for c, lst in by_chr.items()}
    hit = 0
    for c, s1, e1, _, s2, e2, _ in reference:
        lst = by_chr.get(c, [])
        k = bisect.bisect_right(starts.get(c, []), e1) - 1
        found = False
        while k >= 0 and lst[k][1] > s1 - 50_000:
            a1, b1, a2, b2 = lst[k]
            if a1 < e1 and b1 > s1 and a2 < e2 and b2 > s2:
                found = True
                break
            k -= 1
        hit += found
    return hit / max(len(reference), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/_trio")
    ap.add_argument("--factor", choices=("CTCF", "RNAPOL2"), default="CTCF")
    ap.add_argument("--families", help="comma separated among CHS, PUR, YRI, default all")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-span", type=int, default=MAX_SPAN)
    ap.add_argument("--validate", action="store_true", help="also score the peak rule")
    ap.add_argument("--dry-run", action="store_true", help="scan and report, write nothing")
    args = ap.parse_args()

    raw = Path(args.raw)
    tag = "" if args.factor == "CTCF" else f"_{args.factor.lower()}"
    fams = [f.strip() for f in args.families.split(",")] if args.families else ["CHS", "PUR", "YRI"]
    for fam_name in fams:
        fam = trio_samples.family(fam_name)
        peak_files = [raw / s.name / f"{s.name}{tag}_peaks.broadPeak" for s in fam]
        for f in peak_files:
            if not f.is_file():
                raise SystemExit(f"[resample:{fam_name}] missing {f}, fetch --factor {args.factor} first")
        union = peak_union(peak_files)
        n_peaks = sum(len(v) for v in union.values())
        print(f"[resample:{fam_name}] peak union {n_peaks} intervals over {len(fam)} members", flush=True)
        peaks = PeakIndex(union)

        scans: dict[str, tuple[int, int, list[Cluster]]] = {}
        for s in fam:
            src = raw / s.name / f"{s.name}{tag}_merged_loops.cis.gz"
            if not src.is_file():
                raise SystemExit(f"[resample:{s.name}] missing {src}, fetch --arm merged --include-gz")
            depth, n3, kept = scan(src, peaks, ENDS_ON_PEAKS[args.factor], args.max_span)
            scans[s.name] = (depth, n3, kept)
            print(
                f"[resample:{s.name}] {depth / 1e6:.2f} M intra PETs, {n3} clusters at 3+, "
                f"{len(kept)} pass the peak and span rule ({100 * len(kept) / max(n3, 1):.1f}%)",
                flush=True,
            )
            if args.validate:
                ref = raw / s.name / f"{s.name}{tag}_hq.BE3"
                if ref.is_file():
                    provided = read_loops(ref)
                    frac = fraction_recovered(provided, kept)
                    print(
                        f"[resample:{s.name}]   providers' set {len(provided)} loops, "
                        f"{100 * frac:.1f}% found in the filtered merged set"
                    )

        target = min(d for d, _, _ in scans.values())
        print(f"[resample:{fam_name}] depth target {target / 1e6:.2f} M PETs")
        for i, s in enumerate(fam):
            depth, _, kept = scans[s.name]
            p = target / depth
            rng = np.random.default_rng([args.seed, i, len(s.name)])
            loops = thin(kept, p, rng)
            pets = sum(c[6] for c in loops)
            print(
                f"[resample:{s.name}] keep {100 * p:.1f}% of PETs -> {len(loops)} loops of 3+, "
                f"{pets / 1e3:.0f} k PETs{'  dry run' if args.dry_run else ''}"
            )
            if args.dry_run:
                continue
            dest = raw / s.name / f"{s.name}{tag}_hq.resampled.BE3"
            tmp = dest.with_suffix(".part")
            loops.sort()
            with tmp.open("w") as fh:
                for c1, s1, e1, c2, s2, e2, n in loops:
                    fh.write(f"{c1}\t{s1}\t{e1}\t{c2}\t{s2}\t{e2}\t{n}\n")
            tmp.replace(dest)


if __name__ == "__main__":
    main()
