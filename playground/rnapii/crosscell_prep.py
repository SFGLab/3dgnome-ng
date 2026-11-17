"""The cross line RNAPII inputs of `design/2026-11-11-rnapii-release.md`, Part 1.

Loops of GM12878, H1ESC and HFFc6 are pooled over replicates, filtered on the line's own peaks
and a span cap, and thinned binomially so every line of one factor carries the same PET total.
The rule is the trio rule of `triosformer/pipeline/trio_resample.py`. A CTCF loop needs both
ends on a CTCF peak and an RNAPII loop one end on an RNAPII peak. The span is the gap between
the two ends. For the merged anchor arm an RNAPII loop end within `absorb_bp` of a CTCF anchor
takes that anchor's interval, so the loader maps it onto the anchor and no new bead is made.

A loop is `(chrA, startA, endA, chrB, startB, endB, pets)`, intra chromosomal, end A first.
"""

from __future__ import annotations

import bisect
import gzip
from pathlib import Path

import numpy as np

Loop = tuple[str, int, int, str, int, int, int]
Interval = tuple[int, int]
AnchorRow = tuple[str, int, int, str]

# No CTCF anchor is wider than this, which bounds the scan back from a query position.
MAX_ANCHOR_BP = 50_000


def read_bedpe(path: Path) -> list[Loop]:
    """Intra chromosomal loops of a seven column bedpe, gzip aware, end A first."""
    opener = gzip.open if str(path).endswith(".gz") else open
    out: list[Loop] = []
    with opener(path, "rt") as fh:
        for line in fh:
            f = line.split()
            if len(f) < 7 or f[0] != f[3]:
                continue
            a = (int(f[1]), int(f[2]))
            b = (int(f[4]), int(f[5]))
            if b < a:
                a, b = b, a
            out.append((f[0], a[0], a[1], f[0], b[0], b[1], int(float(f[6]))))
    return out


def write_bedpe(path: Path, loops: list[Loop]) -> None:
    """The loops sorted, one tab separated row each."""
    with open(path, "w") as out:
        for c, s1, e1, c2, s2, e2, p in sorted(loops):
            out.write(f"{c}\t{s1}\t{e1}\t{c2}\t{s2}\t{e2}\t{p}\n")


def pool_replicates(reps: list[list[Loop]]) -> list[Loop]:
    """One loop set from several. Loops whose two ends both overlap join into one loop with the
    union interval at each end and the PET counts summed."""
    # Rows are [startA, endA, startB, endB, pets]. Input is sorted by end A, so a pooled loop
    # whose end A finishes before the current start can never be joined again.
    pooled: list[tuple[str, list[int]]] = []
    active: dict[str, list[list[int]]] = {}
    for c, s1, e1, _, s2, e2, p in sorted(lp for rep in reps for lp in rep):
        live = [q for q in active.get(c, []) if q[1] > s1]
        hit = next((q for q in live if q[2] < e2 and q[3] > s2), None)
        if hit is None:
            row = [s1, e1, s2, e2, p]
            pooled.append((c, row))
            live.append(row)
        else:
            hit[1] = max(hit[1], e1)
            hit[2] = min(hit[2], s2)
            hit[3] = max(hit[3], e2)
            hit[4] += p
        active[c] = live
    return sorted((c, r[0], r[1], c, r[2], r[3], r[4]) for c, r in pooled)


class PeakIndex:
    """Overlap queries on merged, non overlapping intervals per chromosome."""

    def __init__(self, union: dict[str, list[Interval]]) -> None:
        self.starts = {c: [s for s, _ in lst] for c, lst in union.items()}
        self.ends = {c: [e for _, e in lst] for c, lst in union.items()}

    @classmethod
    def from_beds(cls, paths: list[Path], max_per_file: int | None = None) -> PeakIndex:
        """The merged union of every peak file, each capped by `top_peaks`."""
        return cls(top_peaks(paths, max_per_file))

    def overlaps(self, chrom: str, start: int, end: int) -> bool:
        starts = self.starts.get(chrom)
        if not starts:
            return False
        k = bisect.bisect_right(starts, end - 1) - 1
        return k >= 0 and self.ends[chrom][k] > start


def top_peaks(paths: list[Path], max_per_file: int | None) -> dict[str, list[Interval]]:
    """Merged intervals per chromosome over the peak files, gzip aware. With `max_per_file` each
    file keeps only its strongest peaks by the signal in column 7."""
    by: dict[str, list[Interval]] = {}
    for path in paths:
        opener = gzip.open if str(path).endswith(".gz") else open
        rows: list[tuple[float, str, int, int]] = []
        with opener(path, "rt") as fh:
            for line in fh:
                f = line.split()
                if len(f) >= 3 and not line.startswith(("#", "track")):
                    signal = float(f[6]) if len(f) > 6 else 0.0
                    rows.append((signal, f[0], int(float(f[1])), int(float(f[2]))))
        if max_per_file is not None and len(rows) > max_per_file:
            rows.sort(key=lambda r: -r[0])
            rows = rows[:max_per_file]
        for _, c, a, b in rows:
            by.setdefault(c, []).append((a, b))
    union: dict[str, list[Interval]] = {}
    for c, lst in by.items():
        lst.sort()
        merged: list[list[int]] = []
        for a, b in lst:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        union[c] = [(a, b) for a, b in merged]
    return union


def filter_loops(
    loops: list[Loop], peaks: PeakIndex, ends_on_peaks: int, max_span: int
) -> list[Loop]:
    """Loops whose gap is at most `max_span` and with at least `ends_on_peaks` ends on a peak."""
    out: list[Loop] = []
    for lp in loops:
        c, s1, e1, _, s2, e2, _ = lp
        if s2 - e1 > max_span:
            continue
        on = int(peaks.overlaps(c, s1, e1)) + int(peaks.overlaps(c, s2, e2))
        if on >= ends_on_peaks:
            out.append(lp)
    return out


def depth(loops: list[Loop]) -> int:
    """The PET total."""
    return sum(lp[6] for lp in loops)


def thin(loops: list[Loop], p: float, rng: np.random.Generator, min_pets: int = 3) -> list[Loop]:
    """Every loop's PET count drawn from Binomial(n, p), the loops still at `min_pets` kept."""
    if p >= 1.0:
        return list(loops)
    drawn = rng.binomial(np.array([lp[6] for lp in loops], dtype=np.int64), p)
    return [(*lp[:6], int(d)) for lp, d in zip(loops, drawn, strict=True) if d >= min_pets]


def thin_to_depth(
    loops: list[Loop], target: int, rng: np.random.Generator, min_pets: int = 3
) -> tuple[list[Loop], float]:
    """Loops thinned so their PET total after the `min_pets` floor is `target`, and the keep
    probability used.

    Every PET draws one uniform number, and at probability p a loop keeps the PETs whose number
    is under p. The total is then monotone in p, so p is found by bisection, and the floor is
    counted in the total rather than applied after it. A target at or above the depth keeps
    every loop as it is.
    """
    if target >= depth(loops):
        return list(loops), 1.0
    counts = np.array([lp[6] for lp in loops], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    u = rng.random(int(counts.sum()))

    def kept(p: float) -> np.ndarray:
        k = np.add.reduceat((u < p).astype(np.int64), starts)
        return np.where(k >= min_pets, k, 0)

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if kept(mid).sum() < target:
            lo = mid
        else:
            hi = mid
    k = kept(hi)
    return [(*lp[:6], int(n)) for lp, n in zip(loops, k, strict=True) if n > 0], hi


def _nearest(
    by_chr: dict[str, list[AnchorRow]], starts: dict[str, list[int]], c: str, s: int, e: int, reach: int
) -> AnchorRow | None:
    """The CTCF anchor with the smallest gap to [s, e) when that gap is at most `reach`."""
    rows = by_chr.get(c)
    if not rows:
        return None
    k = bisect.bisect_right(starts[c], e + reach) - 1
    best: AnchorRow | None = None
    best_gap = reach + 1
    while k >= 0 and rows[k][1] >= s - reach - MAX_ANCHOR_BP:
        a = rows[k]
        gap = max(0, a[1] - e, s - a[2])
        if gap < best_gap or (gap == best_gap and best is not None and a[1] < best[1]):
            best, best_gap = a, gap
        k -= 1
    return best


def absorb_ends(
    loops: list[Loop], ctcf: list[AnchorRow], absorb_bp: int
) -> tuple[list[Loop], int]:
    """Loops with every end within `absorb_bp` of a CTCF anchor moved onto that anchor.

    A loop whose two ends land on one anchor is dropped. Loops that land on the same pair of
    intervals join with their PET counts summed. Returns the loops and the number of ends moved.
    """
    by_chr: dict[str, list[AnchorRow]] = {}
    for row in ctcf:
        by_chr.setdefault(row[0], []).append(row)
    for lst in by_chr.values():
        lst.sort(key=lambda r: (r[1], r[2]))
    starts = {c: [r[1] for r in lst] for c, lst in by_chr.items()}
    moved = 0
    joined: dict[tuple[str, int, int, int, int], int] = {}
    for c, s1, e1, _, s2, e2, p in loops:
        ends = []
        for s, e in ((s1, e1), (s2, e2)):
            a = _nearest(by_chr, starts, c, s, e, absorb_bp)
            if a is not None and (a[1], a[2]) != (s, e):
                moved += 1
                ends.append((a[1], a[2]))
            else:
                ends.append((s, e))
        ends.sort()
        if ends[0] == ends[1]:
            continue
        key = (c, ends[0][0], ends[0][1], ends[1][0], ends[1][1])
        joined[key] = joined.get(key, 0) + p
    out = [(c, a, b, c, x, y, p) for (c, a, b, x, y), p in sorted(joined.items())]
    return out, moved


# The driver. Writes, per line, under data/<C>/
#   <C>_xc_ctcf_clusters_3+.bedpe, <C>_xc_rnapii_clusters_3+.bedpe,
#   <C>_xc_rnapii_absorbed_clusters_3+.bedpe, <C>_xc_anchors_ctcf_rnapii.bed,
#   <C>_xc_anchors_ctcf_rnapii_absorbed.bed and the record <C>_xc_prep.json.
#
#     python playground/rnapii/crosscell_prep.py [--max-span 1000000] [--seed 20261111] [--dry-run]
#
# Inputs are the production CTCF loops and anchors under data/<C>/ and the ENCODE files the
# manifests validation/manifests/<C>_rnapii.json fetch into data/_rnapii/<C>/.

LINES = ("GM12878", "H1ESC", "HFFC6")
FACTORS = ("ctcf", "rnapii")
ENDS_ON_PEAKS = {"ctcf": 2, "rnapii": 1}
KEEP_FLOOR = 0.2


def check_keep_fractions(depths: dict[str, int], floor: float) -> dict[str, float]:
    """Each line's keep fraction when thinned to the lowest depth. Exits naming the lines under
    `floor`, since thinning them all to one shallow line would leave the others nearly empty."""
    low = min(depths.values())
    frac = {k: low / v if v else 0.0 for k, v in depths.items()}
    under = [k for k, f in frac.items() if f < floor]
    if under:
        raise SystemExit(
            f"keep fraction under {floor} for {', '.join(under)}: depths {depths}. "
            "Thinning every line to the shallowest would leave these nearly empty; stopping."
        )
    return frac


def build_record(line: str, **fields: object) -> dict[str, object]:
    """The prep record of one line, the line first."""
    return {"line": line, **fields}


def _manifest(root: Path, line: str) -> dict[str, object]:
    import json

    return json.loads((root / "validation" / "manifests" / f"{line}_rnapii.json").read_text())


def _fetched(root: Path, line: str, prefix: str) -> list[Path]:
    paths = sorted((root / "data" / "_rnapii" / line).glob(f"{prefix}*.ENCFF*"))
    if not paths:
        raise SystemExit(f"{line}: no {prefix} file under data/_rnapii/{line}, run the fetch first")
    return paths


def main() -> None:
    import argparse
    import json
    import sys

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from anchor_union import read_anchors, union_anchors

    from validation.core.config import CANONICAL

    ap = argparse.ArgumentParser(description="cross line RNAPII inputs")
    ap.add_argument("--max-span", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=20261111)
    # 25 kb, the Hi-C pixel the battery reads; at the spec's 1 kb arm M barely differs from R
    ap.add_argument("--absorb-bp", type=int, default=25_000, help="0 takes target_bp_per_subanchor")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    absorb_bp = args.absorb_bp or int(CANONICAL["main"]["target_bp_per_subanchor"])  # type: ignore[call-overload]

    # Every RNAPII peak file is capped at the median peak count of the thresholded ENCODE sets of
    # GM12878 and H1ESC. BJ's files are unthresholded MACS2 calls with ten times as many peaks,
    # which would let three times as many RNAPII loops through the filter.
    native = [p for line in ("GM12878", "H1ESC") for p in _fetched(root, line, "RNAPII_peaks")]
    peak_cap = int(np.median([sum(1 for _ in gzip.open(p, "rt")) for p in native]))
    print(f"RNAPII peaks capped at {peak_cap} per file")
    filtered: dict[tuple[str, str], list[Loop]] = {}
    before: dict[tuple[str, str], int] = {}
    peak_counts: dict[str, dict[str, int]] = {}
    for line in LINES:
        data = root / "data" / line
        ctcf_loops = read_bedpe(data / f"{line}_clusters_3+.bedpe")
        rnapii_loops = pool_replicates([read_bedpe(p) for p in _fetched(root, line, "RNAPII_loops")])
        peaks = {
            "ctcf": PeakIndex.from_beds(_fetched(root, line, "CTCF_peaks")),
            "rnapii": PeakIndex.from_beds(_fetched(root, line, "RNAPII_peaks"), peak_cap),
        }
        peak_counts[line] = {
            p.name: sum(1 for _ in gzip.open(p, "rt"))
            for p in _fetched(root, line, "RNAPII_peaks") + _fetched(root, line, "CTCF_peaks")
        }
        for fac, loops in (("ctcf", ctcf_loops), ("rnapii", rnapii_loops)):
            before[(line, fac)] = len(loops)
            filtered[(line, fac)] = filter_loops(loops, peaks[fac], ENDS_ON_PEAKS[fac], args.max_span)

    fractions: dict[tuple[str, str], float] = {}
    targets: dict[str, int] = {}
    for fac in FACTORS:
        depths = {line: depth(filtered[(line, fac)]) for line in LINES}
        print(f"{fac} filtered depths: {depths}")
        targets[fac] = min(depths.values())
        for line, f in check_keep_fractions(depths, KEEP_FLOOR).items():
            fractions[(line, fac)] = f

    print(f"\n{'line':<8} {'factor':<7} {'loops in':>9} {'filtered':>9} {'depth':>10} {'keep':>6} {'thinned':>9} {'PETs':>10}")
    for li, line in enumerate(LINES):
        data = root / "data" / line
        man = _manifest(root, line)
        record = build_record(
            line,
            rnapii_source=str(man.get("_rnapii_source", man["_note"])),
            accessions=[a["accession"] for a in man["assays"]],  # type: ignore[index, union-attr]
            seed=args.seed,
            max_span=args.max_span,
            absorb_bp=absorb_bp,
            rnapii_peak_cap=peak_cap,
            peak_counts=peak_counts[line],
        )
        thinned: dict[str, list[Loop]] = {}
        for fi, fac in enumerate(FACTORS):
            loops = filtered[(line, fac)]
            rng = np.random.default_rng([args.seed, li, fi])
            thinned[fac], p_keep = thin_to_depth(loops, targets[fac], rng)
            record[fac] = {
                "loops_before": before[(line, fac)],
                "loops_filtered": len(loops),
                "depth_before": depth(loops),
                "keep_fraction": fractions[(line, fac)],
                "keep_probability": p_keep,
                "loops_after": len(thinned[fac]),
                "depth_after": depth(thinned[fac]),
            }
            print(
                f"{line:<8} {fac:<7} {before[(line, fac)]:>9} {len(loops):>9} {depth(loops):>10} "
                f"{fractions[(line, fac)]:>6.3f} {len(thinned[fac]):>9} {depth(thinned[fac]):>10}"
            )
        ctcf_anchors = read_anchors(data / f"{line}_anchors_3+_oriented.bed")
        absorbed, moved = absorb_ends(thinned["rnapii"], ctcf_anchors, absorb_bp)
        anchor_sets = {}
        for tag, loops in (("", thinned["rnapii"]), ("_absorbed", absorbed)):
            ends = [(lp[0], s, e) for lp in loops for s, e in ((lp[1], lp[2]), (lp[4], lp[5]))]
            rows, st = union_anchors(ctcf_anchors, ends)
            anchor_sets[tag] = rows
            record[f"anchors{tag or '_plain'}"] = {"added": st.added, "summary": st.describe()}
            print(f"  {line} RNAPII{tag or ''}: {st.describe()}")
        record["ends_absorbed"] = moved
        record["loops_absorbed_arm"] = len(absorbed)
        if args.dry_run:
            continue
        write_bedpe(data / f"{line}_xc_ctcf_clusters_3+.bedpe", thinned["ctcf"])
        write_bedpe(data / f"{line}_xc_rnapii_clusters_3+.bedpe", thinned["rnapii"])
        write_bedpe(data / f"{line}_xc_rnapii_absorbed_clusters_3+.bedpe", absorbed)
        for tag, rows in anchor_sets.items():
            with open(data / f"{line}_xc_anchors_ctcf_rnapii{tag}.bed", "w") as out:
                for c, s, e, o in rows:
                    out.write(f"{c}\t{s}\t{e}\t{o}\n")
        (data / f"{line}_xc_prep.json").write_text(json.dumps(record, indent=2) + "\n")
    print("\ndry run, nothing written" if args.dry_run else "\nwritten under data/<line>/")


if __name__ == "__main__":
    main()
