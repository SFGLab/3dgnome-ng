#!/usr/bin/env python3
"""
harness/find_region.py  -  find good test regions for 3dgnome integration tests.

A "good" region satisfies three constraints:
  1. All consecutive non-empty anchor pairs are non-overlapping (gap >= 0),
     which avoids NaN in the C++ smooth MC (genomicLengthToDistance(negative)).
  2. The chromosome has entries in the predefined segment-split breakpoints file,
     so the C++ findSplit() does not error out.
  3. The region has at least --min-anchors non-empty anchors with arcs.

Usage:
    python harness/find_region.py
    python harness/find_region.py --min-anchors 10 --max-span 3
    python harness/find_region.py --config data/GM12878/config.ini --data data/GM12878
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.settings import Settings
from src.io import load_anchors, load_arcs, mark_arcs, remove_empty_anchors, parse_region, load_breakpoints


def find_regions(
    config_path: str,
    data_dir: str,
    min_anchors: int = 8,
    max_span_mb: float = 5.0,
    chromosomes: list = None,
) -> list:
    """
    Scan chromosomes and return candidate regions sorted by quality.

    Returns list of dicts with keys:
        region, chrom, start, end, n_anchors, n_arcs, min_gap, span_mb
    """
    s = Settings()
    s.load_ini(config_path)
    s.data_dir = data_dir

    anchor_path = s.data_path(s.data_anchors)
    arc_path    = s.data_path(s.data_pet_clusters)
    bp_path     = s.data_path(s.data_segment_split)

    # Chromosomes that have predefined segment splits (C++ requirement)
    if bp_path:
        bp_chrs = set()
        try:
            with open(bp_path) as f:
                for line in f:
                    parts = line.split()
                    if parts:
                        bp_chrs.add(parts[0])
        except FileNotFoundError:
            bp_chrs = set()
    else:
        bp_chrs = set()

    if chromosomes is None:
        chromosomes = [f"chr{i}" for i in list(range(1, 23)) + ["X", "Y"]]

    results = []

    for chrom in chromosomes:
        if bp_chrs and chrom not in bp_chrs:
            continue  # C++ will error - no segment splits for this chr

        # Load all non-empty anchors for this chromosome (no region filter)
        all_a = load_anchors(anchor_path, {chrom})
        raw   = load_arcs(arc_path, {chrom}, max_pet_length=s.max_pet_length)
        mk    = mark_arcs(all_a, raw)
        cl    = remove_empty_anchors(all_a, mk)
        anch  = cl.get(chrom, [])
        if len(anch) < min_anchors:
            continue

        max_span_bp = int(max_span_mb * 1_000_000)

        # Sliding window: longest run of consecutive non-overlapping anchors within span
        best = []
        for i in range(len(anch)):
            run = [anch[i]]
            for j in range(i + 1, len(anch)):
                gap  = anch[j].start - run[-1].end
                span = anch[j].end   - run[0].start
                if gap >= 0 and span <= max_span_bp:
                    run.append(anch[j])
                elif gap < 0:
                    break   # overlap - stop extending
                else:
                    break   # span exceeded
            if len(run) > len(best):
                best = run[:]
            if len(best) >= len(anch):
                break  # can't do better

        if len(best) < min_anchors:
            continue

        region_str = f"{chrom}:{best[0].start}-{best[-1].end}"
        bed = parse_region(region_str)

        # Re-check with region filter to get accurate arc count
        all_a2 = load_anchors(anchor_path, {chrom}, bed)
        raw2   = load_arcs(arc_path, {chrom}, bed, max_pet_length=s.max_pet_length)
        mk2    = mark_arcs(all_a2, raw2)
        cl2    = remove_empty_anchors(all_a2, mk2)
        anchors2 = cl2.get(chrom, [])
        arcs2    = mk2.get(chrom, [])

        if len(anchors2) < min_anchors:
            continue

        gaps = [anchors2[k + 1].start - anchors2[k].end for k in range(len(anchors2) - 1)]
        if any(g < 0 for g in gaps):
            continue  # overlap crept in after region re-filter

        span_mb = (best[-1].end - best[0].start) / 1e6
        min_gap = min(gaps) if gaps else 0

        results.append({
            "region":    region_str,
            "chrom":     chrom,
            "start":     best[0].start,
            "end":       best[-1].end,
            "n_anchors": len(anchors2),
            "n_arcs":    len(arcs2),
            "min_gap":   min_gap,
            "span_mb":   span_mb,
        })

    # Sort: most anchors first, then fewest arcs-per-anchor (well-connected),
    # then smallest span (fastest to run)
    results.sort(key=lambda r: (-r["n_anchors"], r["span_mb"]))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "data/GM12878/config.ini"),
                    help="Path to config.ini (default: data/GM12878/config.ini)")
    ap.add_argument("--data", default=str(ROOT / "data/GM12878"),
                    help="Data directory override (default: data/GM12878)")
    ap.add_argument("--min-anchors", type=int, default=8,
                    help="Minimum non-empty anchors in region (default: 8)")
    ap.add_argument("--max-span", type=float, default=5.0,
                    help="Maximum region span in Mb (default: 5.0)")
    ap.add_argument("--chrom", nargs="+",
                    help="Restrict to specific chromosomes (default: all)")
    ap.add_argument("--top", type=int, default=15,
                    help="Show top N results (default: 15)")
    args = ap.parse_args()

    print(f"Scanning for clean regions (min_anchors={args.min_anchors}, max_span={args.max_span} Mb) ...\n",
          flush=True)

    results = find_regions(
        config_path=args.config,
        data_dir=args.data,
        min_anchors=args.min_anchors,
        max_span_mb=args.max_span,
        chromosomes=args.chrom,
    )

    if not results:
        print("No regions found matching criteria.")
        return

    hdr = f"{'Region':<32}  {'anchors':>7}  {'arcs':>5}  {'min_gap':>10}  {'span':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in results[: args.top]:
        print(
            f"{r['region']:<32}  {r['n_anchors']:>7}  {r['n_arcs']:>5}"
            f"  {r['min_gap']:>10,}  {r['span_mb']:>7.2f} Mb"
        )

    print(f"\nBest region: {results[0]['region']}")
    print(f"  {results[0]['n_anchors']} anchors, {results[0]['n_arcs']} arcs, "
          f"min gap {results[0]['min_gap']:,} bp, {results[0]['span_mb']:.2f} Mb")


if __name__ == "__main__":
    main()
