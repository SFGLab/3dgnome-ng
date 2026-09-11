"""Spike. Derive the segment boundary file for a mouse condition from called TAD boundaries.

TAD boundaries come from cooltools insulation on the condition's own balanced map, at 25 kb
with a 250 kb window, and are thinned so that consecutive kept boundaries are at least 2 Mb
apart, the rule the trios use. Segments are the grouping above blocks and a segment holding
one block or fewer is skipped by block placement, so they must be much coarser than a TAD.
Blocks are not derived here. They come from arc gaps.

    python playground/mouse/mouse_segments.py --conditions E18_WT,P60_WT,P60_KO
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from validation.core.boundaries import call_boundaries  # noqa: E402

CHROMS = ["chr10", "chr19"]


def thin(positions: list[int], min_span: int) -> list[int]:
    kept: list[int] = []
    for p in sorted(positions):
        if not kept or p - kept[-1] >= min_span:
            kept.append(p)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conditions", default="E18_WT,P60_WT,P60_KO")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--min-segment", type=int, default=2_000_000)
    args = ap.parse_args()
    for name in args.conditions.split(","):
        mcool = Path(args.data_root) / "_hic" / name / f"{name}.mcool"
        bnds = call_boundaries(str(mcool), CHROMS, window=250_000, binsize=25_000)
        by: dict[str, list[int]] = {c: [] for c in CHROMS}
        for c, p in bnds:
            by[c].append(p)
        out = Path(args.data_root) / name / f"{name}_segments.bed"
        with out.open("w") as fh:
            for c in CHROMS:
                kept = thin(by[c], args.min_segment)
                for p in kept:
                    fh.write(f"{c}\t{p}\t{p}\n")
                print(f"[segments:{name}] {c}: {len(by[c])} TAD boundaries, {len(kept)} kept at {args.min_segment / 1e6:.0f} Mb")
        print(f"[segments:{name}] -> {out}")


if __name__ == "__main__":
    main()
