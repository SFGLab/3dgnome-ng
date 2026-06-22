#!/usr/bin/env python3
"""Call TAD boundaries from a Hi-C .mcool (cooltools insulation) → a 3dgnome breakpoints BED.

The 3dgnome segment-split (and our region selection) is CTCF/CCD-derived, whose domain boundaries
**mismatch the TAD boundaries of the Hi-C we score against** — so IBs cut through Hi-C TADs. For
Hi-C-driven validation (the self-HiC study), we instead segment by boundaries called from the Hi-C
itself (cooltools insulation), so the model's IBs align with the TADs we correlate against.

Output matches the CTCF breakpoints file format (``chrom<TAB>pos<TAB>pos``), so it's a drop-in for
both ``gnome3d.io.load_breakpoints`` (model segmentation, via ``data_segment_split``) and
``validation.sweep.enumerate_regions`` (region selection).

    python -m validation.hic_boundaries --hic data/_hic/GM12878/hic.4DNFIQ32RWCQ.mcool \\
        --chroms chr1 --out data/_hic/GM12878/hic_tad_breakpoints.bed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def call_boundaries(
    mcool_path: str,
    chroms: list[str] | None = None,
    window: int = 250_000,
    binsize: int = 25_000,
    min_strength: float = 0.0,
) -> list[tuple[str, int]]:
    """Sorted ``[(chrom, pos)]`` TAD boundaries from the mcool via cooltools insulation. ``window``
    (the insulation diamond, bp) is snapped to a multiple of the available resolution. ``pos`` is
    the boundary bin's midpoint. ``min_strength`` filters weak boundaries (0 = keep all called)."""
    import cooler
    import cooltools
    import pandas as pd

    avail = sorted(
        int(p.rsplit("/", 1)[-1])
        for p in cooler.fileops.list_coolers(mcool_path)
        if p.rsplit("/", 1)[-1].isdigit()
    )
    res = (binsize if binsize in avail else min(avail, key=lambda r: abs(r - binsize))) if avail else binsize
    uri = f"{mcool_path}::/resolutions/{res}" if avail else mcool_path
    clr = cooler.Cooler(uri)
    if window % res != 0:  # cooltools requires window to be a multiple of the resolution
        window = max(res * 4, round(window / res) * res)

    view_df = None
    if chroms:
        sizes = clr.chromsizes
        rows = [(c, 0, int(sizes[c]), c) for c in chroms if c in sizes.index]
        view_df = pd.DataFrame(rows, columns=["chrom", "start", "end", "name"])
    ins = cooltools.insulation(clr, [window], view_df=view_df, verbose=False)
    bcol, scol = f"is_boundary_{window}", f"boundary_strength_{window}"
    sel = (ins[bcol] == True) & (ins[scol].fillna(0.0) >= min_strength)  # noqa: E712
    b = ins[sel]
    out = [(str(r.chrom), int((r.start + r.end) // 2)) for r in b.itertuples()]
    return sorted(out)


def write_breakpoints(
    mcool_path: str,
    out_path: str | Path,
    chroms: list[str] | None = None,
    window: int = 250_000,
    binsize: int = 25_000,
    min_strength: float = 0.0,
) -> Path:
    """Write Hi-C TAD boundaries as a 3dgnome breakpoints BED (``chrom<TAB>pos<TAB>pos``)."""
    bnds = call_boundaries(mcool_path, chroms, window, binsize, min_strength)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for chrom, pos in bnds:
            f.write(f"{chrom}\t{pos}\t{pos}\n")
    print(f"[hic_boundaries] {len(bnds)} TAD boundaries (window {window}, res {binsize}) -> {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--hic", required=True, help="Hi-C .mcool")
    p.add_argument("--out", required=True, help="output breakpoints BED")
    p.add_argument("--chroms", default=None, help="comma-separated; default all in the mcool")
    p.add_argument("--window", type=int, default=250_000, help="insulation diamond window (bp)")
    p.add_argument("--binsize", type=int, default=25_000)
    p.add_argument("--min-strength", type=float, default=0.0, help="min boundary strength to keep")
    args = p.parse_args()
    chroms = args.chroms.split(",") if args.chroms else None
    write_breakpoints(args.hic, args.out, chroms, args.window, args.binsize, args.min_strength)


if __name__ == "__main__":
    main()
