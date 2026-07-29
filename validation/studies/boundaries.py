"""Call TAD boundaries from a Hi-C .mcool via cooltools insulation into a 3dgnome breakpoints BED.

The 3dgnome segment-split, and our region selection, is CTCF/CCD-derived, whose domain boundaries
mismatch the TAD boundaries of the Hi-C we score against, so IBs cut through Hi-C TADs. For
Hi-C-driven validation, the self-HiC study, we instead segment by boundaries called from the Hi-C
itself with cooltools insulation, so the model's IBs align with the TADs we correlate against.

Output matches the CTCF breakpoints file format, chrom<TAB>pos<TAB>pos, so it is a drop-in for both
gnome3d.io.load_breakpoints, the model segmentation via data_segment_split, and
validation.core.regions.enumerate_regions, the region selection.

    python -m validation boundaries --hic data/_hic/GM12878/hic.4DNFIQ32RWCQ.mcool \\
        --chroms chr1 --out data/_hic/GM12878/hic_tad_breakpoints.bed
"""

from __future__ import annotations

import argparse
import sys

from validation.core.boundaries import call_boundaries, write_breakpoints  # noqa: F401
from validation.studies import Context, Study, register


class Boundaries(Study):
    name = "boundaries"
    help = "call TAD boundaries from a Hi-C .mcool via cooltools insulation"

    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--out", required=True, help="output breakpoints BED")
        p.add_argument("--chroms", default=None, help="comma-separated; default all in the mcool")
        p.add_argument("--window", type=int, default=250_000, help="insulation diamond window (bp)")
        p.add_argument("--binsize", type=int, default=25_000)
        p.add_argument(
            "--min-strength", type=float, default=0.0, help="min boundary strength to keep"
        )

    def run(self, ctx: Context, args: argparse.Namespace) -> None:
        if ctx.hic is None:
            sys.exit("[error] boundaries requires --hic, an observed Hi-C .mcool")
        chroms = args.chroms.split(",") if args.chroms else None
        write_breakpoints(ctx.hic, args.out, chroms, args.window, args.binsize, args.min_strength)


register(Boundaries())
