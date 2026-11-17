"""Catalog loading and region parsing.

    python web/tests/test_catalog.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from _check import check, done
from fixture import make_data

from gnome3d_web.catalog import Catalog
from gnome3d_web.regions import RegionError, parse_region


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        cat = Catalog.load(make_data(Path(d)))
        ct = cat.cell_types["GMX"]
        check("a cell type loads with absolute paths", ct.lymphoblastoid and ct.path("genes").is_file())
        check("an SV set lists its samples", cat.sv_sets["DEMO"].samples == ["P1", "P2"])
        genes = ct.genes()
        r = parse_region("chr1:1,000,000-2,500,000", genes)
        check("coordinates with commas parse", (r.chrom, r.start, r.end) == ("chr1", 1_000_000, 2_500_000))
        r = parse_region("alpha", genes)
        check("a gene name centres a 2 Mb window on its TSS", (r.chrom, r.start, r.end) == ("chr1", 310_000, 2_310_000), str(r))
        for bad in ("chr1:0-20000001", "NOPE", "chr1:500-100"):
            try:
                parse_region(bad, genes)
                ok = False
            except RegionError:
                ok = True
            check(f"{bad!r} is refused", ok)
    return done()


if __name__ == "__main__":
    sys.exit(main())
