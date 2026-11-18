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
        rows = cat.sv_sets["DEMO"].sample_rows()
        check("a sample table gives sex, population and the parents the set holds", rows == [["P1", "female", "PUR", "AMR", "", "P2"], ["P2", "female", "PUR", "AMR", "", ""]], str(rows))
        check("a call set without a table has no rows", cat.sv_sets["BARE"].sample_rows() == [])
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from add_sample_table import read_pedigree, table_rows

        ped = Path(d) / "ped.txt"
        ped.write_text("FamilyID SampleID FatherID MotherID Sex Population Superpopulation\nF1 C1 A1 A2 1 YRI AFR\nF1 A1 0 0 1 YRI AFR\nF2 C2 X9 A2 2 YRI AFR\n")
        rows, counts = table_rows(["C1", "A1", "C2", "Z"], read_pedigree(ped))
        check("the pedigree becomes one row per sample of the set", rows == [["C1", "male", "YRI", "AFR", "A1", ""], ["A1", "male", "YRI", "AFR", "", ""], ["C2", "female", "YRI", "AFR", "", ""], ["Z", "", "", "", "", ""]], str(rows))
        check("the script counts parents outside the set", counts["parents outside the set"] == 3 and counts["with a parent in the set"] == 1 and counts["not in the pedigree"] == 1, str(counts))
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
