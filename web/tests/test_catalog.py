"""Catalog loading and region parsing.

    pytest web/tests/test_catalog.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fixture import make_data

from gnome3d_web.catalog.model import Catalog
from gnome3d_web.regions import RegionError, parse_region


@pytest.fixture(scope="module", autouse=True)
def shared(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    d = str(tmp_path_factory.mktemp("work"))
    cat = Catalog.load(make_data(Path(d)))
    ct = cat.cell_types["GMX"]
    return SimpleNamespace(ct=ct, cat=cat, d=d)


def test_a_cell_type_loads_with_absolute_paths(shared: SimpleNamespace) -> None:
    ct, cat, d = shared.ct, shared.cat, shared.d
    assert ct.lymphoblastoid and ct.path("genes").is_file(), "a cell type loads with absolute paths"
    assert cat.sv_sets["DEMO"].samples == ["P1", "P2"], "an SV set lists its samples"
    rows = cat.sv_sets["DEMO"].sample_rows()
    assert rows == [["P1", "female", "PUR", "AMR", "", "P2"], ["P2", "female", "PUR", "AMR", "", ""]], "a sample table gives sex, population and the parents the set holds: " + str(rows)
    assert cat.sv_sets["BARE"].sample_rows() == [], "a call set without a table has no rows"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from add_sample_table import read_pedigree, table_rows

    ped = Path(d) / "ped.txt"
    ped.write_text("FamilyID SampleID FatherID MotherID Sex Population Superpopulation\nF1 C1 A1 A2 1 YRI AFR\nF1 A1 0 0 1 YRI AFR\nF2 C2 X9 A2 2 YRI AFR\n")
    rows, counts = table_rows(["C1", "A1", "C2", "Z"], read_pedigree(ped))
    assert rows == [["C1", "male", "YRI", "AFR", "A1", ""], ["A1", "male", "YRI", "AFR", "", ""], ["C2", "female", "YRI", "AFR", "", ""], ["Z", "", "", "", "", ""]], "the pedigree becomes one row per sample of the set: " + str(rows)
    assert counts["parents outside the set"] == 3 and counts["with a parent in the set"] == 1 and counts["not in the pedigree"] == 1, "the script counts parents outside the set: " + str(counts)
    genes = ct.genes()
    r = parse_region("chr1:1,000,000-2,500,000", genes)
    assert (r.chrom, r.start, r.end) == ("chr1", 1_000_000, 2_500_000), "coordinates with commas parse"
    r = parse_region("alpha", genes)
    assert (r.chrom, r.start, r.end) == ("chr1", 310_000, 2_310_000), "a gene name centres a 2 Mb window on its TSS: " + str(r)
    for bad in ("chr1:0-20000001", "NOPE", "chr1:500-100"):
        try:
            parse_region(bad, genes)
            ok = False
        except RegionError:
            ok = True
        assert ok, f"{bad!r} is refused"
