"""The call set of the Human Genome Diversity Project, written by web/scripts/make_hgdp_sv_set.py."""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from add_sample_table import read_pedigree, table_rows  # noqa: E402
from make_hgdp_sv_set import main  # noqa: E402

from gnome3d_web.regions import Region  # noqa: E402
from gnome3d_web.variants.vcf import sample_variants  # noqa: E402

COLS = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tERS1\tLP1-DNA_A01\n"
FMT = "GT:FT:AD"


def _gz(path: Path, text: str) -> Path:
    with gzip.open(path, "wt") as fh:
        fh.write("##fileformat=VCFv4.2\n" + COLS + text)
    return path


@pytest.fixture
def hgdp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, str]:
    """The call set and the table of a project of two samples, with the lines the script printed."""
    (tmp_path / "HGDP.metadata").write_text(
        "sample\tlibrary\tsample_accession\tsource\tlibrary_type\tsource.library_type\tpopulation\tregion\tsex\tmeancov\n"
        "HGDP00001\tHGDP00001.111\tERS1\tsanger\tPCRfree\tsanger.PCRfree\tHazara\tCENTRAL_SOUTH_ASIA\tM\t36\n"
        "HGDP00002\tHGDP00002.LP1-DNA_A01\tERS2\tsgdp\tPCRfree\tsgdp.PCRfree\tAdygei\tEUROPE\tF\t41\n"
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    _gz(
        raw / "chr1.raw.vcf.gz",
        f"chr1\t5000\tchr1:5000:FG\tN\t<INS:SVSIZE=6:AGGREGATED>\t50\tPASS\tEND=5000;SEQ=ACGTAC;SVTYPE=INS\t{FMT}\t0/1:PASS:5,5\t0/0:PASS:9,0\n"
        f"chr1\t5000\tchr1:5000:FG.0\tN\t<INS:SVSIZE=6:BREAKPOINT>\t50\tPASS\tEND=5000;SEQ=TTTTTT;SVTYPE=INS\t{FMT}\t0/1:PASS:5,5\t0/0:PASS:9,0\n"
        f"chr1\t6000\tchr1:6000:FG\tN\t<INS:SVSIZE=40+:AGGREGATED>\t50\tPASS\tEND=6000;LEFT_SVINSSEQ=AAAA;SVTYPE=INS\t{FMT}\t0/1:PASS:5,5\t0/0:PASS:9,0\n",
    )
    files = [
        _gz(
            tmp_path / "del.manta.vcf.gz",
            f"chr1\t1000\tchr1:1000:DG\tN\t<DEL:SVSIZE=100:AGGREGATED>\t90\tPASS\t.\t{FMT}\t0/1:PASS:5,5\t1/1:PASS:0,9\n"
            f"chr1\t2000\tchr1:2000:DG\tN\t<DEL:SVSIZE=50:AGGREGATED>\t90\tPratio\t.\t{FMT}\t0/1:PASS:5,5\t0/0:PASS:9,0\n"
            f"chr1\t3000\tchr1:3000:DG\tN\t<DEL:SVSIZE=70:AGGREGATED>\t90\tPASS\t.\t{FMT}\t0/1:FAIL1:5,1\t0/0:PASS:9,0\n"
            f"chr1\t4000\tchr1:4000:DG\tN\t<DEL:SVSIZE=30:AGGREGATED>\t90\tPASS\t.\t{FMT}\t0/1:FAIL1:5,1\t0/1:PASS:4,5\n",
        ),
        _gz(
            tmp_path / "ins.manta.vcf.gz",
            f"chr1\t5000\tchr1:5000:FG\tN\t<INS:SVSIZE=6:AGGREGATED>\t50\tPASS\t.\t{FMT}\t0/1:PASS:5,5\t0/0:PASS:9,0\n"
            f"chr1\t6000\tchr1:6000:FG\tN\t<INS:SVSIZE=40::AGGREGATED>\t50\tPASS\t.\t{FMT}\t0/1:PASS:5,5\t0/0:PASS:9,0\n"
            f"chrX\t6500\tchrX:6500:FG\tN\t<INS:SVSIZE=20:AGGREGATED>\t50\tPASS\t.\t{FMT}\t0/1:PASS:5,5\t0/0:PASS:9,0\n",
        ),
        _gz(tmp_path / "dup.manta.vcf.gz", f"chr1\t7000\tchr1:7000:UG\tN\t<DUP:SVSIZE=200:AGGREGATED>\t70\tPASS\tAC=1;AN=4\t{FMT}\t0/0:PASS:9,0\t0/1:PASS:5,5\n"),
        _gz(
            tmp_path / "inv.manta.vcf.gz",
            f"chr1\t8000\tchr1:8000:TG\tN\t<INV:SVSIZE=300:BREAKPOINT1>\t40\tPASS\t.\t{FMT}\t0/1:PASS:5,5\t0/0:PASS:9,0\n"
            f"chr1\t8300\tchr1:8300:TG\tN\t<INV:SVSIZE=300:BREAKPOINT2>\t60\tPASS\t.\t{FMT}\t0/1:PASS:5,5\t0/0:PASS:9,0\n"
            f"chr1\t9500\tchr1:9500:TG\tN\t<INV:SVSIZE=400:BREAKPOINT2>\t60\tPASS\t.\t{FMT}\t0/0:PASS:9,0\t0/1:PASS:5,5\n",
        ),
    ]
    out, table = tmp_path / "HGDP.vcf", tmp_path / "HGDP.samples.tsv"
    monkeypatch.setattr(sys, "argv", ["make_hgdp_sv_set.py", str(tmp_path / "HGDP.metadata"), str(raw), str(out), str(table), *map(str, files)])
    main()
    packed = tmp_path / "HGDP.vcf.gz"
    with gzip.open(packed, "wt") as fh:
        fh.write(out.read_text())
    return packed, table, out.read_text()


def _rows(text: str) -> list[list[str]]:
    return [line.split("\t") for line in text.splitlines() if not line.startswith("#")]


def test_the_columns_are_the_names_of_the_project(hgdp: tuple[Path, Path, str]) -> None:
    header = next(line for line in hgdp[2].splitlines() if line.startswith("#CHROM"))
    assert header.split("\t")[9:] == ["HGDP00001", "HGDP00002"], "an accession and a library name both map to the name of the sample"


def test_only_passing_records_with_a_passing_carrier_are_kept(hgdp: tuple[Path, Path, str]) -> None:
    ids = [r[2] for r in _rows(hgdp[2])]
    assert ids == ["chr1:1000:DG", "chr1:4000:DG", "chr1:5000:FG", "chr1:7000:UG", "chr1:8300:TG", "chr1:9500:TG", "chrX:6500:FG"], (
        "left out: a record that is not PASS, a record whose one carrier failed its filter, an insertion of unknown length"
        " and the inversion with the span and the lower QUAL of another"
    )
    failed = next(r for r in _rows(hgdp[2]) if r[2] == "chr1:4000:DG")
    assert failed[9:] == ["./.", "0/1"], "a genotype that failed its filter is missing"


def test_the_spans_and_the_sequence(hgdp: tuple[Path, Path, str]) -> None:
    by_id = {r[2]: r for r in _rows(hgdp[2])}
    assert by_id["chr1:1000:DG"][7] == "SVTYPE=DEL;END=1100;SVLEN=-100", "a deletion spans POS to POS plus its size"
    assert by_id["chr1:8300:TG"][1] == "8000" and by_id["chr1:8300:TG"][7] == "SVTYPE=INV;END=8300;SVLEN=300", (
        "an inversion called at its second breakpoint starts its size before it"
    )
    assert by_id["chr1:9500:TG"][1] == "9100", "the same for an inversion without a twin"
    assert by_id["chr1:5000:FG"][4] == "NACGTAC", "an insertion has the sequence of the aggregated record of the raw calls"
    assert by_id["chrX:6500:FG"][4] == "<INS>" and by_id["chrX:6500:FG"][7] == "SVTYPE=INS;END=6500;SVLEN=20", (
        "an insertion without a raw record keeps its size"
    )


def test_the_site_reads_the_set(hgdp: tuple[Path, Path, str]) -> None:
    records, skipped = sample_variants(hgdp[0], "HGDP00001", Region("chr1", 1, 20000))
    assert [(r["type"], r["pos"], r["length"]) for r in records] == [("DEL", 1000, 100), ("INS", 5000, 6), ("INV", 8000, 300)], (
        "the first sample carries a deletion, an insertion of six bases and an inversion"
    )
    assert skipped == {}, "every record of the set has a type the site applies"
    assert all(r["haps"] == (True, False) for r in records), "an unphased genotype is on the first haplotype"
    second, _ = sample_variants(hgdp[0], "HGDP00002", Region("chr1", 1, 20000))
    assert [r["type"] for r in second] == ["DEL", "DEL", "DUP", "INV"], "the second sample carries its four"


def test_the_table_of_the_samples(hgdp: tuple[Path, Path, str]) -> None:
    rows, counts = table_rows(["HGDP00001", "HGDP00002"], read_pedigree(hgdp[1]))
    assert rows == [["HGDP00001", "male", "Hazara", "CENTRAL_SOUTH_ASIA", "", ""], ["HGDP00002", "female", "Adygei", "EUROPE", "", ""]], (
        "the table of the script is read as it is"
    )
    assert counts["not in the pedigree"] == 0, "every sample of the set is in its table"
