"""SVs from a VCF for one sample and region.

    pytest web/tests/test_svs.py
"""

from __future__ import annotations

from pathlib import Path

from fixture import make_data

from gnome3d_web.regions import Region
from gnome3d_web.svs import read_svs


def test_a_phased_deletion_lands_on_the_first_haplotype(tmp_path: Path) -> None:
    d = str(tmp_path)
    root = make_data(Path(d))
    vcf = root / "svs" / "demo.vcf.gz"
    h0, h1, skipped = read_svs(vcf, "P1", Region("chr1", 0, 3_000_000))
    assert [(s.kind, s.start, s.end) for s in h0] == [("DEL", 900_001, 960_001)], "a phased deletion lands on the first haplotype: " + str(h0)
    assert [(s.kind, s.start, len(s.seq)) for s in h1] == [("INS", 1_700_001, 56)], "an insertion with its sequence lands on the second haplotype: " + str([(s.kind, s.start, len(s.seq)) for s in h1])
    assert skipped == 1, "a CNV is skipped and counted"
    h0, h1, _ = read_svs(vcf, "P1", Region("chr1", 1_000_000, 3_000_000))
    assert h0 == [] and len(h1) == 1, "variants outside the region are left out"
    h0, h1, skipped = read_svs(vcf, "P2", Region("chr1", 0, 3_000_000))
    assert h0 == [] and h1 == [], "a sample with no alternate alleles has none"
    assert skipped == 0, "a record of another type that the sample does not carry is not counted: " + str(skipped)
