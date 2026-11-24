"""SVs from a VCF for one sample and region."""

from __future__ import annotations

from pathlib import Path

import pytest
from fixture import make_data

from gnome3d_web.regions import Region
from gnome3d_web.variants.vcf import UnknownSample, read_svs, sample_variants


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


def test_the_preview_and_the_job_read_the_same_variants(tmp_path: Path) -> None:
    vcf = make_data(tmp_path) / "svs" / "demo.vcf.gz"
    # The deletion is the record at 900,000 and its first deleted base is 900,001.
    for start in (0, 900_000, 900_001, 900_002):
        region = Region("chr1", start, 3_000_000)
        h0, h1, skipped = read_svs(vcf, "P1", region)
        records, other = sample_variants(vcf, "P1", region)
        for h, hap in enumerate((h0, h1)):
            assert [r["sv"] for r in records if r["haps"][h]] == hap, f"the preview lists the variants the job applies on haplotype {h}, from {start}: " + str(records)  # type: ignore[index]
        assert sum(other.values()) == skipped, f"both count the same skipped records, from {start}"
    assert [s.start for s in read_svs(vcf, "P1", Region("chr1", 900_001, 3_000_000))[0]] == [900_001], "a deletion that begins at the first base of the region is inside"
    assert read_svs(vcf, "P1", Region("chr1", 900_002, 3_000_000))[0] == [], "one that begins a base before the region is not"


def test_a_sample_the_vcf_lacks_is_one_error_for_both(tmp_path: Path) -> None:
    vcf = make_data(tmp_path) / "svs" / "demo.vcf.gz"
    for read in (read_svs, sample_variants):
        with pytest.raises(UnknownSample, match="NOBODY"):
            read(vcf, "NOBODY", Region("chr1", 0, 3_000_000))
    with pytest.raises(UnknownSample, match="FORMAT"):
        read_svs(vcf, "FORMAT", Region("chr1", 0, 3_000_000))


# A network with anchors that share a start, rewritten by an inversion and a deletion. The loops
# of each result are printed in sorted order.
REWRITE = """
import random, sys
sys.path[:0] = [{web!r}, {vendor!r}]
from personal3d import motifs, rewrite
from personal3d.variants import SV
rng = random.Random(3)
starts = sorted(rng.sample(range(10_000, 3_000_000, 5_000), 90))
anchors = [(s, s + rng.choice([400, 900, 1500])) for s in starts] + [(s, s + 2200) for s in starts[::3]]
clusters = []
for _ in range(400):
    a, b = sorted(rng.sample(anchors, 2))
    if a[1] < b[0]:
        clusters.append(["chr1", a[0], a[1], "chr1", b[0], b[1], rng.choice("RLN"), rng.choice("RLN"), float(rng.randint(3, 40))])
for sv in ([SV("chr1", 800_000, 1_400_000, "INV")], [SV("chr1", 2_000_000, 2_300_000, "DEL")]):
    loops = rewrite.apply(clusters, rewrite.domains(clusters), {{"chr1": 4_000_000}}, sv, None, None, motifs.THRESHOLD)
    print(sorted((lp.a, lp.b, lp.orient, lp.weight, lp.ref_a, lp.ref_b) for lp in loops))
"""


def test_the_rewritten_loops_do_not_depend_on_the_hash_seed_of_the_process() -> None:
    import os
    import subprocess
    import sys

    web = Path(__file__).resolve().parents[1]
    code = REWRITE.format(web=str(web), vendor=str(web / "vendor"))
    said = [subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": seed}, capture_output=True, text=True, check=True).stdout for seed in ("1", "2", "3")]
    assert len(said[0]) > 1000, "the network is rewritten: " + said[0][:200]
    assert said[0] == said[1] == said[2], "the loops after the variants are the same under three hash seeds"
