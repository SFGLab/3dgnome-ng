"""SVs from a VCF for one sample and region.

    python web/tests/test_svs.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from _check import check, done
from fixture import make_data

from gnome3d_web.regions import Region
from gnome3d_web.svs import read_svs


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        root = make_data(Path(d))
        vcf = root / "svs" / "demo.vcf.gz"
        h0, h1, skipped = read_svs(vcf, "P1", Region("chr1", 0, 3_000_000))
        check("a phased deletion lands on the first haplotype", [(s.kind, s.start, s.end) for s in h0] == [("DEL", 900_001, 960_001)], str(h0))
        check(
            "an insertion with its sequence lands on the second haplotype",
            [(s.kind, s.start, len(s.seq)) for s in h1] == [("INS", 1_700_001, 56)],
            str([(s.kind, s.start, len(s.seq)) for s in h1]),
        )
        check("a CNV is skipped and counted", skipped == 1)
        h0, h1, _ = read_svs(vcf, "P1", Region("chr1", 1_000_000, 3_000_000))
        check("variants outside the region are left out", h0 == [] and len(h1) == 1)
        h0, h1, _ = read_svs(vcf, "P2", Region("chr1", 0, 3_000_000))
        check("a sample with no alternate alleles has none", h0 == [] and h1 == [])
    return done()


if __name__ == "__main__":
    sys.exit(main())
