"""A tandem duplication through the lift, the contacts, the sequence and the rules."""

from __future__ import annotations

import gzip
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gnome3d_web.regions import Region
from gnome3d_web.variants.vcf import read_svs, sample_variants
from personal3d import engine, rewrite
from personal3d.hicdiffusion import haplotype_window
from personal3d.lift import Lift
from personal3d.variants import SV


class Ref:
    """A reference whose base at a position is a letter of that position."""

    def fetch(self, chrom: str, a: int, b: int) -> str:
        return "".join("ACGT"[i % 4] for i in range(a, b))


@pytest.fixture(scope="module", autouse=True)
def shared() -> SimpleNamespace:
    dup = SV("chr1", 1001, 2001, "DUP")
    L = Lift([dup])
    return SimpleNamespace(L=L, dup=dup)


def test_a_position_before_the_duplication_stays(shared: SimpleNamespace) -> None:
    L = shared.L
    assert L(500) == 500, "a position before the duplication stays"
    assert L(1500) == 1500, "a position inside lifts to the first copy"
    assert L(2000) == 3000 and L(2500) == 3500, "a position after is shifted by the length"
    assert L.many(np.array([500, 1000, 1999, 2000, 4000])).tolist() == [500, 1000, 1999, 3000, 5000], "many agrees with one at a time"
    assert L.copies(np.array([999, 1000, 1999, 2000])).tolist() == [-1, 2000, 2999, -1], "copies gives the second copy inside and nothing outside"


def test_a_deletion_before_shifts_both_copies(shared: SimpleNamespace) -> None:
    dup = shared.dup
    both = Lift([SV("chr1", 101, 201, "DEL"), dup, SV("chr1", 3001, 3001, "INS", "A" * 50)])
    assert both(1500) == 1400 and both.copies(np.array([1500])).tolist() == [2400], "a deletion before shifts both copies"
    assert both(3500) == 3500 - 100 + 1000 + 50, "an insertion after adds to the duplication's shift"


def test_contacts_follow_the_copy_they_keep_their_distance_to(shared: SimpleNamespace) -> None:
    L = shared.L
    df = pd.DataFrame(
        [
            ["chr1", 100, 110, "chr1", 300, 310, 1],  # before
            ["chr1", 1100, 1110, "chr1", 1800, 1810, 2],  # inside
            ["chr1", 500, 510, "chr1", 1500, 1510, 3],  # from before to inside
            ["chr1", 1500, 1510, "chr1", 2500, 2510, 4],  # from inside to after
            ["chr1", 2500, 2510, "chr1", 2900, 2910, 5],  # after
        ]
    )
    got = engine.lift_bedpe(df, {"chr1": L})
    rows = sorted(tuple(int(v) for v in r[[1, 4, 6]]) for _, r in got.iterrows())
    assert rows == [(100, 300, 1), (500, 1500, 3), (1100, 1800, 2), (2100, 2800, 2), (2500, 3500, 4), (3500, 3900, 5)], "contacts follow the copy they keep their distance to: " + str(rows)


def test_the_sequence_has_the_duplicated_bases_twice(shared: SimpleNamespace) -> None:
    dup = shared.dup
    w = haplotype_window(Ref(), "chr1", 0, [dup], size=4000)  # type: ignore[arg-type]
    ref = Ref().fetch("chr1", 0, 4000)
    assert w.seq == ref[:2000] + ref[1000:2000] + ref[2000:3000], "the sequence has the duplicated bases twice"
    assert w.lift(2500) == 3500 and w.lift(1500) == 1500, "the window lifts a position after the duplication"


def test_the_loop_inside_is_there_twice_the_copy_one_length_on() -> None:
    # Two loops, one inside the duplicated interval and one wholly before it.
    clusters = [
        ["chr1", 10_000, 10_500, "chr1", 20_000, 20_500, "R", "L", 10],
        ["chr1", 52_000, 52_500, "chr1", 58_000, 58_500, "R", "L", 8],
        ["chr1", 80_000, 80_500, "chr1", 95_000, 95_500, "R", "L", 9],
    ]
    loops = rewrite.apply(clusters, rewrite.domains(clusters), {"chr1": 1_000_000}, [SV("chr1", 50_001, 60_001, "DUP")])
    spans = sorted((lp.a[0], lp.b[0]) for lp in loops)
    assert (52_000, 58_000) in spans and (62_000, 68_000) in spans, "the loop inside is there twice, the copy one length on: " + str(spans)
    assert (10_000, 20_000) in spans and (90_000, 105_000) in spans, "the loop before is unchanged and the loop after is shifted: " + str(spans)
    copy = next(lp for lp in loops if lp.a[0] == 62_000)
    assert copy.ref_a == (52_000, 52_500) and copy.ref_b == (58_000, 58_500), "the copy's anchors keep the reference intervals of their originals: " + str(copy)


def test_a_phased_duplication_is_read_onto_its_haplotype(tmp_path: Path) -> None:
    d = str(tmp_path)
    vcf = Path(d) / "dup.vcf.gz"
    with gzip.open(vcf, "wt") as f:
        f.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tP1\tP2\n")
        f.write("chr1\t50000\tu1\tN\t<DUP>\t.\tPASS\tSVTYPE=DUP;END=60000;SVLEN=10000\tGT\t0|1\t0|0\n")
    h0, h1, skipped = read_svs(vcf, "P1", Region("chr1", 0, 1_000_000))
    assert h0 == [] and [(s.kind, s.start, s.end) for s in h1] == [("DUP", 50_001, 60_001)] and skipped == 0, "a phased duplication is read onto its haplotype: " + str(h1)
    records, other = sample_variants(vcf, "P1", Region("chr1", 0, 1_000_000))
    assert [(r["type"], r["length"]) for r in records] == [("DUP", 10_000)] and other == {}, "the dialog lists it with its length: " + str(records)
