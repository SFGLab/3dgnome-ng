"""A tiny hosted data directory: one cell type on chr1:0-4 Mb and one SV call set."""

from __future__ import annotations

import gzip
import json
from pathlib import Path


def make_data(root: Path) -> Path:
    ct = root / "cell_types" / "GMX"
    ct.mkdir(parents=True)
    anchors = [(100_000 + i * 200_000, 101_000 + i * 200_000) for i in range(15)]
    (ct / "anchors_ctcf.bed").write_text("".join(f"chr1\t{s}\t{e}\tR\n" for s, e in anchors))
    (ct / "anchors_ctcf_rnapii.bed").write_text(
        "".join(f"chr1\t{s}\t{e}\tR\n" for s, e in anchors) + "chr1\t1500000\t1501000\tN\n"
    )
    loops = "".join(
        f"chr1\t{anchors[i][0]}\t{anchors[i][1]}\tchr1\t{anchors[i + 1][0]}\t{anchors[i + 1][1]}\t{5 + i}\n"
        for i in range(14)
    )
    (ct / "ctcf.bedpe").write_text(loops)
    (ct / "rnapii.bedpe").write_text("chr1\t1300000\t1301000\tchr1\t1500000\t1501000\t4\n")
    (ct / "singletons.bedpe").write_text(
        "".join(f"chr1\t{a}\t{a + 25000}\tchr1\t{a + 50000}\t{a + 75000}\t3\n" for a in range(0, 3_900_000, 100_000))
    )
    (ct / "segments.bed").write_text("chr1\t0\t0\nchr1\t2000000\t2000000\n")
    (ct / "centromeres.bed").write_text("chr1\t121700000\t125100000\n")
    (ct / "compartments.bed").write_text("chr1\t0\t1000000\tA\nchr1\t1000000\t2500000\tB\nchr1\t2500000\t4000000\tA\nchr2\t0\t1000000\tB\n")
    (ct / "config.ini").write_text("[main]\nmax_pet_length = 1000000\n\n[data]\n")
    (ct / "genes.tsv").write_text("gene_id\tname\tchrom\ttss\tstrand\nG1\tALPHA\tchr1\t1310000\t+\nG2\tBETA\tchr1\t2600000\t-\n")
    (ct / "enhancers.bed").write_text("chr1\t1490000\t1492000\nchr1\t2700000\t2702000\n")
    svs = root / "svs"
    svs.mkdir()
    with gzip.open(svs / "demo.vcf.gz", "wt") as f:
        f.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tP1\tP2\n")
        f.write("chr1\t900000\td1\tN\t<DEL>\t.\tPASS\tSVTYPE=DEL;END=960000;SVLEN=-60000\tGT\t1|0\t0|0\n")
        f.write("chr1\t1700000\ti1\tN\tNACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\t.\tPASS\tSVTYPE=INS;SVLEN=56\tGT\t0|1\t0|0\n")
        f.write("chr1\t2500000\tb1\tN\t<CNV>\t.\tPASS\tSVTYPE=CNV;END=2600000\tGT\t1|1\t0|0\n")
    # P1 is the child of P2 and of a father the call set does not hold.
    (svs / "demo.samples.tsv").write_text(
        "sample\tsex\tpopulation\tsuperpopulation\tfather\tmother\nP1\tfemale\tPUR\tAMR\tP0\tP2\nP2\tfemale\tPUR\tAMR\t\t\n")
    catalog = {
        "cell_types": {
            "GMX": {
                "label": "GMX, lymphoblastoid", "lymphoblastoid": True, "dir": "cell_types/GMX",
                "anchors": {"CTCF": "anchors_ctcf.bed", "CTCF+RNAPII": "anchors_ctcf_rnapii.bed"},
                "clusters": {"CTCF": "ctcf.bedpe", "RNAPII": "rnapii.bedpe"},
                "singletons": "singletons.bedpe", "segments": "segments.bed", "centromeres": "centromeres.bed",
                "config": "config.ini", "genes": "genes.tsv", "enhancers": "enhancers.bed", "note": "",
                "compartments": "compartments.bed",
            }
        },
        "sv_sets": {
            "DEMO": {"label": "demo call set", "vcf": "svs/demo.vcf.gz", "samples": ["P1", "P2"], "samples_table": "svs/demo.samples.tsv"},
            "BARE": {"label": "a call set without a table", "vcf": "svs/demo.vcf.gz", "samples": ["P1", "P2"]},
        },
    }  # fmt: skip
    (root / "catalog.json").write_text(json.dumps(catalog, indent=2))
    return root
