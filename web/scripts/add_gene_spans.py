"""Adds the span of each gene to the gene table of the cell types of a data directory.

    python web/scripts/add_gene_spans.py DATA_DIR gencode.v40.annotation.gtf.gz

A genes.tsv made before 2026-11-24 has gene_id, name, chrom, tss and strand. The drawings of the
genes under the loops need the start and the end of a gene as well. The script reads them from the
gene lines of the annotation the table was made from, by gene_id. It writes the table again with
the columns start and end, 0 based with the end left out. The table as it was is kept beside it as
genes.tsv.before-spans. A table that has the columns is left as it is. A gene the annotation lacks
keeps its TSS as its span and is counted.

The script needs no package. Restart the site after it: the site reads a gene table once.
"""

from __future__ import annotations

import gzip
import shutil
import sys
from pathlib import Path


def spans(gtf: Path) -> dict[str, tuple[str, str]]:
    """Returns the start and the end of every gene line of the annotation by gene_id."""
    opener = gzip.open if str(gtf).endswith(".gz") else open
    out: dict[str, tuple[str, str]] = {}
    with opener(gtf, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            gid = next((a.strip().split(" ", 1)[1].strip('"') for a in f[8].split(";") if a.strip().startswith("gene_id ")), "")
            out[gid] = (str(int(f[3]) - 1), f[4])
    return out


def main() -> None:
    data, gtf = Path(sys.argv[1]), Path(sys.argv[2])
    known = spans(gtf)
    print(f"{len(known):,} genes in {gtf.name}")
    for table in sorted(data.glob("cell_types/*/genes.tsv")):
        lines = table.read_text().splitlines()
        head = lines[0].split("\t")
        if "start" in head and "end" in head:
            print(f"{table}: has the spans already")
            continue
        gid, tss = head.index("gene_id"), head.index("tss")
        rows, missing = ["\t".join([*head, "start", "end"])], 0
        for line in lines[1:]:
            f = line.split("\t")
            start, end = known.get(f[gid], ("", ""))
            if not start:
                missing += 1
                start, end = f[tss], str(int(f[tss]) + 1)
            rows.append("\t".join([*f, start, end]))
        kept = table.with_name("genes.tsv.before-spans")
        assert not kept.exists(), f"{kept} is there already"
        shutil.copy2(table, kept)
        fresh = table.with_name("genes.tsv.with-spans")
        fresh.write_text("\n".join(rows) + "\n")
        fresh.replace(table)
        print(f"{table}: {len(rows) - 1:,} genes, {missing:,} not in the annotation, the table before is {kept.name}")


if __name__ == "__main__":
    main()
