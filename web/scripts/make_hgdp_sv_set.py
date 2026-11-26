"""Writes the structural variants of the Human Genome Diversity Project as a call set of the site.

    python web/scripts/make_hgdp_sv_set.py HGDP.metadata RAW_DIR OUT.vcf OUT.samples.tsv \\
        del.manta.vcf.gz ins.manta.vcf.gz dup.manta.vcf.gz inv.manta.vcf.gz

The inputs are files of Almarri et al., Cell 2020, under
https://ngs.sanger.ac.uk/production/hgdp/hgdp_structural_variation/. They are the metadata of the
samples and the calls of Manta genotyped with Graphtyper, one file a type, on GRCh38. RAW_DIR
holds the files of `SV_CALLSET/RAW/`, the calls before the filters, one file a chromosome.

A record of those files has no INFO. Its type and its size are in the ALT field, as in
`<DEL:SVSIZE=54:AGGREGATED>`. The site reads SVTYPE with END or SVLEN from INFO. This script writes
one sorted VCF with those fields and the genotype alone, by these rules.

- A record is kept when its FILTER is PASS and a sample carries it.
- A genotype is kept when its own filter, FT, is PASS. Any other genotype is written as missing.
- A deletion, a duplication and an inversion of the model BREAKPOINT1 span POS to POS plus the
  size. An inversion of the model BREAKPOINT2 is placed at its second breakpoint and spans POS
  minus the size to POS. Two inversions with one span are one record, the one of the higher QUAL.
- An insertion with a known sequence gets that sequence as its ALT allele. The files by type hold
  no sequence. The raw calls have it in the field SEQ of the record with the same ID. They cover
  chromosomes 1 to 22. An insertion without a raw record keeps its size alone.
- An insertion of unknown length is left out. The raw calls give its two flanks and its least
  size, `SVSIZE=435+`. The files by type have it as `<INS:SVSIZE=435::AGGREGATED>`.
- A column is the accession of a sample at ENA or, for the 120 samples that SGDP sequenced, the
  name of its library. The metadata gives the ids of the project, HGDP00129 for one.

The genotypes are unphased. OUT.vcf is plain text, for bgzip and tabix. OUT.samples.tsv is the
table of the samples for `add_sample_table.py`. Its columns are sample, sex, population,
superpopulation, father and mother, the last two empty.
"""

from __future__ import annotations

import gzip
import re
import sys
from pathlib import Path

ALT = re.compile(r"^<(DEL|INS|DUP|INV):SVSIZE=(\d+):(:?)([A-Z0-9]+)>$")
SEQ = re.compile(r"(?:^|;)SEQ=([ACGTNacgtn]+)(?:;|$)")
CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
SEX = {"M": "male", "F": "female"}
HEADER = (
    "##fileformat=VCFv4.2\n"
    "##source=HGDP structural variants, Almarri et al., Cell 2020, Manta and Graphtyper\n"
    "##reference=GRCh38\n"
    '##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">\n'
    '##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant">\n'
    '##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="Length of the variant in bp">\n'
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype, unphased, missing when the call did not pass its filter">\n'
)


def read_metadata(path: Path) -> dict[str, dict[str, str]]:
    """Returns the rows of the metadata by the name of the column of the sample in the calls: the
    accession at ENA for a sample sequenced at Sanger and the name of the library for one of SGDP."""
    rows: dict[str, dict[str, str]] = {}
    with path.open() as fh:
        col = {name: i for i, name in enumerate(fh.readline().rstrip("\n").split("\t"))}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            row = {
                "sample": f[col["sample"]],
                "sex": SEX.get(f[col["sex"]], ""),
                "population": f[col["population"]],
                "superpopulation": f[col["region"]],
            }
            rows[f[col["sample_accession"]]] = row
            rows[f[col["library"]].split(".", 1)[1]] = row
    return rows


def insertion_sequences(raw: Path) -> dict[str, str]:
    """Returns the inserted sequence of the insertions of the raw calls by the ID of the record."""
    out: dict[str, str] = {}
    for path in sorted(raw.glob("*.raw.vcf.gz")):
        with gzip.open(path, "rt") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                x = line.split("\t", 8)
                if not x[4].startswith("<INS") or not x[4].endswith(":AGGREGATED>"):
                    continue
                m = SEQ.search(x[7])
                if m is not None:
                    out[x[2]] = m.group(1).upper()
    return out


def span(kind: str, model: str, pos: int, size: int) -> tuple[int, int]:
    """Returns POS and END of a record of the site for a call of the project."""
    if kind == "INS":
        return pos, pos
    if kind == "INV" and model == "BREAKPOINT2":
        return pos - size, pos
    return pos, pos + size


def genotypes(fields: list[str], ft: int) -> tuple[list[str], bool]:
    """Returns the genotypes of a record with those that failed their filter as missing, and
    whether a sample carries the variant."""
    out = []
    carried = False
    for cell in fields:
        part = cell.split(":")
        gt = part[0]
        if len(part) <= ft or part[ft] != "PASS":
            out.append("./.")
            continue
        out.append(gt)
        carried = carried or any(a not in ("0", ".") for a in gt.replace("|", "/").split("/"))
    return out, carried


def convert(paths: list[Path], names: dict[str, str], sequences: dict[str, str]) -> tuple[list[str], dict[str, list[tuple[int, int, str, float, str]]], dict[str, int]]:
    """Returns the sample ids in the order of the columns, the records by chromosome as position,
    end, type, QUAL and line. Also returns the counts that the script prints."""
    samples: list[str] = []
    records: dict[str, list[tuple[int, int, str, float, str]]] = {c: [] for c in CHROMS}
    counts = {"read": 0, "not PASS": 0, "insertion of unknown length": 0, "insertion kept without a sequence": 0, "no carrier with a PASS genotype": 0, "other chromosome": 0, "kept": 0}
    for path in paths:
        order: list[int] = []
        with gzip.open(path, "rt") as fh:
            for line in fh:
                if line.startswith("##"):
                    continue
                x = line.rstrip("\n").split("\t")
                if line.startswith("#"):
                    here = [names[a] for a in x[9:]]
                    if not samples:
                        samples = here
                    order = [here.index(s) for s in samples]
                    continue
                counts["read"] += 1
                if x[6] != "PASS":
                    counts["not PASS"] += 1
                    continue
                if x[0] not in records:
                    counts["other chromosome"] += 1
                    continue
                m = ALT.match(x[4])
                if m is None:
                    raise SystemExit(f"{path}: ALT {x[4]!r} at {x[0]}:{x[1]} is not a call of a known form")
                kind, size, model = m.group(1), int(m.group(2)), m.group(4)
                if m.group(3):
                    counts["insertion of unknown length"] += 1
                    continue
                alt = f"<{kind}>"
                if kind == "INS" and x[2] in sequences:
                    alt = "N" + sequences[x[2]]
                    size = len(sequences[x[2]])
                elif kind == "INS":
                    counts["insertion kept without a sequence"] += 1
                fmt = x[8].split(":")
                cells = x[9:]
                gts, carried = genotypes([cells[k] for k in order], fmt.index("FT"))
                if not carried:
                    counts["no carrier with a PASS genotype"] += 1
                    continue
                pos, end = span(kind, model, int(x[1]), size)
                info = f"SVTYPE={kind};END={end};SVLEN={-size if kind == 'DEL' else size}"
                row = "\t".join([x[0], str(pos), x[2], "N", alt, x[5], "PASS", info, "GT", *gts])
                records[x[0]].append((pos, end, kind, float(x[5]), row))
                counts["kept"] += 1
    return samples, records, counts


def main() -> None:
    metadata, raw, out, table = (Path(p) for p in sys.argv[1:5])
    meta = read_metadata(metadata)
    sequences = insertion_sequences(raw)
    samples, records, counts = convert([Path(p) for p in sys.argv[5:]], {a: m["sample"] for a, m in meta.items()}, sequences)
    merged = 0
    by_type: dict[str, int] = {}
    with out.open("w") as fh:
        fh.write(HEADER)
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples) + "\n")
        for chrom in CHROMS:
            best: dict[tuple[int, int, str], tuple[int, int, str, float, str]] = {}
            for rec in records[chrom]:
                key = (rec[0], rec[1], rec[2])
                if rec[2] == "INV" and key in best:
                    merged += 1
                    if rec[3] <= best[key][3]:
                        continue
                    best[key] = rec
                elif rec[2] == "INV":
                    best[key] = rec
            rows = [r for r in records[chrom] if r[2] != "INV"] + list(best.values())
            for rec in sorted(rows, key=lambda r: (r[0], r[1])):
                fh.write(rec[4] + "\n")
                by_type[rec[2]] = by_type.get(rec[2], 0) + 1
    by_name = {m["sample"]: m for m in meta.values()}
    table.write_text(
        "sample\tsex\tpopulation\tsuperpopulation\tfather\tmother\n"
        + "".join(f"{s}\t{by_name[s]['sex']}\t{by_name[s]['population']}\t{by_name[s]['superpopulation']}\t\t\n" for s in samples)
    )
    print(f"{len(samples)} samples, {sum(by_type.values()):,} records: " + ", ".join(f"{k} {v:,}" for k, v in sorted(by_type.items())))
    for k, v in counts.items():
        print(f"  {k}: {v:,}")
    print(f"  inversions with the span of another: {merged:,}")


if __name__ == "__main__":
    main()
