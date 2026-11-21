"""Writes the hosted expression tables of the site from their published files.

    python web/scripts/make_expression_tables.py <out dir> \\
        --encode GM12878=<featureCounts table> --encode H1ESC=<table> --encode HFFC6=<table> \\
        --trios <counts of the nine people> --mage <TMM bed.gz> --geuvadis <GD660 RPKM txt.gz>

Every source is optional. The script writes one table for each source given and
`expression.json`, the entries of the catalog for them. `add_expression.py` puts both into a
site data directory.

ENCODE     read counts of featureCounts, one table for each cell line with a column for each
           replicate. The three tables must hold the same genes. A column gets the name of its
           cell line and of its ENCODE file.
TRIOS      read counts of the nine people of three 1000 Genomes trios, one library each, from
           the ENA study PRJEB11271, with the genes by name. The columns are named by family
           and member and get the sample ids.
MAGE       TMM values of 731 lymphoblastoid lines of the 1000 Genomes panel, from the file
           `TMM.filtered.TSS.MAGE.v1.0.bed.gz` of MAGE v1.0.
GEUVADIS   RPKM of 462 lymphoblastoid lines, from `GD660.GeneQuantRPKM.txt.gz` of E-GEUV-1. A
           person with several libraries gets their mean.

The table of the trios belongs to the feature `trios` of config.json and is offered while that
feature is on. The people of the trios, MAGE and GEUVADIS are lymphoblastoid cell lines. Their entries state it and
the site warns when such a table is taken beside another cell type.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# The columns of the trio counts by sample: the family, then daughter, mother or father.
TRIOS = {
    "hcs_f": "HG00512", "hcs_m": "HG00513", "hcs_d": "HG00514",
    "pur_f": "HG00731", "pur_m": "HG00732", "pur_d": "HG00733",
    "yur_m": "NA19238", "yur_f": "NA19239", "yur_d": "NA19240",
}  # fmt: skip


def encode(tables: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, object]]:
    """The replicates of every cell line in one table, with the length of each gene."""
    out: pd.DataFrame | None = None
    columns: dict[str, list[str]] = {}
    for cell, path in tables.items():
        t = pd.read_csv(path, sep="\t", comment="#").set_index("Geneid")
        samples = [c for c in t.columns if c not in ("Chr", "Start", "End", "Strand", "Length")]
        named = {c: f"{cell}_{Path(c).name.split('.')[0]}" for c in samples}
        columns[cell] = list(named.values())
        part = t[samples].rename(columns=named)
        if out is None:
            out = pd.concat([t["Length"].rename("length"), part], axis=1)
        else:
            if not out.index.equals(t.index) or not (out["length"] == t["Length"]).all():
                raise SystemExit(f"{path} does not hold the genes and lengths of the tables before it")
            out = pd.concat([out, part], axis=1)
    assert out is not None
    out.index.name = "gene_id"
    return out, {"label": "ENCODE RNA-seq", "kind": "counts", "cell_types": columns}


def trios(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    t = pd.read_csv(path, index_col=0)
    missing = [c for c in TRIOS if c not in t.columns]
    if missing:
        raise SystemExit(f"{path} has no column {', '.join(missing)}")
    out = t[list(TRIOS)].rename(columns=TRIOS)
    out.index.name = "gene"
    return out, {"label": "Trio mRNA-seq", "kind": "counts", "people": True, "lymphoblastoid": True, "feature": "trios"}


def mage(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    t = pd.read_csv(path, sep="\t")
    out = t.drop(columns=["#Chr", "start", "end"]).set_index("ID").round(3)
    out.index.name = "gene_id"
    return out, {"label": "MAGE RNA-seq", "kind": "values", "unit": "TMM", "people": True, "lymphoblastoid": True}


def geuvadis(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    t = pd.read_csv(path, sep="\t").drop(columns=["Gene_Symbol", "Chr", "Coord"]).set_index("TargetID")
    # A column is sample.library.lab_date_lane.
    out = t.T.groupby(t.columns.str.split(".").str[0]).mean().T.round(3)
    out.index.name = "gene_id"
    return out, {"label": "GEUVADIS RNA-seq", "kind": "values", "unit": "RPKM", "people": True, "lymphoblastoid": True}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--encode", action="append", default=[], metavar="CELL=TABLE")
    ap.add_argument("--trios")
    ap.add_argument("--mage")
    ap.add_argument("--geuvadis")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    made: dict[str, tuple[pd.DataFrame, dict[str, object]]] = {}
    if a.encode:
        made["ENCODE"] = encode({cell: Path(path) for cell, path in (item.split("=", 1) for item in a.encode)})
    if a.trios:
        made["TRIOS"] = trios(Path(a.trios))
    if a.mage:
        made["MAGE"] = mage(Path(a.mage))
    if a.geuvadis:
        made["GEUVADIS"] = geuvadis(Path(a.geuvadis))
    entries: dict[str, dict[str, object]] = {}
    for name, (table, entry) in made.items():
        dest = out / f"{name}.tsv"
        if dest.exists():
            raise SystemExit(f"{dest} exists; refusing to overwrite it")
        table.to_csv(dest, sep="\t")
        entries[name] = {**entry, "file": f"expression/{name}.tsv"}
        print(f"{name}: {table.shape[0]:,} genes, {table.shape[1] - ('length' in table.columns):,} samples, lowest value {table.drop(columns='length', errors='ignore').min().min():g}")
    (out / "expression.json").write_text(json.dumps(entries, indent=2) + "\n")


if __name__ == "__main__":
    main()
