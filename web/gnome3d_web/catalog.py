"""The hosted data, described by `catalog.json` at the root of the data directory.

A cell type is a directory of the files one modelling run reads, the base loops per factor,
the anchors per factor choice, the base contact map as singletons, segments, centromeres and a
production config, plus the genes and enhancers the result pages mark. `contact_map`, when
given, is a cooler URI at 1 kb on the server that the stretches without loops are drawn against. An SV set is one
bgzipped VCF and the samples it holds. `samples_table`, when given, is a table of those samples
with sex, population and parents. Paths in the catalog are relative to the data directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

FACTOR_CHOICES = ("CTCF", "RNAPII", "CTCF+RNAPII")
BACKGROUNDS = ("hic", "hicdiffusion", "none")


def _no_strings() -> dict[str, str]:
    return {}


@dataclass
class CellType:
    name: str
    label: str
    lymphoblastoid: bool
    root: Path
    files: dict[str, str]
    anchors: dict[str, str] = field(default_factory=_no_strings)
    clusters: dict[str, str] = field(default_factory=_no_strings)
    note: str = ""
    description: str = ""
    hic_url: str = ""
    hic_name: str = ""
    contact_map: str = ""
    data: list[str] = field(default_factory=list)

    def path(self, key: str) -> Path:
        """The absolute path of one of the cell type's single files, by catalog key."""
        return self.root / self.files[key]

    def anchors_for(self, factors: str) -> Path:
        """The anchors file of a factor choice. RNAPII alone reads the CTCF and RNAPII anchors,
        which hold the RNAPII loop ends."""
        return self.root / self.anchors["CTCF+RNAPII" if factors == "RNAPII" else factors]

    def clusters_for(self, factors: str) -> list[tuple[str, Path]]:
        """The loop files a factor choice reads, CTCF first."""
        names = {"CTCF": ["CTCF"], "RNAPII": ["RNAPII"]}.get(factors, ["CTCF", "RNAPII"])
        return [(n, self.root / self.clusters[n]) for n in names]

    def genes(self) -> pd.DataFrame:
        """Columns gene_id, name, chrom, tss, strand."""
        return pd.read_csv(self.path("genes"), sep="\t")


SAMPLE_COLUMNS = ("sample", "sex", "population", "superpopulation", "father", "mother")


@dataclass
class SVSet:
    name: str
    label: str
    vcf: Path
    samples: list[str]
    samples_table: Path | None = None

    def sample_rows(self) -> list[list[str]]:
        """The table's rows of the set's samples, in the order of `SAMPLE_COLUMNS`.

        A sample the table lacks gets empty fields. A parent outside the set is left out, so
        every parent named is a sample a job can be made for.
        """
        if self.samples_table is None:
            return []
        table = pd.read_csv(self.samples_table, sep="\t", dtype=str, keep_default_na=False).set_index("sample")
        table = table.reindex(columns=list(SAMPLE_COLUMNS[1:]), fill_value="")
        known = set(self.samples)
        rows: list[list[str]] = []
        for name in self.samples:
            fields = [str(v) for v in table.loc[name]] if name in table.index else [""] * 5
            fields[3:] = [parent if parent in known else "" for parent in fields[3:]]
            rows.append([name, *fields])
        return rows


@dataclass
class Catalog:
    root: Path
    cell_types: dict[str, CellType]
    sv_sets: dict[str, SVSet]

    @classmethod
    def load(cls, root: Path) -> Catalog:
        root = Path(root)
        raw = json.loads((root / "catalog.json").read_text())
        cts: dict[str, CellType] = {}
        for name, c in raw.get("cell_types", {}).items():
            singles = {k: c[k] for k in ("singletons", "segments", "centromeres", "config", "genes", "enhancers")}
            cts[name] = CellType(
                name=name,
                label=c.get("label", name),
                lymphoblastoid=bool(c.get("lymphoblastoid", False)),
                root=root / c["dir"],
                files=singles,
                anchors=dict(c["anchors"]),
                clusters=dict(c["clusters"]),
                note=c.get("note", ""),
                hic_url=c.get("hic_url", ""),
                hic_name=c.get("hic_name", ""),
                contact_map=c.get("contact_map", ""),
                description=c.get("description", c.get("label", name).partition(", ")[2]),
                data=list(c.get("data", [*c["clusters"], "Hi-C", *(["enhancers"] if c.get("enhancers") else [])])),
            )
        svs = {
            name: SVSet(
                name=name, label=s.get("label", name), vcf=root / s["vcf"], samples=list(s["samples"]),
                samples_table=root / s["samples_table"] if s.get("samples_table") else None,
            )
            for name, s in raw.get("sv_sets", {}).items()
        }
        return cls(root=root, cell_types=cts, sv_sets=svs)
