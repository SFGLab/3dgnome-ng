"""The hosted data, described by `catalog.json` at the root of the data directory.

A cell type is a directory of the files one modelling run reads, the base loops per factor,
the anchors per factor choice, the base contact map as singletons, segments, centromeres and a
production config, plus the genes and enhancers the result pages mark. An SV set is one
bgzipped VCF and the samples it holds. Paths in the catalog are relative to the data directory.
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


@dataclass
class SVSet:
    name: str
    label: str
    vcf: Path
    samples: list[str]


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
                description=c.get("description", c.get("label", name).partition(", ")[2]),
                data=list(c.get("data", [*c["clusters"], "Hi-C", *(["enhancers"] if c.get("enhancers") else [])])),
            )
        svs = {
            name: SVSet(name=name, label=s.get("label", name), vcf=root / s["vcf"], samples=list(s["samples"]))
            for name, s in raw.get("sv_sets", {}).items()
        }
        return cls(root=root, cell_types=cts, sv_sets=svs)
