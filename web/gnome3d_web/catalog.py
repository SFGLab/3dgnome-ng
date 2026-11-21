"""The hosted data, described by `catalog.json` at the root of the data directory.

A cell type is a directory of the files that one modelling run reads, plus the genes and
enhancers that the result pages mark. `contact_map` is a cooler URI at 1 kb for the stretches
without loops. An SV set is one bgzipped VCF and its samples, with an optional
`samples_table`. Paths in the catalog are relative to the data directory.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .expression import KINDS as EXPRESSION_KINDS
from .expression import Source

FACTOR_CHOICES = ("CTCF", "RNAPII", "CTCF+RNAPII")
BACKGROUNDS = ("hic", "hicdiffusion", "none")


@lru_cache(maxsize=16)
def _columns(path: Path) -> frozenset[str]:
    """The column names of a hosted table, read from its first line."""
    with open(path) as fh:
        return frozenset(re.split(r"[\t,]", fh.readline().rstrip("\n")))


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
        """The rows of the set's samples in the order of `SAMPLE_COLUMNS`. A sample the table
        lacks gets empty fields and a parent outside the set is left out.
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


def _no_expression() -> dict[str, Source]:
    return {}


@dataclass
class Catalog:
    root: Path
    cell_types: dict[str, CellType]
    sv_sets: dict[str, SVSet]
    expression: dict[str, Source] = field(default_factory=_no_expression)

    def expression_options(self, cell: str, sample: str = "", hidden: tuple[str, ...] = ()) -> list[dict[str, object]]:
        """The hosted expression a job of a cell type can take: the replicates of the cell type,
        and with a hosted sample the libraries of that person. Each option has the id of its
        source, a label, the kind, the unit, the columns and `other_cell_type`, true for a
        person's lymphoblastoid line beside a cell type that is not lymphoblastoid. A source of
        a feature in `hidden` is left out."""
        ct = self.cell_types.get(cell)
        out: list[dict[str, object]] = []
        for src in self.expression.values():
            if src.feature in hidden:
                continue
            columns = src.cell_types.get(cell, [])
            who = cell
            if not columns and src.people and sample and sample in _columns(src.path):
                columns, who = [sample], sample
            if not columns:
                continue
            n = len(columns)
            count = f"{n} replicates" if n > 1 and who == cell else f"{n} librar{'y' if n == 1 else 'ies'}"
            out.append(
                {
                    "id": src.name,
                    "label": f"{src.label} of {who}, {count}",
                    "kind": src.kind,
                    "unit": src.unit,
                    "columns": columns,
                    "other_cell_type": bool(src.lymphoblastoid and who != cell and ct is not None and not ct.lymphoblastoid),
                }
            )
        return out

    @classmethod
    def load(cls, root: Path) -> Catalog:
        root = Path(root)
        raw = json.loads((root / "catalog.json").read_text())
        cts: dict[str, CellType] = {}
        for name, c in raw.get("cell_types", {}).items():
            singles = {k: c[k] for k in ("singletons", "segments", "centromeres", "config", "genes", "enhancers")}
            # A and B compartments as chr, start, end and label, for the compartment term.
            if c.get("compartments"):
                singles["compartments"] = c["compartments"]
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
        tables: dict[str, Source] = {}
        for name, e in raw.get("expression", {}).items():
            if e.get("kind") not in EXPRESSION_KINDS:
                raise ValueError(f"catalog.json: the expression source {name} needs a kind, counts or values")
            tables[name] = Source(
                name=name, label=e.get("label", name), path=root / e["file"], kind=e["kind"], unit=e.get("unit", ""),
                cell_types={k: list(v) for k, v in e.get("cell_types", {}).items()},
                people=bool(e.get("people", False)), lymphoblastoid=bool(e.get("lymphoblastoid", False)), feature=str(e.get("feature", "")),
            )  # fmt: skip
        return cls(root=root, cell_types=cts, sv_sets=svs, expression=tables)
