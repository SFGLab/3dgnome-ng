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

from ..expression import KINDS as EXPRESSION_KINDS
from ..expression import Source

FACTOR_CHOICES = ("CTCF", "RNAPII", "CTCF+RNAPII")
BACKGROUNDS = ("hic", "hicdiffusion", "none")
# The chain of a model: loop anchors with the chromatin between them, or one bead per kilobase.
CHAINS = ("anchors", "uniform")


@lru_cache(maxsize=16)
def _columns(path: Path) -> frozenset[str]:
    """Returns the column names of a hosted table, read from its first line."""
    with open(path) as fh:
        return frozenset(re.split(r"[\t,]", fh.readline().rstrip("\n")))


@lru_cache(maxsize=None)
def gene_table(path: Path) -> pd.DataFrame:
    """Returns the gene table of a cell type, read once for the process. A caller takes rows of it
    and leaves the table as it is."""
    return pd.read_csv(path, sep="\t")


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
        """Returns the absolute path of one of the single files of the cell type, by its key in the catalog."""
        return self.root / self.files[key]

    def anchors_for(self, factors: str) -> Path:
        """Returns the anchors file of a factor choice. RNAPII alone reads the CTCF and RNAPII
        anchors, which hold the RNAPII loop ends."""
        return self.root / self.anchors["CTCF+RNAPII" if factors == "RNAPII" else factors]

    def clusters_for(self, factors: str) -> list[tuple[str, Path]]:
        """Returns the loop files a factor choice reads, CTCF first."""
        names = {"CTCF": ["CTCF"], "RNAPII": ["RNAPII"]}.get(factors, ["CTCF", "RNAPII"])
        return [(n, self.root / self.clusters[n]) for n in names]

    def genes(self) -> pd.DataFrame:
        """Returns the gene table with the columns gene_id, name, chrom, tss and strand."""
        return gene_table(self.path("genes"))


SAMPLE_COLUMNS = ("sample", "sex", "population", "superpopulation", "father", "mother")


@lru_cache(maxsize=None)
def _sample_rows(path: Path, samples: tuple[str, ...]) -> list[list[str]]:
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False).set_index("sample")
    table = table.reindex(columns=list(SAMPLE_COLUMNS[1:]), fill_value="")
    known = set(samples)
    rows: list[list[str]] = []
    for name in samples:
        fields = [str(v) for v in table.loc[name]] if name in table.index else [""] * 5
        fields[3:] = [parent if parent in known else "" for parent in fields[3:]]
        rows.append([name, *fields])
    return rows


@dataclass
class SVSet:
    name: str
    label: str
    vcf: Path
    samples: list[str]
    samples_table: Path | None = None

    def sample_rows(self) -> list[list[str]]:
        """Returns the rows of the samples of the set in the order of `SAMPLE_COLUMNS`, read once. A
        sample the table lacks gets empty fields and a parent outside the set is left out.
        """
        return _sample_rows(self.samples_table, tuple(self.samples)) if self.samples_table else []


def _no_expression() -> dict[str, Source]:
    return {}


@dataclass
class Catalog:
    cell_types: dict[str, CellType]
    sv_sets: dict[str, SVSet]
    expression: dict[str, Source] = field(default_factory=_no_expression)

    def sample_note(self, sv_set: str, sample: str) -> str:
        """Returns the population, the superpopulation and the sex of a hosted sample, or an empty string."""
        rows = self.sv_sets[sv_set].sample_rows() if sv_set in self.sv_sets else []
        row = next((r for r in rows if r[0] == sample), None)
        return ", ".join(v for v in (row[2], row[3], row[1]) if v) if row else ""

    def expression_options(self, cell: str, sample: str = "", hidden: tuple[str, ...] = ()) -> list[dict[str, object]]:
        """Returns the hosted expression a job of a cell type can take. It is the replicates of the
        cell type and, with a hosted sample, the libraries of that person. Each option has the id of
        its source, a label, the kind, the unit, the columns and `other_cell_type`, true for a
        person's lymphoblastoid line beside a cell type that is not lymphoblastoid. A source of a
        feature in `hidden` is left out."""
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
        # A cell type with replicates in a table that no feature hides lists RNA-seq among its data.
        for src in tables.values():
            for cell in src.cell_types if not src.feature else ():
                if cell in cts and "RNA-seq" not in cts[cell].data:
                    cts[cell].data.append("RNA-seq")
        return cls(cell_types=cts, sv_sets=svs, expression=tables)
