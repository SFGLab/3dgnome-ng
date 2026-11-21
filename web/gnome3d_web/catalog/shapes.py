"""The shapes of the answers of the form's routes."""

from __future__ import annotations

from typing import Literal

from typing_extensions import NotRequired, TypedDict


class CellTypeInfo(TypedDict):
    label: str
    description: str
    data: list[str]
    lymphoblastoid: bool
    note: str


class CallSetInfo(TypedDict):
    label: str
    samples: list[str]
    sample_table: bool


class CatalogInfo(TypedDict):
    cell_types: dict[str, CellTypeInfo]
    sv_sets: dict[str, CallSetInfo]
    factors: list[str]
    max_region_mb: float
    hicdiffusion_mb: float
    max_structures: int
    hicdiffusion: bool


class GeneHit(TypedDict):
    name: str
    chrom: str
    tss: int


class RegionInfo(TypedDict):
    chrom: str
    start: int
    end: int
    label: str
    gene: NotRequired[str]


class NamedGene(TypedDict):
    name: str
    tss: int


class Preview(TypedDict):
    # A loop is its two anchor middles, its PET count and the outer edges of its anchors.
    loops: dict[str, list[tuple[int, int, int, int, int]]]
    counts: dict[str, int]
    genes: list[NamedGene]


class SampleTable(TypedDict):
    columns: list[str]
    rows: list[list[str]]


class ExpressionOption(TypedDict):
    id: str
    label: str
    kind: Literal["counts", "values"]
    unit: str
    columns: list[str]
    other_cell_type: bool


class ExpressionOptions(TypedDict):
    options: list[ExpressionOption]
