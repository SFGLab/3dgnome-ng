"""The shapes of the answers of a job page and of the comparison of two jobs."""

from __future__ import annotations

from typing import Literal

from typing_extensions import NotRequired, TypedDict

from ..catalog.shapes import NamedGene
from ..expression import Contrast
from ..jobs.shapes import FileSize, RequestStated, Stage, State


class ExpressionStated(TypedDict):
    label: str
    kind: Literal["counts", "values"]
    unit: str
    samples: int
    source: str


class Summary(TypedDict):
    state: State
    stage: str | None
    error: str | None
    log_tail: str | None
    # The number of jobs that wait before this one, None for a job that does not wait.
    ahead: int | None
    started: float | None
    finished: float | None
    region: str | None
    request: RequestStated
    loops: dict[str, int]
    n_svs: int | None
    # The variants of the sample in the region that have a type the site does not apply.
    skipped_svs: int | None
    structures: int
    genes: int
    loop_count: int
    files: list[FileSize]
    models: list[FileSize]
    expression: ExpressionStated | None
    stages: list[Stage]
    # The clock that times the stages. A page counts the time of the running stage from it.
    now: float
    structures_done: int
    structures_asked: int | None
    log_recent: str
    # Population, superpopulation and sex of a hosted sample, empty without them.
    sample_note: NotRequired[str]


class Promoter(TypedDict):
    bead: int
    genes: list[str]


class Marks(TypedDict):
    promoters: list[Promoter]
    enhancers: list[int]


class LoopEnd(TypedDict):
    factor: str
    other: tuple[int, int]
    pets: int


class AnchorBead(TypedDict):
    bead: int
    start: int
    end: int
    loops: list[LoopEnd]


class Structure(TypedDict):
    # A point of the smooth chain is x, y, z and its place along the beads.
    chain: list[tuple[float, float, float, float]]
    # A bead is its start, its end, x, y, z and its kind.
    beads: list[tuple[int, int, float, float, float, Literal["anchor", "subanchor"]]]
    marks: Marks
    anchors: list[AnchorBead]


class Loop(TypedDict):
    factor: str
    chrom: str
    a: tuple[int, int]
    b: tuple[int, int]
    span: int
    pets: int


class Annotations(TypedDict):
    genes: list[NamedGene]
    enhancers: list[tuple[int, int]]
    loops: list[Loop]


class Picks(TypedDict):
    medoid: int
    compact: int
    random: int


class End(TypedDict):
    bead: int
    bp: int
    offset: int


class Distance(TypedDict):
    a: End
    b: End
    values: list[float]
    mean: float
    sd: float
    n: int


class GeneRow(TypedDict):
    gene: str
    gene_id: str
    # The level of the gene in the expression of the job, None without it.
    expression: float | None
    tss: int
    enhancer: int
    linear_bp: int
    distance: float
    promoter_bead: int
    enhancer_bead: int
    min: float
    q1: float
    median: float
    q3: float
    max: float
    sd: float
    n: int
    joined_by: str
    promoter_loops: int


class ContactMaps(TypedDict):
    bins: list[int]
    # Each pair of bins once. Row i has the cells of the columns from i to the last.
    upper: list[list[float]]
    lower: list[list[float]]
    upper_label: str
    lower_label: str
    scc: float | None
    scc_about: str
    rank: float | None
    rank_reach: int
    rank_about: str
    loop_enrichment: float | None


class Spread(TypedDict):
    mean: float
    sd: float


class DistanceChange(TypedDict):
    gene: str
    linear_bp: int
    linear_bp_b: int
    enhancer_bead_a: int
    enhancer_bead_b: int
    a: Spread
    b: Spread
    ratio: float
    # None with one structure on a side.
    p: float | None


class DistanceChanges(TypedDict):
    same_region: bool
    rows: list[DistanceChange]
    differ: int
    shared: bool
    shared_enhancers: int | None


class GeneDistances(TypedDict):
    gene: str
    a: list[float]
    b: list[float]
    p: float | None


class ExpressionChange(TypedDict):
    gene: str
    distance: float
    log2fc: float
    padj: float | None
    a: float
    b: float


class Volcano(TypedDict):
    fold: float
    p: float
    points: list[tuple[float, float]]


class ExpressionChanges(Contrast):
    rows: list[ExpressionChange]
    # The rank correlation of the two changes over the rows, None under five rows.
    rho: float | None
    volcano: Volcano | None
