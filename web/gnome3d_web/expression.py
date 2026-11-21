"""Gene expression beside a job. The module has a table of genes by samples, the value of a gene in
a job and the contrast of two jobs.

A table holds read counts or values that are normalised already, TMM or RPKM. Counts of two jobs
with two or more samples each are compared with DESeq2, by PyDESeq2. Any other pair of tables of one
kind gives a log2 ratio and no p-value.

A table is tab or comma separated. Its first column has the gene, by Ensembl id or by name. Every
other numeric column is a sample. A column `length` gives the length of the gene in bp. The output
of featureCounts is read as it is.
"""

# pandas states the type of a column and of a selection too loosely for these checks.
# pyright: reportCallIssue=false, reportArgumentType=false, reportIndexIssue=false, reportAttributeAccessIssue=false, reportReturnType=false

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from typing_extensions import NotRequired, TypedDict

import numpy as np
import pandas as pd

# The source of a table that came with the request.
UPLOAD = "upload"
KINDS = ("counts", "values")
TABLE = "expression.tsv"
META = "expression.json"
# The columns of featureCounts that are not samples.
_PLACE = {"chr", "start", "end", "strand"}
_ENSEMBL = re.compile(r"^ENS[A-Z]*G\d+")


class ExpressionError(ValueError):
    """An expression table or a request for one that the site cannot use, with the reason."""


def _no_groups() -> dict[str, list[str]]:
    return {}


@dataclass
class Source:
    """A hosted table. `cell_types` gives the columns of each cell type it holds. With `people`
    every other column is a person, under the name the call sets give their samples."""

    name: str
    label: str
    path: Path
    kind: str
    unit: str = ""
    cell_types: dict[str, list[str]] = field(default_factory=_no_groups)
    people: bool = False
    # The samples are lymphoblastoid cell lines. Another cell type gets a warning with them.
    lymphoblastoid: bool = False
    # The feature of config.json the table belongs to. The table is not offered while it is off.
    feature: str = ""


@dataclass
class Table:
    """Genes by samples. `key` is gene_id for Ensembl ids, without their version, or name."""

    key: str
    data: pd.DataFrame
    length: pd.Series | None = None


def read_table(path: Path, samples: list[str] | None = None) -> Table:
    """Reads a table of genes by samples. With `samples` it reads those columns alone, beside the
    genes and their lengths."""
    try:
        head = pd.read_csv(path, sep="\t", comment="#", nrows=0)
        sep = "\t" if head.shape[1] >= 2 else ","
        names = list(pd.read_csv(path, sep=sep, comment="#", nrows=0).columns)
        keep = None if samples is None else [c for i, c in enumerate(names) if i == 0 or str(c).lower() == "length" or str(c) in samples]
        t = pd.read_csv(path, sep=sep, comment="#", usecols=keep)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        raise ExpressionError(f"{Path(path).name} is not a table of genes by samples: {e}") from e
    if t.shape[1] < 2 or t.empty:
        raise ExpressionError(f"{Path(path).name} needs a column of genes and one column or more of samples")
    genes = t.iloc[:, 0].astype(str)
    key = "gene_id" if _ENSEMBL.match(genes.iloc[0]) else "name"
    if key == "gene_id":
        genes = genes.str.replace(r"\.\d+$", "", regex=True)
    rest = t.iloc[:, 1:]
    rest = rest[[c for c in rest.columns if str(c).lower() not in _PLACE]]
    length = None
    for c in rest.columns:
        if str(c).lower() == "length":
            length = pd.to_numeric(rest[c], errors="coerce")
            rest = rest.drop(columns=c)
            break
    data = rest.select_dtypes(include="number").dropna(axis=1, how="all")
    if data.shape[1] == 0:
        raise ExpressionError(f"{Path(path).name} has no column of numbers after its gene column")
    data = data.set_axis([str(c) for c in data.columns], axis=1)
    data.index = pd.Index(genes, name=key)
    first = ~data.index.duplicated()
    if length is not None:
        length = pd.Series(length.to_numpy()[first], index=data.index[first])
    return Table(key, data[first], length)


def check_counts(table: Table, columns: list[str], name: str) -> None:
    """Raises when the columns are not read counts, which are whole numbers and none below zero."""
    part = table.data[columns]
    if (part < 0).any().any() or not np.allclose(part.to_numpy(), np.round(part.to_numpy())):
        raise ExpressionError(
            f"{name} holds values that are not whole numbers. Read counts are whole numbers. Upload the counts, or state that the table is normalised."
        )


def write_job(job: Path, table: Table, columns: list[str], meta: dict[str, object]) -> None:
    """Writes the samples of a job with every gene of the table and the record of what they are."""
    missing = [c for c in columns if c not in table.data.columns]
    if missing:
        raise ExpressionError(f"the expression table has no column {', '.join(missing)}")
    out = table.data[columns].copy()
    if table.length is not None:
        out.insert(0, "length", table.length)
    out.to_csv(job / TABLE, sep="\t")
    (job / META).write_text(json.dumps({**meta, "key": table.key, "columns": columns}, indent=1))


class Meta(TypedDict):
    """The record of the expression of a job, expression.json."""

    source: NotRequired[str]
    label: NotRequired[str]
    kind: Literal["counts", "values"]
    unit: NotRequired[str]
    key: str
    columns: list[str]


def job_meta(job: Path) -> Meta | None:
    """Returns the record of the expression of a job, None for a job without it."""
    p = job / META
    return json.loads(p.read_text()) if p.is_file() and (job / TABLE).is_file() else None


def _job_table(job: Path) -> Table:
    t = pd.read_csv(job / TABLE, sep="\t", index_col=0)
    t.index = t.index.astype(str)
    length = t.pop("length") if "length" in t.columns else None
    return Table(str(t.index.name), t, length)


def level(table: Table, kind: str) -> tuple[pd.Series, str]:
    """Returns the level of each gene over the samples and its unit. The unit is TPM for counts with
    gene lengths, counts per million for counts without and the mean for values."""
    if kind != "counts":
        return table.data.mean(axis=1), ""
    if table.length is not None and (table.length > 0).all():
        rate = table.data.div(table.length, axis=0)
        return (rate / rate.sum(axis=0) * 1e6).mean(axis=1), "TPM"
    return (table.data / table.data.sum(axis=0) * 1e6).mean(axis=1), "CPM"


def job_levels(job: Path) -> tuple[dict[str, float], str, str] | None:
    """Returns the level of every gene of a job, the unit and the key of the genes."""
    meta = job_meta(job)
    if meta is None:
        return None
    table = _job_table(job)
    values, unit = level(table, meta["kind"])
    return {str(k): float(v) for k, v in values.items() if np.isfinite(v)}, unit or str(meta.get("unit", "")), table.key


def size_factors(counts: pd.DataFrame) -> pd.Series:
    """Returns the size factor of each sample, the median over genes of its count over the geometric
    mean of the gene, as DESeq2 computes it. Genes with a zero count in a sample are left out."""
    logs = np.log(counts.where(counts > 0))
    ref = logs.mean(axis=1, skipna=False)
    ok = np.isfinite(ref)
    if not ok.any():
        return pd.Series(1.0, index=counts.columns)
    return np.exp((logs[ok].sub(ref[ok], axis=0)).median(axis=0))


def _ratio(a: pd.DataFrame, b: pd.DataFrame, counts: bool) -> pd.DataFrame:
    """Returns log2 of the level in b over the level in a, with one added to both for counts and the
    smallest positive value for values."""
    if counts:
        both = pd.concat([a, b], axis=1)
        both = both / size_factors(both)
        ma, mb = both[a.columns].mean(axis=1), both[b.columns].mean(axis=1)
        add = 1.0
    else:
        ma, mb = a.mean(axis=1), b.mean(axis=1)
        positive = pd.concat([ma, mb])
        add = float(positive[positive > 0].min()) if (positive > 0).any() else 1.0
    return pd.DataFrame({"log2fc": np.log2((mb + add) / (ma + add)), "padj": np.nan, "mean_a": ma, "mean_b": mb})


def _deseq(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Runs DESeq2 of b against a over the genes with ten reads or more in all samples together."""
    from pydeseq2.dds import DeseqDataSet  # noqa: PLC0415
    from pydeseq2.ds import DeseqStats  # noqa: PLC0415

    counts = pd.concat([a.add_prefix("a:"), b.add_prefix("b:")], axis=1).round().astype(int)
    counts = counts[counts.sum(axis=1) >= 10]
    groups = pd.DataFrame({"group": ["a"] * a.shape[1] + ["b"] * b.shape[1]}, index=counts.columns)
    dds = DeseqDataSet(counts=counts.T, metadata=groups, design="~group", quiet=True)
    dds.deseq2()
    stats = DeseqStats(dds, contrast=["group", "b", "a"], quiet=True)
    stats.summary()
    res = stats.results_df
    norm = counts / size_factors(counts)
    return pd.DataFrame(
        {"log2fc": res["log2FoldChange"], "padj": res["padj"], "mean_a": norm[[c for c in norm if c.startswith("a:")]].mean(axis=1), "mean_b": norm[[c for c in norm if c.startswith("b:")]].mean(axis=1)}
    )


def _by_name(table: Table, names: dict[str, str], counts: bool) -> Table:
    """Turns a table of Ensembl ids into a table of gene names. A gene without a name is left out.
    Two ids of one name are added for counts and averaged for values."""
    named = table.data.copy()
    named.index = pd.Index([names.get(str(g)) for g in named.index], name="name")
    named = named[named.index.notna()]
    grouped = named.groupby(level=0, sort=False)
    return Table("name", grouped.sum() if counts else grouped.mean(), None)


class Contrast(TypedDict):
    """The comparison of the expression of two jobs. `mode` is deseq2, ratio or none and `reason`
    explains a mode other than deseq2."""

    mode: Literal["deseq2", "ratio", "none"]
    reason: str
    label_a: NotRequired[str]
    label_b: NotRequired[str]
    key: NotRequired[str]
    samples_a: NotRequired[int]
    samples_b: NotRequired[int]


class Tested(Contrast):
    # For each gene the log2 fold change, the adjusted p or None, the level in a and in b.
    genes: NotRequired[dict[str, tuple[float, float | None, float, float]]]


class _Labels(TypedDict):
    label_a: str
    label_b: str
    key: str


def contrast(a: Path, b: Path, cache: Path, names: Callable[[], dict[str, str]] | None = None) -> Tested:
    """Compares the expression of job b against job a, gene by gene. Returns `mode`, deseq2, ratio
    or none, the `reason` of a mode other than deseq2 and `genes`, for each gene `[log2 fold change,
    adjusted p or None, level in a, level in b]`. The genes have the names of the tables, by `key`.
    The result of a pair of tables is kept in `cache`. One table by Ensembl id and one by name are
    compared by name, with `names`, the name of each id.
    """
    ma, mb = job_meta(a), job_meta(b)
    if ma is None or mb is None:
        return {"mode": "none", "reason": "One of the two jobs has no expression data."}
    base: _Labels = {"label_a": ma.get("label", ""), "label_b": mb.get("label", ""), "key": ma["key"]}
    if ma["kind"] != mb["kind"]:
        return {**base, "mode": "none", "reason": "One job has read counts and the other normalised values. The two cannot be compared."}
    mixed = ma["key"] != mb["key"]
    if mixed and names is None:
        return {**base, "mode": "none", "reason": "One table names its genes by Ensembl id and the other by name."}
    if mixed:
        base["key"] = "name"
    counts = ma["kind"] == "counts"
    if not counts and (ma.get("source"), ma.get("unit")) != (mb.get("source"), mb.get("unit")):
        return {**base, "mode": "none", "reason": "The normalised values of the two jobs come from two sources with units of their own."}
    digest = hashlib.sha256((a / TABLE).read_bytes() + b"|" + (b / TABLE).read_bytes() + (b"|by name" if mixed else b"")).hexdigest()[:24]
    kept = cache / f"{digest}.json"
    if kept.is_file():
        return cast(Tested, {**base, **json.loads(kept.read_text())})
    ta, tb = _job_table(a), _job_table(b)
    if mixed and names is not None:
        ta = _by_name(ta, names(), ma["kind"] == "counts") if ta.key == "gene_id" else ta
        tb = _by_name(tb, names(), mb["kind"] == "counts") if tb.key == "gene_id" else tb
    if list(ta.data.columns) == list(tb.data.columns) and ta.data.equals(tb.data):
        return {**base, "mode": "none", "reason": "The two jobs have the same samples. Their expression does not differ."}
    genes = ta.data.index.intersection(tb.data.index)
    da, db = ta.data.loc[genes], tb.data.loc[genes]
    reason = ""
    if counts and da.shape[1] >= 2 and db.shape[1] >= 2:
        try:
            table, mode = _deseq(da, db), "deseq2"
        except ImportError:
            table, mode = _ratio(da, db, True), "ratio"
            reason = "PyDESeq2 is not installed on this server. The fold change is a ratio of normalised counts with no test."
    else:
        table, mode = _ratio(da, db, counts), "ratio"
        reason = (
            "A test needs two samples or more on each side. The fold change is a ratio of normalised counts with no test."
            if counts
            else "Normalised values give a ratio and no test."
        )
    table = table[np.isfinite(table["log2fc"])]
    out = {
        "mode": mode,
        "reason": reason,
        "samples_a": int(da.shape[1]),
        "samples_b": int(db.shape[1]),
        "genes": {
            str(g): [round(float(r.log2fc), 4), None if not np.isfinite(r.padj) else float(f"{r.padj:.4g}"), round(float(r.mean_a), 3), round(float(r.mean_b), 3)]
            for g, r in zip(table.index, table.itertuples(index=False), strict=True)
        },
    }
    cache.mkdir(parents=True, exist_ok=True)
    kept.write_text(json.dumps(out))
    return cast(Tested, {**base, **out})
