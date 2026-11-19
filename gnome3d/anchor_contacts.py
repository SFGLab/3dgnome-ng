"""Contact counts between anchor pairs, read from a full depth map at fine resolution.

The table holds, per chromosome, the anchors of the anchor file and the summed counts between
each pair. A pixel is shared between the anchors it overlaps in proportion to the overlap.
`slurm/ensemble/prep_anchor_contacts.py` writes the table from an mcool. A run matches its
anchors to the table's by interval. The table must come from the run's anchor file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import scipy.sparse as sp

from gnome3d.types import F32Array, F64Array, I32Array, I64Array


@dataclass(frozen=True)
class AnchorContacts:
    """The table of one chromosome.

    `starts` and `ends` are its anchors, sorted by start. `i`, `j` and `count` are the pairs with
    a count, `i < j`.
    """

    starts: I64Array
    ends: I64Array
    i: I32Array
    j: I32Array
    count: F32Array


AnchorContactMap = dict[str, AnchorContacts]


def overlap_weights(
    bin_starts: I64Array, res: int, starts: I64Array, ends: I64Array
) -> sp.csr_matrix:
    """Returns the bins by anchors matrix of the share of each bin inside each anchor.

    Parameters
    ----------
    bin_starts
        Left edges of a chromosome's bins, ascending, `res` apart.
    res
        Bin width in bp.
    starts, ends
        Anchor intervals, half open.
    """
    first = np.searchsorted(bin_starts, starts, side="right") - 1
    last = np.searchsorted(bin_starts, ends, side="left")
    span = np.maximum(last - np.maximum(first, 0), 0)
    rows = np.repeat(np.arange(len(starts)), span)
    offs = np.arange(int(span.sum())) - np.repeat(np.cumsum(span) - span, span)
    b = np.repeat(np.maximum(first, 0), span) + offs
    lo = np.maximum(starts[rows], bin_starts[b])
    hi = np.minimum(ends[rows], bin_starts[b] + res)
    ov = (hi - lo).astype(np.float64) / res
    keep = ov > 0
    return sp.csr_matrix((ov[keep], (b[keep], rows[keep])), shape=(len(bin_starts), len(starts)))


def aggregate_pixels(
    w: sp.csr_matrix, bin1: I64Array, bin2: I64Array, count: F64Array
) -> sp.coo_matrix:
    """Sums the pixels `(bin1, bin2, count)` of an upper triangle map into anchor pair counts.

    `w` is from `overlap_weights`. Returns the strict upper triangle. A pixel inside one anchor
    and a pixel on the diagonal count for no pair.
    """
    nb, na = int(w.shape[0]), int(w.shape[1])  # pyright: ignore[reportOptionalSubscript]
    off = bin1 != bin2
    m = sp.coo_matrix((count[off], (bin1[off], bin2[off])), shape=(nb, nb)).tocsr()
    m = m + m.T
    c = (w.T @ m @ w).tocoo()
    up = c.row < c.col
    return sp.coo_matrix((c.data[up], (c.row[up], c.col[up])), shape=(na, na))


def save(path: Path, tables: AnchorContactMap, res: int, source: str) -> None:
    """Writes the tables as one npz, five arrays per chromosome."""
    out: dict[str, np.ndarray] = {"res": np.array(res), "source": np.array(source)}
    for chrom, t in tables.items():
        for key in ("starts", "ends", "i", "j", "count"):
            out[f"{chrom}/{key}"] = getattr(t, key)
    cast(Any, np.savez_compressed)(path, **out)


def load(path: str, chr_set: set[str]) -> AnchorContactMap:
    """Loads the tables of the chromosomes in `chr_set`."""
    out: AnchorContactMap = {}
    with np.load(path) as z:
        for chrom in sorted({k.split("/")[0] for k in z.files if "/" in k} & chr_set):
            out[chrom] = AnchorContacts(
                starts=z[f"{chrom}/starts"].astype(np.int64),
                ends=z[f"{chrom}/ends"].astype(np.int64),
                i=z[f"{chrom}/i"].astype(np.int32),
                j=z[f"{chrom}/j"].astype(np.int32),
                count=z[f"{chrom}/count"].astype(np.float32),
            )
    return out


def match(table: AnchorContacts, starts: I64Array, ends: I64Array) -> I64Array:
    """Returns the table index of each given anchor, -1 for an anchor the table lacks."""
    k = np.searchsorted(table.starts, starts, side="left")
    idx = np.full(len(starts), -1, dtype=np.int64)
    n = len(table.starts)
    # Several table anchors may share a start; walk forward to the one with the same end.
    for a in range(len(starts)):
        p = int(k[a])
        while p < n and table.starts[p] == starts[a]:
            if table.ends[p] == ends[a]:
                idx[a] = p
                break
            p += 1
    return idx


def dense_counts(table: AnchorContacts, starts: I64Array, ends: I64Array) -> F64Array:
    """Returns the symmetric counts between the given anchors.

    The diagonal and any anchor the table lacks are zero.
    """
    idx = match(table, starts, ends)
    n = len(starts)
    pos = np.full(len(table.starts), -1, dtype=np.int64)
    pos[idx[idx >= 0]] = np.flatnonzero(idx >= 0)
    a, b = pos[table.i], pos[table.j]
    keep = (a >= 0) & (b >= 0)
    out = np.zeros((n, n), dtype=np.float64)
    np.add.at(out, (a[keep], b[keep]), table.count[keep].astype(np.float64))
    return out + out.T
