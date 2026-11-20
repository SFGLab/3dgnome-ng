"""The Hi-C cells of the pairs of nodes of a uniform chain, `design/uniform-chain.md`.

A pair of nodes reads the cell of `[data] contact_map` that holds its two beads, at
`[springs] contact_resolution`. The counts are raw. The expectation of a cell is the mean count at
its separation over the stretch of the chromosome that holds the nodes. A pair without a loop is
held at the distance of the law for its cell when the cell passes the Poisson test of
`contact_background_z`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from gnome3d import log

if TYPE_CHECKING:
    from gnome3d.polymer import PolymerLaw
    from gnome3d.settings import Settings
    from gnome3d.types import BoolArray, F64Array, I64Array

LOG = log.get("cells")
_ROWS = 512


def cell_uri(uri: str, resolution: int) -> str:
    """Returns the cooler URI of `uri` at `resolution`. A multi resolution file is addressed
    by `::/resolutions/<bp>`. A file of one resolution is returned as it is.
    """
    path, _, _group = uri.partition("::")
    if path.endswith(".mcool"):
        return f"{path}::/resolutions/{int(resolution)}"
    return uri


def expected_by_offset(counts: Any, valid: BoolArray) -> F64Array:
    """Returns the mean count of a cell at each offset from the diagonal, in bins.

    `counts` is a square sparse matrix, symmetric, of raw counts. A cell counts when both of its
    bins are valid. The mean is taken over eight bins of offset a decade. Offset 0 is 0.
    """
    n = int(valid.shape[0])
    coo = counts.tocoo()
    d = coo.col - coo.row
    keep = (d >= 1) & valid[coo.row] & valid[coo.col]
    sums = np.bincount(d[keep], coo.data[keep].astype(np.float64), minlength=n)[:n]
    v = valid.astype(np.float64)
    spectrum = np.fft.rfft(v, 2 * n)
    cells = np.rint(np.fft.irfft(spectrum * np.conj(spectrum), 2 * n)[:n])
    offset = np.arange(n)
    key = np.zeros(n, dtype=np.int64)
    key[1:] = np.floor(np.log10(offset[1:]) * 8.0).astype(np.int64) + 1
    num = np.bincount(key, sums, minlength=int(key.max()) + 1)
    den = np.bincount(key, cells, minlength=int(key.max()) + 1)
    mean = np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)
    out: F64Array = mean[key]
    out[0] = 0.0
    return out


def read_cells(
    uri: str, chrom: str, mids: I64Array, resolution: int
) -> tuple[F64Array, I64Array, F64Array, BoolArray]:
    """Reads the cells of the pairs of nodes from a cooler.

    Returns `(count, offset, expected, mapped)`: the raw count of the cell of each pair of nodes,
    the offset of that cell from the diagonal in bins, the mean count by offset over the stretch
    that holds the nodes, and the nodes whose bin has a count.
    """
    from scipy import sparse  # noqa: PLC0415

    from gnome3d.pipeline.ib.maxent import COOLER_LOCK, open_cooler  # noqa: PLC0415

    try:
        clr = open_cooler(cell_uri(uri, resolution))
    except (KeyError, OSError) as err:
        raise ValueError(
            f"[data] contact_map {uri!r} has no map at [springs] contact_resolution"
            f" {int(resolution)}. Give an mcool that holds that resolution or set"
            " contact_resolution to one the file has."
        ) from err
    res = int(clr.binsize)
    if res != int(resolution):
        raise ValueError(
            f"[data] contact_map has bins of {res} bp and [springs] contact_resolution is"
            f" {int(resolution)}. Give an mcool that holds that resolution or set"
            " contact_resolution to the bin of the file."
        )
    first = int(mids.min()) // res
    last = int(mids.max()) // res
    with COOLER_LOCK:
        matrix = clr.matrix(balance=False, sparse=True).fetch(
            f"{chrom}:{first * res}-{(last + 1) * res}"
        )
    upper: Any = sparse.triu(matrix, 1)  # pyright: ignore[reportUnknownMemberType]
    counts: Any = (upper + upper.T).tocsr()
    valid: BoolArray = np.asarray(counts.sum(axis=1)).ravel() > 0
    bins = (mids // res - first).astype(np.int64)
    count: F64Array = np.asarray(counts[bins][:, bins].todense(), dtype=np.float64)
    offset: I64Array = np.abs(bins[:, None] - bins[None, :])
    return count, offset, expected_by_offset(counts, valid), valid[bins]


def hold_by_cells(
    mat: F64Array,
    mids: I64Array,
    count: F64Array,
    offset: I64Array,
    expected: F64Array,
    mapped: BoolArray,
    law: PolymerLaw,
    s: Settings,
    tested: BoolArray | None = None,
    resolution: int = 0,
) -> F64Array:
    """Returns the target matrix with the pairs that their cells hold.

    A pair without a loop whose cell has an expectation above z squared and a count more than z
    Poisson deviations from it is held at the background times
    `((count + 0.5) / (expected + 0.5)) ^ (-1/3)`, clipped to one bead and to
    `distance_heatmap_stretching` backgrounds. It is entered as minus its distance, which the
    kernels score with the background spring. Two nodes in one cell and a node in a bin without
    a count are left as they are.

    `tested` marks the pairs whose cell had an expectation above z squared at a finer resolution.
    Such a pair is left as it is and the pairs that this resolution can test are added to the
    marks. A pair is then read at the finest resolution that can test it.
    """
    z_cut = float(s.contact_background_z)
    s0 = max(int(law.s0_bp), 1)
    stretch = float(s.heatmap_distance_stretching)
    pos = mids.astype(np.float64)
    m = int(pos.shape[0])
    out: F64Array = np.array(mat, dtype=np.float64, copy=True)
    held_in = held_out = 0
    for r0 in range(0, m, _ROWS):
        r1 = min(m, r0 + _ROWS)
        ii, jj = np.nonzero(np.arange(m)[None, :] > np.arange(r0, r1)[:, None])
        ii = ii + r0
        off = offset[ii, jj]
        exp = expected[np.minimum(off, expected.shape[0] - 1)]
        obs = count[ii, jj]
        z = (obs - exp) / np.sqrt(np.maximum(exp, 1e-300))
        testable = (
            (mat[ii, jj] < 0.0) & (off >= 1) & mapped[ii] & mapped[jj] & (exp > z_cut * z_cut)
        )
        if tested is not None:
            testable &= ~tested[ii, jj]
            tested[ii[testable], jj[testable]] = True
        sel = testable & (np.abs(z) > z_cut)
        if not sel.any():
            continue
        sep = np.abs(pos[jj[sel]] - pos[ii[sel]])
        bg = np.maximum(1.0, (np.maximum(sep, 1.0) / s0) ** law.nu)
        ratio = (obs[sel] + 0.5) / (exp[sel] + 0.5)
        target = np.clip(bg * ratio ** (-1.0 / 3.0), 1.0, stretch * bg)
        out[ii[sel], jj[sel]] = -target
        out[jj[sel], ii[sel]] = -target
        held_in += int((z[sel] > 0).sum())
        held_out += int((z[sel] < 0).sum())
    log.status(
        LOG,
        "contact cells at %d bp: %s pairs of nodes held in, %s held out, of %s at z %.1f",
        int(resolution),
        f"{held_in:,}",
        f"{held_out:,}",
        f"{m * (m - 1) // 2:,}",
        z_cut,
    )
    return out
