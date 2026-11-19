"""Helpers of the integration tests: a known structure, its input files and two distances to it.

The synthetic inputs follow the model validation of the 3D-GNOME paper of 2016, supplement
sections III and IV. Each bead of the known structure is one bin that holds a small loop of two
anchors and the singleton count of a pair of bins is the inverse of a power law of their
distance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from gnome3d.production import settings_for_cell
from gnome3d.settings import Settings
from gnome3d.types import BeadOut, F64Array, I64Array

CELL = "SYN"

# Every stage on the numba kernels in one thread. A run is then the same with and without JAX.
CPU = {
    "simulation_backend": {
        "mc_executor_arcs": "serial",
        "mc_executor_densify": "serial",
        "mc_executor_estimate_dist": "serial",
        "mc_executor_smooth": "serial",
    }
}


def synthetic_structure(n: int, seed: int, step: float = 1.0, ev: float = 0.75) -> F64Array:
    """Builds a random walk of `n` beads that avoids its last 15 beads within `ev`."""
    rng = np.random.default_rng(seed)
    pts = [np.zeros(3)]
    while len(pts) < n:
        d = rng.normal(size=3)
        for _ in range(300):
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            cand = pts[-1] + step * d
            if all(np.linalg.norm(cand - p) > ev for p in pts[-15:]):
                pts.append(cand)
                break
        else:
            pts.append(pts[-1] + step * d)
    return np.asarray(pts[:n])


def write_inputs(
    coords: F64Array,
    data_root: Path,
    chrom: str = "chr1",
    binsize: int = 1_000_000,
    start: int = 5_000_000,
    scale: float = 25.0,
    power: float = -0.6,
    k: float = 3000.0,
) -> tuple[str, I64Array]:
    """Writes the input files of a known structure under `data_root/SYN`.

    The singleton count of a pair of bins is `k * (d / scale) ** (1 / power)`, the inverse of
    the distance law `d = scale * count ** power`. Returns the region and the bin edges.
    """
    d = data_root / CELL
    d.mkdir(parents=True, exist_ok=True)
    n = len(coords)
    mid = start + np.arange(n) * binsize + binsize // 2
    with (
        open(d / f"{CELL}_anchors_3+_oriented.bed", "w") as fa,
        open(d / f"{CELL}_clusters_3+.bedpe", "w") as fc,
    ):
        for m in mid:
            fa.write(f"{chrom}\t{m - 3000}\t{m - 1000}\tN\n")
            fa.write(f"{chrom}\t{m + 1000}\t{m + 3000}\tN\n")
            fc.write(f"{chrom}\t{m - 3000}\t{m - 1000}\t{chrom}\t{m + 1000}\t{m + 3000}\t50\n")
    (d / f"ccds_all_hg38_merged100k_{CELL}.breakpoints.bed").write_text("")
    (d / "hg38_centromeres.bed").write_text("")
    iu = np.triu_indices(n, 1)
    counts = k * (pdist(coords) / scale) ** (1.0 / power)
    rows = [
        f"{chrom}\t{mid[i]}\t{mid[i] + 1}\t{chrom}\t{mid[j]}\t{mid[j] + 1}\t{int(round(c))}"
        for i, j, c in zip(*iu, counts, strict=True)
        if round(c) >= 1
    ]
    (d / f"{CELL}_singletons_lessthan3.bedpe").write_text("\n".join(rows) + "\n")
    edges = start + np.arange(n + 1) * binsize
    return f"{chrom}:{start}-{start + n * binsize}", edges


def cpu_settings(cell: str, data_root: Path | str, quality: str = "fast") -> Settings:
    """Returns the production settings of a cell line with every stage on one numba thread."""
    return settings_for_cell(cell, str(data_root), quality, overrides=CPU)


def arrays(beads: list[BeadOut]) -> tuple[F64Array, I64Array]:
    """Returns the coordinates and the genomic midpoints of a bead list."""
    coords = np.array([(b.x, b.y, b.z) for b in beads], dtype=np.float64)
    mids = np.array([(b.start + b.end) // 2 for b in beads], dtype=np.int64)
    return coords, mids


def bin_centroids(beads: list[BeadOut], edges: I64Array) -> F64Array:
    """Averages the beads of each bin. A bin without a bead is a row of NaN."""
    coords, mids = arrays(beads)
    idx = np.searchsorted(edges, mids, side="right") - 1
    out = np.full((len(edges) - 1, 3), np.nan)
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.any():
            out[b] = coords[sel].mean(axis=0)
    return out


def scale_to(truth: F64Array, x: F64Array) -> F64Array:
    """Scales `x` to the median pair distance of `truth`. The model's unit of length is its own."""
    return x * (np.median(pdist(truth)) / np.median(pdist(x)))


def _kabsch(a: F64Array, b: F64Array) -> float:
    ac = a - a.mean(axis=0)
    bc = b - b.mean(axis=0)
    u, _s, vt = np.linalg.svd(bc.T @ ac)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return float(np.sqrt(((ac - bc @ rot.T) ** 2).sum(axis=1).mean()))


def rmsd(truth: F64Array, x: F64Array) -> float:
    """Returns the RMSD after the best superposition, the smaller of the two mirror images."""
    return min(_kabsch(truth, x), _kabsch(truth, x * np.array([1.0, 1.0, -1.0])))


def spearman(a: F64Array, b: F64Array) -> float:
    """Returns the Spearman rank correlation of two vectors."""
    return float(spearmanr(a, b).statistic)
