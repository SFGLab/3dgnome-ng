"""Gathers the track of components per locus of a cell line from its Hi-C map, the extension of the A/B
compartment track. `[data] components` reads it.

    python playground/component_track.py MCOOL OUT.bed [--chroms chr1,chr2] [--k 5] [--res 100000]
        [--leave-out chr:a-b,...]

Every chromosome of the map is gathered unless `--chroms` names some. The balanced map of a chromosome at
`--res` is split into its arms at the longest run of bins without a count. Each arm gives the log of the
count over the mean count of its separation, with the mean of each separation taken off, a symmetric
matrix whose `--k` eigenvectors of largest absolute eigenvalue are the components. A bin gets
`x_c = v_c * sqrt(|lambda_c|)` for each. The affinity of two bins of one arm is
`sum over c of sign_c * x_c(i) * x_c(j)`, an estimate of the log of their observed over expected count
from 2 k numbers, with a mean of zero at every separation.

`--leave-out` is for trials. It names regions whose own block of the matrix is set to no signal before the
eigenvectors are taken. The components of a locus of such a region then come from its counts with the rest
of the arm alone, and a score of a model on the counts inside the region is a score on counts the track did
not read.

The output has a line `# sign CHROM ARM s_1 ... s_k` per arm and a line `CHROM START END ARM x_1 ... x_k`
per bin with a count.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "playground"))

from band_scc import balanced_map  # noqa: E402


def log_observed_over_expected(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns the log of the count over the mean count of its diagonal, with the mean of each
    diagonal taken off, and the bins with a count.

    A count is taken with half of the mean of its diagonal added on both sides, which bounds the log
    of an empty cell at log of a third. The mean of the log over a diagonal is then taken off. The
    affinity of two bins is their contact against the other pairs at their separation, and its mean
    at a separation is zero.
    """
    n = m.shape[0]
    valid = m.sum(axis=0) > 0
    out = np.zeros_like(m)
    for d in range(1, n):
        v = np.diagonal(m, d)
        ok = valid[: n - d] & valid[d:]
        if not ok.any():
            continue
        mean = float(v[ok].mean())
        if mean <= 0.0:
            continue
        x = np.log((v + 0.5 * mean) / (1.5 * mean))
        x = np.where(ok, x - float(x[ok].mean()), 0.0)
        i = np.arange(n - d)
        out[i, i + d] = x
        out[i + d, i] = x
    return out, valid


def arms_of(valid: np.ndarray, res: int) -> list[tuple[int, int]]:
    """Returns the bin ranges of the arms, split at the longest run of bins without a count when
    that run is 2 Mb or longer.
    """
    n = len(valid)
    best = (0, 0)
    i = 0
    while i < n:
        if valid[i]:
            i += 1
            continue
        j = i
        while j < n and not valid[j]:
            j += 1
        if 0 < i and j < n and j - i > best[1] - best[0]:
            best = (i, j)
        i = j
    if (best[1] - best[0]) * res < 2_000_000:
        return [(0, n)]
    return [(0, best[0]), (best[1], n)]


def main() -> None:
    import cooler  # noqa: PLC0415

    args = sys.argv[1:]
    opts = {"--k": "5", "--res": "100000", "--leave-out": "", "--chroms": ""}
    rest: list[str] = []
    while args:
        a = args.pop(0)
        if a in opts:
            opts[a] = args.pop(0)
        else:
            rest.append(a)
    mcool, out = rest
    k, res = int(opts["--k"]), int(opts["--res"])
    groups = cooler.fileops.list_coolers(mcool)
    first = cooler.Cooler(f"{mcool}::{groups[0]}" if groups and groups[0] != "/" else mcool)
    chroms = [c for c in opts["--chroms"].split(",") if c] or list(first.chromnames)
    lines: list[str] = []
    for chrom in chroms:
        m, bins = balanced_map(mcool, chrom, res)
        log_ratio, valid = log_observed_over_expected(m)
        for region in filter(None, opts["--leave-out"].split(",")):
            c, span = region.split(":")
            if c != chrom:
                continue
            lo, hi = (int(x) for x in span.split("-"))
            i0, i1 = int(np.searchsorted(bins, lo)), int(np.searchsorted(bins, hi))
            log_ratio[i0:i1, i0:i1] = 0.0
        held = 0
        for arm, (a0, a1) in enumerate(arms_of(valid, res)):
            if int(valid[a0:a1].sum()) <= k:
                continue
            w, v = np.linalg.eigh(log_ratio[a0:a1, a0:a1])
            order = np.argsort(-np.abs(w))[:k]
            x = v[:, order] * np.sqrt(np.abs(w[order]))
            lines.append(f"# sign {chrom} {arm} " + " ".join(f"{int(np.sign(s)):+d}" for s in w[order]))
            idx = np.arange(a1 - a0)
            gap = np.abs(idx[:, None] - idx[None, :]) * res
            far = (gap >= 1_000_000) & (gap <= 5_000_000) & valid[a0:a1][:, None] & valid[a0:a1][None, :]
            affinity = (x * np.sign(w[order])) @ x.T
            if far.any():
                print(
                    f"  {chrom} arm {arm}: {int(valid[a0:a1].sum())} bins, eigenvalues {np.round(w[order], 1)},"
                    f" affinity of pairs 1 to 5 Mb apart mean {affinity[far].mean():+.3f} sd {affinity[far].std():.3f},"
                    f" of the map there sd {log_ratio[a0:a1, a0:a1][far].std():.3f}"
                )
            for i in range(a1 - a0):
                if valid[a0 + i]:
                    start = int(bins[a0 + i])
                    values = " ".join(f"{val:.5f}" for val in x[i])
                    lines.append(f"{chrom}\t{start}\t{start + res}\t{arm}\t{values}")
                    held += 1
        print(f"  {chrom}: {held} bins")
    Path(out).write_text("\n".join(lines) + "\n")
    print(f"{out}: {sum(1 for ln in lines if not ln.startswith('#'))} bins of {len(chroms)} chromosomes, {k} components")


if __name__ == "__main__":
    main()
