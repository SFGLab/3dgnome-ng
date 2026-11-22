"""The agreement of the pooled map of an arm with Hi-C, split by the separation of two bins.

    python playground/band_scc.py [--res BP] MCOOL REGION OUT_DIR ROW ARM [ARM ...]
    python playground/band_scc.py [--res BP] --halves MCOOL_A MCOOL_B REGION ROW

OUT_DIR holds the structures of a trial, one folder `<ROW>_<ARM>` per cell, region and arm, as
`out/confirm20`. The maps are at `--res`, 25 kb unless given, against the balanced counts of
MCOOL. `_scc` weights the correlation of a diagonal by its length times the two standard
deviations. This script prints the same correlation for four bands of diagonals, under 250 kb,
to 1 Mb, to 2 Mb and beyond.

At 25 kb a pooled map of twenty structures holds few contacts between bins more than 1 Mb apart
and the bands beyond 1 Mb read near zero for any arm. `--res 100000` is the resolution for them.

Per arm, one line:

    BAND <row> <arm> <res> contacts <whole> <band> <band> <band> <band> share <share> ... distance <band> ...

`contacts` is the pooled map of `ensemble_score.py` per bead pair. `share` is the part of the
weight of the whole score that each band holds. `distance` is the mean distance of the centroids
of the beads of two bins over the structures, in bonds, with the sign turned.

With `--halves` the line is the first map against the second, both balanced, as the agreement
that the data has with itself.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "playground"))

from validation_battery import CONTACT_BEADS, load  # noqa: E402

from legacy.validation.metrics.hic import observed_hic, simulated_contacts  # noqa: E402

EDGES_BP = (250_000, 1_000_000, 2_000_000)


def balanced_map(mcool: str, region: str, res: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns the balanced map of `region` at `res` and the starts of its bins.

    A file without that resolution is read at the coarsest resolution it holds that divides `res`,
    and the balanced counts are summed over the blocks. `observed_hic` takes the nearest
    resolution of the file in that case, which this function does not accept.
    """
    import cooler  # noqa: PLC0415

    groups = cooler.fileops.list_coolers(mcool)
    have = sorted(int(g.rsplit("/", 1)[-1]) for g in groups if g.rsplit("/", 1)[-1].isdigit())
    if not have or res in have:
        m, bins = observed_hic(mcool, region, res, balance=True)
        if len(bins) > 1 and int(bins[1] - bins[0]) != res:
            raise SystemExit(f"{mcool} has bins of {int(bins[1] - bins[0])} bp, not {res}")
        return m, bins
    fine = [r for r in have if r < res and res % r == 0]
    if not fine:
        raise SystemExit(f"{mcool} holds {have} and none of them divides {res}")
    step = res // fine[-1]
    m, bins = observed_hic(mcool, region, fine[-1], balance=True)
    n = (len(bins) // step) * step
    blocks = m[:n, :n].reshape(n // step, step, n // step, step).sum(axis=(1, 3))
    return blocks, bins[:n:step]


def band_scores(a: np.ndarray, b: np.ndarray, res: int, sign: float = 1.0) -> tuple[list[float], list[float]]:
    """Returns the correlation of each band of diagonals and the weight of each band."""
    nb = a.shape[0]
    edges = [1, *[math.ceil(e / res) for e in EDGES_BP], nb]
    scores: list[float] = []
    weights: list[float] = []
    for lo, hi in zip(edges, edges[1:], strict=False):
        num = den = 0.0
        for d in range(lo, min(hi, nb - 1)):
            x, y = np.diagonal(a, d), np.diagonal(b, d)
            sx, sy = float(x.std()), float(y.std())
            if x.size < 3 or sx < 1e-12 or sy < 1e-12:
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            if np.isfinite(r):
                num += x.size * sx * sy * sign * r
                den += x.size * sx * sy
        scores.append(num / den if den > 0 else float("nan"))
        weights.append(den)
    return scores, weights


def main() -> None:
    args = sys.argv[1:]
    res = 25_000
    if args[0] == "--res":
        res = int(args[1])
        args = args[2:]
    if args[0] == "--halves":
        first, _ = balanced_map(args[1], args[3], res)
        second, _ = balanced_map(args[2], args[3], res)
        scores, _weights = band_scores(first, second, res)
        print("BAND", args[4], "halves", res, "contacts nan", " ".join(f"{x:.4f}" for x in scores))
        return
    mcool, region, out, row, arms = args[0], args[1], args[2], args[3], args[4:]
    bal, bins = balanced_map(mcool, region, res)
    nb = len(bins)
    for arm in arms:
        cifs = sorted((Path(out) / f"{row}_{arm}").glob("*.cif"))
        if not cifs:
            continue
        sim = np.zeros((nb, nb))
        count = np.zeros(nb)
        dist = np.zeros((nb, nb))
        whole_chains = 0
        for c in cifs:
            p, starts, _, _ = load(c)
            bond = float(np.median(np.linalg.norm(np.diff(p, axis=0), axis=1)))
            sim += simulated_contacts(p, starts, bins, res, CONTACT_BEADS * bond)
            idx = np.searchsorted(bins, starts, side="right") - 1
            ok = (idx >= 0) & (idx < nb) & (starts < int(bins[-1]) + res)
            n = np.bincount(idx[ok], minlength=nb)
            count += n
            if (n == 0).any():
                continue
            centroid = np.stack(
                [np.bincount(idx[ok], weights=p[ok, k], minlength=nb) / n for k in range(3)], axis=1
            )
            dist += np.linalg.norm(centroid[:, None] - centroid[None, :], axis=2) / bond
            whole_chains += 1
        pair = np.outer(count, count)
        per_pair = np.divide(sim, pair, out=np.zeros_like(sim), where=pair > 0)
        scores, weights = band_scores(per_pair, bal, res)
        total = sum(weights)
        whole = sum(s * w for s, w in zip(scores, weights, strict=True) if np.isfinite(s)) / max(total, 1e-300)
        by_dist = (
            band_scores(dist / whole_chains, bal, res, -1.0)[0] if whole_chains else [float("nan")] * 4
        )
        print(
            "BAND", row, arm, res, f"contacts {whole:.4f}",
            " ".join(f"{x:.4f}" for x in scores),
            "share", " ".join(f"{w / max(total, 1e-300):.4f}" for w in weights),
            "distance", " ".join(f"{x:.4f}" for x in by_dist),
        )  # fmt: skip


if __name__ == "__main__":
    main()
