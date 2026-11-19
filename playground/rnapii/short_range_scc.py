"""Hi-C agreement of each arm's pooled ensemble map at short range and fine resolution.

    python playground/rnapii/short_range_scc.py MCOOL CHROM OUT.json ARM_DIR [ARM_DIR ...]

`playground/ensemble_score.py` scores the whole chromosome at 25 kb, where the distance decay
and the compartments carry the correlation. RNAPII loops span tens to hundreds of kilobases,
so this reads the same pooled contacts at 5, 10 and 25 kb and keeps the pairs up to 2 Mb apart.
The measure fixed before the run is the stratum adjusted correlation with the raw map at 10 kb
over every bin, the convention of the whole map score. The other resolutions, the three bands
of separation at 10 kb and the scores over the mapped bins are printed beside it as context.

Only the diagonals up to 2 Mb are held, one row a bin, so a chromosome at 5 kb fits in memory.
The map is read in slabs of rows for the same reason.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "playground"))

from validation_battery import CONTACT_BEADS, load  # noqa: E402

RESOLUTIONS = (5_000, 10_000, 25_000)
MAX_BP = 2_000_000
# Bands of separation in bp, read at 10 kb.
BANDS = ((0, 100_000), (100_000, 500_000), (500_000, 2_000_000))
SLAB_BP = 10_000_000


def observed_band(mcool: str, chrom: str, res: int, depth: int) -> tuple[np.ndarray, np.ndarray]:
    """The raw counts of one chromosome on the diagonals 0 to `depth`.

    Returns the band, entry `[i, d]` the count between bins `i` and `i + d`, and the mask of the
    bins the map's balancing kept.
    """
    import cooler

    c = cooler.Cooler(f"{mcool}::resolutions/{res}")
    n = int(np.ceil(c.chromsizes[chrom] / res))
    band = np.zeros((n, depth + 1))
    step = SLAB_BP // res
    for s0 in range(0, n, step):
        s1 = min(n, s0 + step)
        hi = min(n, s1 + depth)
        m = c.matrix(balance=False).fetch(f"{chrom}:{s0 * res}-{min(hi * res, int(c.chromsizes[chrom]))}")
        m = np.nan_to_num(np.asarray(m, dtype=np.float64))
        for d in range(depth + 1):
            row = np.diagonal(m, d)[: s1 - s0]
            band[s0 : s0 + len(row), d] = row
    weight = c.bins().fetch(chrom)["weight"].to_numpy()
    return band, np.isfinite(weight)


def simulated_band(cifs: list[Path], res: int, n: int, depth: int) -> np.ndarray:
    """The summed contacts of an ensemble on the diagonals 1 to `depth`, at the battery's contact radius."""
    from scipy.spatial import cKDTree

    band = np.zeros((n, depth + 1))
    for cif in cifs:
        pos, start, _, _ = load(cif)
        radius = CONTACT_BEADS * float(np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
        pairs = cKDTree(pos).query_pairs(radius, output_type="ndarray")
        if pairs.size == 0:
            continue
        bi, bj = start[pairs[:, 0]] // res, start[pairs[:, 1]] // res
        lo, d = np.minimum(bi, bj), np.abs(bi - bj)
        ok = (d >= 1) & (d <= depth) & (lo >= 0) & (lo + d < n)
        np.add.at(band, (lo[ok], d[ok]), 1.0)
    return band


def scc(sim: np.ndarray, obs: np.ndarray, d_lo: int, d_hi: int, keep: np.ndarray) -> float:
    """Stratum adjusted correlation over the diagonals `d_lo + 1` to `d_hi`, on the pairs whose
    two bins are kept. Each diagonal's Pearson is weighted by its length and the two spreads."""
    n = sim.shape[0]
    num = den = 0.0
    for d in range(d_lo + 1, d_hi + 1):
        ok = keep[: n - d] & keep[d:]
        a, b = sim[: n - d, d][ok], obs[: n - d, d][ok]
        if a.size < 2:
            continue
        sa, sb = a.std(), b.std()
        if sa < 1e-12 or sb < 1e-12:
            continue
        w = a.size * sa * sb
        num += w * float(np.corrcoef(a, b)[0, 1])
        den += w
    return num / den if den > 0 else float("nan")


def main() -> None:
    mcool, chrom, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    arms = [(d, sorted(d.glob("*.cif"))) for d in map(Path, sys.argv[4:])]
    rows = []
    for res in RESOLUTIONS:
        depth = MAX_BP // res
        obs, mapped = observed_band(mcool, chrom, res, depth)
        every = np.ones(len(mapped), dtype=bool)
        bands = [(0, MAX_BP)] + (list(BANDS) if res == 10_000 else [])
        for d, cifs in arms:
            if not cifs:
                continue
            sim = simulated_band(cifs, res, obs.shape[0], depth)
            for lo, hi in bands:
                for name, keep in (("all", every), ("mapped", mapped)):
                    rows.append({"arm": d.name, "n_models": len(cifs), "res": res, "sep_lo": lo, "sep_hi": hi, "bins": name,
                                 "scc": scc(sim, obs, lo // res, hi // res, keep)})
    out.write_text(json.dumps({"map": Path(mcool).name, "chrom": chrom, "rows": rows}, indent=1))
    print(f"{chrom} against {Path(mcool).name}, raw counts, pairs up to {MAX_BP // 1_000_000} Mb")
    for r in rows:
        if r["bins"] == "all":
            print(f"  {r['arm']:>24s}  {r['res'] // 1000:2d} kb  {r['sep_lo'] // 1000:5d} to {r['sep_hi'] // 1000:5d} kb  scc {r['scc']:.3f}")


if __name__ == "__main__":
    main()
