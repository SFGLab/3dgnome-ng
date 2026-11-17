"""Pile-up of contacts at RNAPII loops, in a measured map and in model ensembles.

Every pixel is divided by the mean of its diagonal, then the window of `2 flank + 1` bins
around each loop's pixel is averaged over the loops. The score is the centre pixel over the mean
of the 3 by 3 lower left corner, the corner nearer the diagonal, Juicer's P2LL. Juicer skips
loops under 30 bins; here the floor is 5, so the windows of loops under 150 kb reach the
diagonal, which the division by the diagonal's mean puts on the same footing. A map with
nothing at the loops scores one. Loops closer than `min_sep` bins, or whose window leaves the
map, are not used.

Defaults are 10 kb bins, a flank of 10 bins and a minimum separation of 5 bins. The measured map
is read raw, since not every 4DN file carries balancing weights, and dividing by the diagonal's
mean removes the distance decay either way. A model map is the sum over its ensemble of the
contacts the battery counts, two beads within 1.33 chain bonds.

    python playground/rnapii/pileup.py --json out.json <mcool> <region> <binsize> <loops.bedpe> [<arm_dir> ...]

Writes one row per source, the map first as arm "map", and the piles beside the JSON as
`<json stem>.npz` keyed by arm.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

Loop = tuple[str, int, int, str, int, int, int]


def observed_over_expected(matrix: np.ndarray) -> np.ndarray:
    """The matrix divided by the mean of each of its diagonals, zero where that mean is zero."""
    n = matrix.shape[0]
    out = np.zeros_like(matrix, dtype=np.float64)
    for k in range(n):
        d = np.diagonal(matrix, k)
        mean = float(d.mean())
        if mean > 0:
            idx = np.arange(n - k)
            out[idx, idx + k] = d / mean
            out[idx + k, idx] = d / mean
    return out


def pile(
    matrix: np.ndarray, bin_starts: np.ndarray, binsize: int, loops: list[Loop], flank: int, min_sep: int
) -> tuple[np.ndarray, int]:
    """The mean observed over expected window around the loops, and how many loops were used.

    Parameters
    ----------
    matrix
        A symmetric contact map on the bins `bin_starts`, each `binsize` wide.
    loops
        Loops on the map's chromosome; an end's bin is the bin of its midpoint.
    """
    oe = observed_over_expected(matrix)
    n = oe.shape[0]
    first = int(bin_starts[0])
    acc = np.zeros((2 * flank + 1, 2 * flank + 1))
    used = 0
    for _, s1, e1, _, s2, e2, _ in loops:
        a = ((s1 + e1) // 2 - first) // binsize
        b = ((s2 + e2) // 2 - first) // binsize
        if b - a < min_sep or a - flank < 0 or b + flank >= n:
            continue
        acc += oe[a - flank : a + flank + 1, b - flank : b + flank + 1]
        used += 1
    return (acc / used if used else acc), used


def apa_score(p: np.ndarray) -> float:
    """The centre pixel over the mean of the 3 by 3 lower left corner."""
    c = p.shape[0] // 2
    corner = float(p[-3:, :3].mean())
    return float(p[c, c]) / corner if corner > 0 else float("nan")


def main() -> None:
    import json

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from crosscell_prep import read_bedpe

    from playground.validation_battery import CONTACT_BEADS, load
    from validation.metrics.hic import observed_hic, simulated_contacts

    args = sys.argv[1:]
    k = args.index("--json")
    json_out = Path(args[k + 1])
    del args[k : k + 2]
    mcool, region, binsize, loops_path, *arms = args
    binsize = int(binsize)
    flank, min_sep = 10, 5

    chrom = region.split(":")[0]
    c_obs, bin_starts = observed_hic(mcool, region, binsize, balance=False)
    # observed_hic snaps to the nearest resolution the file holds
    binsize = int(bin_starts[1] - bin_starts[0]) if len(bin_starts) > 1 else binsize
    lo, hi = int(bin_starts[0]), int(bin_starts[-1]) + binsize
    loops = [lp for lp in read_bedpe(Path(loops_path)) if lp[0] == chrom and lp[1] >= lo and lp[5] <= hi]

    rows: list[dict[str, object]] = []
    piles: dict[str, np.ndarray] = {}
    p, used = pile(c_obs, bin_starts, binsize, loops, flank, min_sep)
    rows.append({"arm": "map", "n_models": 0, "n_loops": used, "apa": apa_score(p)})
    piles["map"] = p
    for d in arms:
        cifs = sorted(Path(d).glob("*.cif"))
        c_sim = np.zeros_like(c_obs)
        for c in cifs:
            pos, _, mid, _ = load(c)
            radius = CONTACT_BEADS * float(np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
            c_sim += simulated_contacts(pos, mid, bin_starts, binsize, radius)
        p, used = pile(c_sim, bin_starts, binsize, loops, flank, min_sep)
        rows.append({"arm": Path(d).name, "n_models": len(cifs), "n_loops": used, "apa": apa_score(p)})
        piles[Path(d).name] = p
    for r in rows:
        print(f"  {r['arm']:<28} models {r['n_models']:>3}  loops {r['n_loops']:>6}  APA {r['apa']:.3f}")
    json_out.write_text(json.dumps(rows, indent=2) + "\n")
    np.savez(json_out.with_suffix(".npz"), **piles)


if __name__ == "__main__":
    main()
