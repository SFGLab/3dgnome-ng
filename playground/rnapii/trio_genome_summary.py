"""The whole genome trio readouts as one paired table, CTCF loops against CTCF and RNAPII loops.

    python playground/rnapii/trio_genome_summary.py <readouts/trio_lr dir> [<out.json>]

Reads what `readouts_trio_genome.sh` wrote for each person and chromosome,
`<SAMPLE>_<chr>_pooled.txt` and `<SAMPLE>_<chr>_pileup.{json,npz}`, and adds the chromosomes up.

Correlation with the person's map is the stratum adjusted correlation of the summed contact
map against raw Hi-C at 25 kb, the column the cell lines' figure reads. A person's value is
the mean over chromosomes weighted by chromosome length. The pile-up is added up loop by loop.
Each chromosome's pile is a mean over its loops, so the piles are averaged with the loop counts
as weights and the score is taken once on the result. It is given relative to the same score
of the person's own map.

Prints the table per person, then per measure how many of the nine are higher with RNAPII
loops and the range and median of the relative change. A person with a chromosome missing is
listed and left out of the counts. The JSON holds the rows for the figure and the page.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np

PEOPLE = ("HG00512", "HG00513", "HG00514", "HG00731", "HG00732", "HG00733", "GM19238", "GM19239", "GM19240")
CHROMS = tuple(f"chr{i}" for i in range(1, 23))
# hg38 autosome lengths in bp, the weights of the correlation.
LENGTH = dict(
    zip(
        CHROMS,
        (248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973, 145138636, 138394717, 133797422, 135086622,
         133275309, 114364328, 107043718, 101991189, 90338345, 83257441, 80373285, 58617616, 64444167, 46709983, 50818468),
        strict=True,
    )
)
ARMS = ("C", "R")


def apa_score(p: np.ndarray) -> float:
    """The centre pixel over the mean of the 3 by 3 lower left corner, as pileup.py scores it."""
    c = p.shape[0] // 2
    corner = float(p[-3:, :3].mean())
    return float(p[c, c]) / corner if corner > 0 else float("nan")


def pooled_scc(path: Path, sample: str, chrom: str) -> dict[str, float]:
    """The raw stratum adjusted correlation of each arm, from one pooled score file."""
    out = {}
    for line in path.read_text().splitlines():
        f = line.split()
        for arm in ARMS:
            if f and f[0] == f"{sample}_{arm}_{chrom}":
                out[arm] = float(f[4])
    return out


def person(d: Path, sample: str) -> dict[str, object] | None:
    """One person's whole genome values, None when a chromosome or an arm is missing."""
    scc = {a: 0.0 for a in ARMS}
    piles: dict[str, np.ndarray | float] = {a: 0.0 for a in (*ARMS, "map")}
    loops = 0
    for chrom in CHROMS:
        pooled, pile_json, pile_npz = d / f"{sample}_{chrom}_pooled.txt", d / f"{sample}_{chrom}_pileup.json", d / f"{sample}_{chrom}_pileup.npz"
        if not (pooled.is_file() and pile_json.is_file() and pile_npz.is_file()):
            return None
        s = pooled_scc(pooled, sample, chrom)
        if set(s) != set(ARMS):
            return None
        for a in ARMS:
            scc[a] += LENGTH[chrom] * s[a]
        rows = {r["arm"]: r for r in json.loads(pile_json.read_text())}
        z = np.load(pile_npz)
        n = int(rows["map"]["n_loops"])
        loops += n
        for key, name in (("map", "map"), *((a, f"{sample}_{a}_{chrom}") for a in ARMS)):
            if name not in z.files:
                return None
            piles[key] = piles[key] + n * z[name]
    total = sum(LENGTH.values())
    ref = apa_score(np.asarray(piles["map"]) / loops)
    return {
        "person": sample,
        "loops": loops,
        "map_pileup": ref,
        **{f"scc_{a}": scc[a] / total for a in ARMS},
        **{f"pileup_{a}": apa_score(np.asarray(piles[a]) / loops) / ref for a in ARMS},
    }


def main() -> None:
    d = Path(sys.argv[1])
    rows = []
    for s in PEOPLE:
        r = person(d, s)
        if r is None:
            have = len(list(d.glob(f"{s}_chr*_pileup.npz")))
            print(f"{s}: {have} of {len(CHROMS)} chromosomes read, left out")
        else:
            rows.append(r)
    print(f"{'person':<9}{'loops':>7}  {'SCC C':>7}{'SCC R':>7}{'change':>8}   {'pile-up C':>10}{'pile-up R':>10}{'change':>8}   pile-up of the map")
    for r in rows:
        print(
            f"{r['person']:<9}{r['loops']:>7}  {r['scc_C']:>7.3f}{r['scc_R']:>7.3f}{100 * (r['scc_R'] / r['scc_C'] - 1):>7.0f}%"
            f"   {r['pileup_C']:>10.3f}{r['pileup_R']:>10.3f}{100 * (r['pileup_R'] / r['pileup_C'] - 1):>7.0f}%   {r['map_pileup']:.3f}"
        )
    summary = {}
    for key, label in (("pileup", "pile-up at RNAPII loops, relative to the map's"), ("scc", "correlation with the whole map")):
        change = [100 * (r[f"{key}_R"] / r[f"{key}_C"] - 1) for r in rows]
        if not change:
            continue
        higher = sum(c > 0 for c in change)
        summary[key] = {"higher": higher, "of": len(change), "min": min(change), "max": max(change), "median": statistics.median(change)}
        print(f"{label}: higher with RNAPII loops in {higher} of {len(change)}, change {min(change):+.0f} to {max(change):+.0f}%, median {statistics.median(change):+.0f}%")
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps({"people": rows, "summary": summary}, indent=1) + "\n")


if __name__ == "__main__":
    main()
