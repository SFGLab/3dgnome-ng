"""The contacts and the spread of the distance of pairs of beads 100 to 500 kb apart, by the loop ends at them.

    python playground/rnapii/end_pair_contacts.py CHROM CTCF.bedpe RNAPII.bedpe OUT.json [NAME=]ARM_DIR [[NAME=]ARM_DIR ...]

A bead is an RNAPII end when it is the bead nearest the middle of an anchor of an RNAPII loop,
within 3 kb, a CTCF end the same for a CTCF loop, and a chain bead otherwise. The pairs are the
RNAPII loops of the band, the CTCF loops of the band, every pair of two RNAPII ends of the band
that no loop joins, and random pairs of an RNAPII end with a chain bead and of two chain beads at
the separations of the pairs of ends. The positions of a pair are fixed over the structures of an
arm and of every arm, and a pair counts in a structure that has a bead at both.

Per arm and set: the pairs, the share within 1.33 bonds, the contact radius of the scores, within
2 and within 3, the median distance in bonds and the spread of a pair over the structures, the
median over the pairs of the standard deviation of the distance over its mean. The last line of an
arm is the distance of consecutive RNAPII ends along the chain and the kilobases between them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "playground"))

from validation_battery import load  # noqa: E402

LO, HI = 100_000, 500_000
SAMPLE = 200_000


def anchors(path: str, chrom: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns the middles of the anchors of the loops of a chromosome and the loops as pairs of middles."""
    pairs = []
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if len(f) >= 6 and f[0] == chrom and f[3] == chrom:
            a, b = (int(f[1]) + int(f[2])) // 2, (int(f[4]) + int(f[5])) // 2
            pairs.append((min(a, b), max(a, b)))
    arr = np.array(pairs)
    return np.unique(arr), arr


def in_band(pairs: np.ndarray) -> np.ndarray:
    span = pairs[:, 1] - pairs[:, 0]
    return pairs[(span >= LO) & (span <= HI)]


def main() -> None:
    chrom, ctcf, rnapii, out = sys.argv[1:5]
    given = [a.split("=", 1) if "=" in a else [Path(a).parent.name, a] for a in sys.argv[5:]]
    names, folders = [g[0] for g in given], [Path(g[1]) for g in given]
    rng = np.random.default_rng(0)
    ends_r, loops_r = anchors(rnapii, chrom)
    ends_c, loops_c = anchors(ctcf, chrom)
    i, j = np.triu_indices(len(ends_r), 1)
    span = ends_r[j] - ends_r[i]
    ok = (span >= LO) & (span <= HI)
    two = np.c_[ends_r[i[ok]], ends_r[j[ok]]]
    looped = {(int(a), int(b)) for a, b in loops_r} | {(int(a), int(b)) for a, b in loops_c}
    two = np.array([p for p in two.tolist() if (p[0], p[1]) not in looped])
    if len(two) > SAMPLE:
        two = two[rng.choice(len(two), SAMPLE, replace=False)]
    # Chain positions are drawn away from every anchor middle by 5 kb or more.
    every_end = np.sort(np.concatenate([ends_r, ends_c]))
    lo, hi = int(every_end.min()), int(every_end.max())

    def chain_positions(k: int) -> np.ndarray:
        got = np.empty(0, dtype=np.int64)
        while len(got) < k:
            x = rng.integers(lo, hi, 2 * k)
            at = np.searchsorted(every_end, x).clip(1, len(every_end) - 1)
            far = np.minimum(np.abs(every_end[at] - x), np.abs(every_end[at - 1] - x)) >= 5000
            got = np.concatenate([got, x[far]])
        return got[:k]

    spans = (two[:, 1] - two[:, 0])[rng.integers(0, len(two), SAMPLE)]
    end = ends_r[rng.integers(0, len(ends_r), SAMPLE)]
    side = rng.choice((-1, 1), SAMPLE)
    other = end + side * spans
    at = np.searchsorted(every_end, other).clip(1, len(every_end) - 1)
    far = np.minimum(np.abs(every_end[at] - other), np.abs(every_end[at - 1] - other)) >= 5000
    one = np.c_[np.minimum(end, other), np.maximum(end, other)][far]
    start = chain_positions(SAMPLE)
    none = np.c_[start, start + spans]
    sets = {
        "RNAPII loop": in_band(loops_r),
        "CTCF loop": in_band(loops_c),
        "two RNAPII ends, no loop": two,
        "RNAPII end and chain": one,
        "two chain beads": none,
    }
    result = {}
    for name, folder in zip(names, folders, strict=True):
        cifs = sorted(folder.glob("*.cif"))
        dist = {k: np.full((len(cifs), len(v)), np.nan) for k, v in sets.items()}
        step, gap = [], []
        for s, cif in enumerate(cifs):
            pos, _, mid, _ = load(cif)
            bond = float(np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1)))

            def bead(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                at = np.searchsorted(mid, x).clip(1, len(mid) - 1)
                at = np.where(np.abs(mid[at - 1] - x) < np.abs(mid[at] - x), at - 1, at)
                return at, np.abs(mid[at] - x) < 3000

            for k, pairs in sets.items():
                a, oka = bead(pairs[:, 0])
                b, okb = bead(pairs[:, 1])
                d = np.linalg.norm(pos[a] - pos[b], axis=1) / bond
                dist[k][s] = np.where(oka & okb, d, np.nan)
            at, ok_end = bead(ends_r)
            at = np.unique(at[ok_end])
            near = np.diff(mid[at]) < 100_000
            step.append(np.linalg.norm(np.diff(pos[at], axis=0), axis=1)[near] / bond)
            gap.append(np.diff(mid[at])[near] / 1000)
        print(f"{name}: {len(cifs)} structures")
        result[name] = {}
        for k, d in dist.items():
            seen = np.isfinite(d).sum(axis=0)
            flat = d[np.isfinite(d)]
            full = d[:, seen >= max(2, len(cifs) // 2)]
            with np.errstate(invalid="ignore"):
                cv = np.nanstd(full, axis=0) / np.nanmean(full, axis=0)
            row = {
                "pairs": int((seen > 0).sum()),
                "within_1.33": float(np.mean(flat <= 1.33)),
                "within_2": float(np.mean(flat <= 2)),
                "within_3": float(np.mean(flat <= 3)),
                "median": float(np.median(flat)),
                "spread": float(np.nanmedian(cv)),
                "percentiles": [float(x) for x in np.percentile(flat, [1, 5, 25, 50, 75, 95])],
                "histogram": np.histogram(flat, bins=np.arange(0, 12.25, 0.25))[0].tolist(),
            }
            result[name][k] = row
            print(
                f"  {k:>26s}  pairs {row['pairs']:7,d}  within 1.33 {row['within_1.33']:.4f}  within 2 {row['within_2']:.4f}"
                f"  within 3 {row['within_3']:.3f}  median {row['median']:5.2f}  spread {row['spread']:.3f}"
            )
        s, g = np.concatenate(step), np.concatenate(gap)
        result[name]["consecutive ends"] = {"median_distance": float(np.median(s)), "median_kb": float(np.median(g)), "count": len(s) // max(len(cifs), 1)}
        print(f"  consecutive RNAPII ends under 100 kb apart: {len(s) // max(len(cifs), 1):,d} a structure, median {np.median(g):.0f} kb and {np.median(s):.2f} bonds")
    Path(out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
