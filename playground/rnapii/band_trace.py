"""The agreement of a band of separation on the bins where every arm has beads, with its classes of pairs.

    python playground/rnapii/band_trace.py MCOOL CHROM CTCF.bedpe RNAPII.bedpe OUT.json [NAME=]ARM_DIR [[NAME=]ARM_DIR ...]

`short_range_scc.py` scores every bin of a chromosome. A bin without beads holds zero contacts
there. Two arms with different loops have beads in different bins, since the uniform chain covers
the blocks of its loops. This script reads the same pooled contacts per bead pair at 10 kb and
prints these for each arm.

- the number of bins with beads and the stratum adjusted correlation of three bands of separation
  against the raw and the balanced map, over every bin, over the bins with beads in every arm and
  over those of them that the balancing of the map kept,
- the same correlation of minus the mean distance of the centroids of two bins, on the bins with
  beads in every arm. A contact is two beads within 1.33 bonds, which is another share of the pairs
  for a tool with another chain. The distance reads no radius,
- the rank correlation of that distance with the counts over every pair up to 2 Mb apart, with the
  separation kept, the score that a fit of distances to a Hi-C map is built for,
- for the pairs 100 to 500 kb apart with beads in every arm, by the classes of `band_pairs.py`,
  the contacts per 1,000 pairs of beads of a structure and the mean contact over the mean of its
  separation, beside the same ratio of the raw and of the balanced map. The pooled count of a cell
  over the product of the pooled bead counts of its two bins is the share of a structure over the
  number of structures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from band_pairs import CLASSES, loops  # noqa: E402
from short_range_scc import BANDS, load, observed_band, scc, simulated_band  # noqa: E402

RES = 10_000
LO, HI = 100_000, 500_000


def beads_per_bin(cifs: list[Path], n: int) -> np.ndarray:
    """Returns the mean number of beads of a structure in each bin."""
    beads = np.zeros(n)
    for cif in cifs:
        _, start, _, _ = load(cif)
        idx = start // RES
        beads += np.bincount(idx[(idx >= 0) & (idx < n)], minlength=n)[:n]
    return beads / max(len(cifs), 1)


def distance_band(cifs: list[Path], n: int, depth: int) -> np.ndarray:
    """Returns minus the mean distance of the centroids of the beads of two bins over the
    structures, in bonds of each structure, on the diagonals 1 to `depth`. A cell that no
    structure has beads at holds zero."""
    total = np.zeros((n, depth + 1))
    seen = np.zeros((n, depth + 1))
    for cif in cifs:
        pos, start, _, _ = load(cif)
        bond = float(np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
        idx = start // RES
        ok = (idx >= 0) & (idx < n)
        count = np.bincount(idx[ok], minlength=n)[:n]
        centre = np.full((n, 3), np.nan)
        for k in range(3):
            centre[:, k] = np.divide(np.bincount(idx[ok], weights=pos[ok, k], minlength=n)[:n], count, out=np.full(n, np.nan), where=count > 0)
        for d in range(1, depth + 1):
            dist = np.linalg.norm(centre[: n - d] - centre[d:], axis=1) / bond
            has = np.isfinite(dist)
            total[: n - d, d][has] += dist[has]
            seen[: n - d, d][has] += 1
    return -np.divide(total, seen, out=np.zeros_like(total), where=seen > 0)


def classes(n: int, d0: int, d1: int, ctcf: str, rnapii: str, chrom: str) -> np.ndarray:
    """Returns the class of every pair of the band, as `band_pairs.py` assigns it, -1 outside the band."""
    joined: dict[tuple[int, int], int] = {}
    for k, path in ((1, ctcf), (0, rnapii)):
        for a, b in loops(path, chrom):
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    if 0 <= a + da < b + db < n:
                        joined[(a + da, b + db)] = k if k == 0 else joined.get((a + da, b + db), 1)
    anchor = np.zeros(n, dtype=bool)
    for a, b in loops(rnapii, chrom):
        anchor[[a, b]] = True
    cls = np.full((n, d1 + 1), -1)
    for d in range(d0, d1 + 1):
        i = np.arange(n - d)
        both, one = anchor[i] & anchor[i + d], anchor[i] ^ anchor[i + d]
        cls[i, d] = np.where(both, 2, np.where(one, 3, 4))
    for (a, b), k in joined.items():
        if d0 <= b - a <= d1:
            cls[a, b - a] = k
    return cls


def ratio(band: np.ndarray, keep: np.ndarray, d0: int, d1: int) -> np.ndarray:
    """Returns each cell over the mean of its separation, the mean taken over the kept pairs."""
    n = band.shape[0]
    out = np.zeros_like(band)
    for d in range(d0, d1 + 1):
        ok = keep[: n - d] & keep[d:]
        mean = band[: n - d, d][ok].mean() if ok.any() else 0.0
        if mean > 0:
            out[: n - d, d] = band[: n - d, d] / mean
    return out


def main() -> None:
    mcool, chrom, ctcf, rnapii, out = sys.argv[1:6]
    given = [a.split("=", 1) if "=" in a else [Path(a).parent.name, a] for a in sys.argv[6:]]
    names, folders = [g[0] for g in given], [Path(g[1]) for g in given]
    depth = BANDS[-1][1] // RES
    obs, mapped, balanced = observed_band(mcool, chrom, RES, depth)
    n = obs.shape[0]
    per_pair, beads, count, distance = {}, {}, {}, {}
    for name, folder in zip(names, folders, strict=True):
        cifs = sorted(folder.glob("*.cif"))
        _, per_pair[name] = simulated_band(cifs, RES, n, depth)
        distance[name] = distance_band(cifs, n, depth)
        beads[name] = beads_per_bin(cifs, n)
        count[name] = len(cifs)
    every = np.ones(n, dtype=bool)
    common = np.all([b > 0 for b in beads.values()], axis=0)
    keeps = {"every bin": every, "bins with beads in every arm": common, "of them mapped": common & mapped}
    rows = []
    print(f"{chrom} at {RES // 1000} kb, {n} bins, {int(common.sum())} with beads in every arm, {int((common & mapped).sum())} of them mapped")
    for name in names:
        print(f"{name}: {count[name]} structures, {int((beads[name] > 0).sum())} bins with beads, {beads[name].sum():,.0f} beads")
        for label, keep in keeps.items():
            parts, far = [], []
            for lo, hi in BANDS:
                raw = scc(per_pair[name], obs, lo // RES, hi // RES, keep)
                bal = scc(per_pair[name], balanced, lo // RES, hi // RES, keep)
                row = {"arm": name, "bins": label, "sep_lo": lo, "sep_hi": hi, "scc_pair": raw, "scc_pair_bal": bal}
                parts.append(f"{lo // 1000} to {hi // 1000} kb raw {raw:.3f} balanced {bal:.3f}")
                if keep is not every:
                    # The distance has no value at a bin without beads.
                    row["scc_distance"] = scc(distance[name], obs, lo // RES, hi // RES, keep)
                    row["scc_distance_bal"] = scc(distance[name], balanced, lo // RES, hi // RES, keep)
                    far.append(f"{lo // 1000} to {hi // 1000} kb raw {row['scc_distance']:.3f} balanced {row['scc_distance_bal']:.3f}")
                rows.append(row)
            print(f"  {label:>28s}  " + "  ".join(parts))
            if far:
                print(f"  {'the same by the distance':>28s}  " + "  ".join(far))
    # The rank correlation of the distance with the counts over every pair up to 2 Mb apart, the
    # separation not removed. A fit of distances to a map is built for this score.
    from scipy.stats import spearmanr

    keep = common & mapped
    pair = np.zeros((n, depth + 1), dtype=bool)
    for d in range(1, depth + 1):
        pair[: n - d, d] = keep[: n - d] & keep[d:]
    rank = {}
    for name in names:
        has = pair & (distance[name] < 0)
        rank[name] = {
            "raw": float(spearmanr(distance[name][has], obs[has])[0]),
            "balanced": float(spearmanr(distance[name][has], balanced[has])[0]),
        }
        print(f"{name}: rank correlation of the distance with the counts, separation kept, raw {rank[name]['raw']:.3f} balanced {rank[name]['balanced']:.3f}")
    d0, d1 = LO // RES + 1, HI // RES
    cls = classes(n, d0, d1, ctcf, rnapii, chrom)
    pair_ok = np.zeros_like(cls, dtype=bool)
    for d in range(d0, d1 + 1):
        pair_ok[: n - d, d] = common[: n - d] & common[d:]
    keep_map = common & mapped
    pair_map = np.zeros_like(pair_ok)
    for d in range(d0, d1 + 1):
        pair_map[: n - d, d] = keep_map[: n - d] & keep_map[d:]
    o_raw = ratio(obs[:, : d1 + 1], common, d0, d1)
    o_bal = ratio(balanced[:, : d1 + 1], keep_map, d0, d1)
    table = []
    print(f"pairs {LO // 1000} to {HI // 1000} kb apart with beads in every arm")
    print(f"  {'class':>20s} {'pairs':>9s}  raw map  balanced map  " + "  ".join(f"{name:>22s}" for name in names))
    print(f"  {'':>20s} {'':>9s}  {'':>7s}  {'':>12s}  " + "  ".join(f"{'per 1,000':>11s} {'over mean':>10s}" for _ in names))
    for k, label in enumerate(CLASSES):
        m = (cls == k) & pair_ok
        if not m.any():
            continue
        mm = (cls == k) & pair_map
        row = {"class": label, "pairs": int(m.sum()), "map_raw": float(o_raw[m].mean()), "map_balanced": float(o_bal[mm].mean())}
        cells = []
        for name in names:
            band = per_pair[name][:, : d1 + 1]
            rate = 1000 * float(band[m].mean()) * count[name]
            over = float(ratio(band, common, d0, d1)[m].mean())
            row[name] = {"per_1000": rate, "over_mean": over}
            cells.append(f"{rate:11.2f} {over:10.2f}")
        table.append(row)
        print(f"  {label:>20s} {int(m.sum()):9,d}  {row['map_raw']:7.2f}  {row['map_balanced']:12.2f}  " + "  ".join(f"{c:>22s}" for c in cells))
    m = (cls >= 0) & pair_ok
    cells = [f"{1000 * float(per_pair[name][:, : d1 + 1][m].mean()) * count[name]:11.2f} {1.0:10.2f}" for name in names]
    print(f"  {'the band':>20s} {int(m.sum()):9,d}  {1.0:7.2f}  {1.0:12.2f}  " + "  ".join(f"{c:>22s}" for c in cells))
    Path(out).write_text(json.dumps({"chrom": chrom, "bins": n, "common": int(common.sum()), "scc": rows, "rank_distance": rank, "classes": table}, indent=1))


if __name__ == "__main__":
    main()
