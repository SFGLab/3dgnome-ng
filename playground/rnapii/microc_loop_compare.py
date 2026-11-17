"""Score model ensembles at the loops called on a Micro-C map, loop by loop.

    python playground/rnapii/microc_loop_compare.py --json out.json <slice.npz> <microc loops.bedpe> \
        ctcf=<loops.bedpe> rnapii=<loops.bedpe> <arm_dir> ...

The loops are the pixels `microc_call_loops.py` wrote. Each loop is compared with control pairs
at the same genomic separation, the loop's two bins shifted together by 3 to 10 bins either way
along the diagonal. Three numbers per loop come out of that.

- The map's enrichment, the loop pixel's raw count over the mean count of its controls.
- A model's contact enrichment, the same ratio on the ensemble's summed contact map, with the
  contact the battery counts, two beads within 1.33 chain bonds.
- A model's distance ratio, the mean 3D distance between the two bins over the median of its
  controls. Under one, the loop's ends are closer than pairs at the same separation.

A loop is scored in an arm when both its bins and at least four controls hold beads. A loop
counts as placed when its ratio is 0.8 or less, and a loop an arm cannot score counts as not
placed. Every loop is put in one class by the input loops that support it, a ChIA-PET loop with
an anchor within one bin of each end. The classes are CTCF, RNAPII alone and none. A model can
only place a loop its input holds. The class without support measures the rest of the model.

The last class is a check of the instrument. Every loop is moved 30 bins along the diagonal,
where no loop was called, and scored the same way. Its ratio should be one and its placed share
is the rate a pair with no loop passes the threshold. That rate differs between tools, since an
ensemble whose structures differ more has noisier mean distances. Two numbers allow for it. The
placed share above the moved loops' share, and the chance that a loop's ratio is under a moved
loop's, one half when the model does not tell them apart.

Prints one row per arm and class and writes the rows and the per loop table to the JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SHIFTS = np.array([k for k in range(-10, 11) if abs(k) >= 3])
MIN_CONTROLS = 4
CLOSER = 0.8
NULL_SHIFT = 30


def coarsen(m: np.ndarray, f: int) -> np.ndarray:
    """Sum a square matrix in blocks of `f` by `f`, dropping a ragged edge."""
    n = (m.shape[0] // f) * f
    return m[:n, :n].reshape(n // f, f, n // f, f).sum(axis=(1, 3))


def controls(i: np.ndarray, j: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The control bins of each loop and a mask of the ones inside the map, loops by shifts."""
    ci, cj = i[:, None] + SHIFTS[None, :], j[:, None] + SHIFTS[None, :]
    ok = (ci >= 0) & (cj < n)
    return np.clip(ci, 0, n - 1), np.clip(cj, 0, n - 1), ok


def count_enrichment(m: np.ndarray, i: np.ndarray, j: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each loop's count and the mean count of its valid controls."""
    ci, cj, ok = controls(i, j, m.shape[0])
    ok &= valid[ci] & valid[cj]
    c = np.where(ok, m[ci, cj], 0.0)
    k = ok.sum(axis=1)
    mean = np.where(k >= MIN_CONTROLS, c.sum(axis=1) / np.maximum(k, 1), np.nan)
    return m[i, j].astype(np.float64), mean


def bin_centroids(pos: np.ndarray, mid: np.ndarray, lo: int, b: int, n: int) -> np.ndarray:
    """The mean position of the beads in each bin, NaN for a bin without beads."""
    idx = (mid - lo) // b
    keep = (idx >= 0) & (idx < n)
    cen = np.full((n, 3), np.nan)
    cnt = np.bincount(idx[keep], minlength=n).astype(np.float64)
    for a in range(3):
        s = np.bincount(idx[keep], weights=pos[keep, a], minlength=n)
        cen[:, a] = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return cen


def supported(loops: list[tuple[int, int]], anchors: list[tuple[int, int, int, int]], b: int) -> np.ndarray:
    """A mask of the loops that an input loop supports, an anchor within one bin of each end."""
    if not anchors:
        return np.zeros(len(loops), dtype=bool)
    a = np.array(anchors, dtype=np.int64)
    order = np.argsort(a[:, 0])
    a = a[order]
    out = np.zeros(len(loops), dtype=bool)
    reach = int((a[:, 1] - a[:, 0]).max()) + 2 * b
    for n, (s1, s2) in enumerate(loops):
        lo = np.searchsorted(a[:, 0], s1 - reach)
        hi = np.searchsorted(a[:, 0], s1 + 2 * b, side="right")
        c = a[lo:hi]
        hit = (c[:, 1] >= s1 - b) & (c[:, 0] <= s1 + 2 * b) & (c[:, 3] >= s2 - b) & (c[:, 2] <= s2 + 2 * b)
        out[n] = bool(hit.any())
    return out


def chance_under(r: np.ndarray, null: np.ndarray) -> float:
    """The chance that a value of `r` is under a value of `null`, ties counted as half."""
    from scipy.stats import mannwhitneyu

    if r.size == 0 or null.size == 0:
        return float("nan")
    return float(mannwhitneyu(null, r).statistic / (len(null) * len(r)))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation over the entries finite in both."""
    from scipy.stats import spearmanr

    k = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[k], y[k]).statistic) if k.sum() > 10 else float("nan")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from crosscell_prep import read_bedpe
    from microc_readout import load_slice

    from playground.validation_battery import CONTACT_BEADS, load
    from validation.metrics.hic import simulated_contacts

    args = sys.argv[1:]
    k = args.index("--json")
    out = Path(args[k + 1])
    del args[k : k + 2]
    slice_path, loops_path, *rest = args
    inputs = {a.split("=", 1)[0]: Path(a.split("=", 1)[1]) for a in rest if "=" in a}
    arms = [a for a in rest if "=" not in a]

    obs5, starts5, b5 = load_slice(Path(slice_path))
    lo_bp = int(starts5[0])
    called = [lp for lp in read_bedpe(Path(loops_path)) if lp[1] >= lo_bp]
    b = called[0][2] - called[0][1]
    f = b // b5
    obs = coarsen(obs5.astype(np.float64), f)
    n = obs.shape[0]
    i = np.array([(lp[1] - lo_bp) // b for lp in called])
    j = np.array([(lp[4] - lo_bp) // b for lp in called])
    keep = (j < n) & (j - i >= 4)
    i, j = i[keep], j[keep]
    called = [lp for lp, kk in zip(called, keep, strict=True) if kk]
    ends = [(lp[1], lp[4]) for lp in called]

    chrom = called[0][0]
    hi_bp = lo_bp + n * b
    marks: dict[str, np.ndarray] = {}
    for name, path in inputs.items():
        own = [(lp[1], lp[2], lp[4], lp[5]) for lp in read_bedpe(path) if lp[0] == chrom and lp[1] >= lo_bp and lp[5] <= hi_bp]
        marks[name] = supported(ends, own, b)
        back = supported([(a[0], a[2]) for a in own], [(s1, s1 + b, s2, s2 + b) for s1, s2 in ends], b)
        print(f"{name}: {len(own)} input loops in the region, {int(back.sum())} at a Micro-C loop", flush=True)
    real = len(called)
    pixels = set(zip(i.tolist(), j.tolist(), strict=True))
    moved = [(a + NULL_SHIFT, c + NULL_SHIFT) for a, c in zip(i.tolist(), j.tolist(), strict=True)]
    moved = [m for m in moved if m[1] < n and not any((m[0] + da, m[1] + db) in pixels for da in (-1, 0, 1) for db in (-1, 0, 1))]
    i = np.concatenate([i, np.array([m[0] for m in moved], dtype=i.dtype)])
    j = np.concatenate([j, np.array([m[1] for m in moved], dtype=j.dtype)])
    is_real = np.arange(len(i)) < real

    def pad(m: np.ndarray) -> np.ndarray:
        return np.concatenate([m, np.zeros(len(i) - real, dtype=bool)])

    ctcf = pad(marks.get("ctcf", np.zeros(real, dtype=bool)))
    rnapii = pad(marks.get("rnapii", np.zeros(real, dtype=bool)))
    classes = {
        "all": is_real,
        "CTCF": ctcf,
        "RNAPII alone": rnapii & ~ctcf,
        "none": is_real & ~ctcf & ~rnapii,
        "moved, no loop": ~is_real,
    }

    cov = obs.sum(axis=0) > 0
    cnt, ctl = count_enrichment(obs, i, j, cov)
    with np.errstate(divide="ignore", invalid="ignore"):
        map_enr = cnt / ctl
    print(f"{real} Micro-C loops at {b} bp, " + ", ".join(f"{c} {int(m.sum())}" for c, m in classes.items()), flush=True)
    rows = []
    for c, m in classes.items():
        ok = m & np.isfinite(ctl) & (ctl > 0)
        rows.append({"arm": "map", "class": c, "loops": int(m.sum()), "scored": int(ok.sum()), "contact_enrichment": float(cnt[ok].sum() / ctl[ok].sum())})
    per_loop: dict[str, list[float]] = {"start1": [e[0] for e in ends], "start2": [e[1] for e in ends], "map": map_enr[:real].tolist()}

    ci, cj, inside = controls(i, j, n)
    for d in arms:
        cifs = sorted(Path(d).glob("*.cif"))
        sim5 = np.zeros_like(obs5, dtype=np.float32)
        dist = np.zeros(len(i))
        dctl = np.zeros(len(i))
        seen = np.zeros(len(i))
        has = np.zeros(n, dtype=bool)
        for cif in cifs:
            pos, _, mid, _ = load(cif)
            rad = CONTACT_BEADS * float(np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
            sim5 += simulated_contacts(pos, mid, starts5, b5, rad).astype(np.float32)
            cen = bin_centroids(pos, mid, lo_bp, b, n)
            has |= np.isfinite(cen[:, 0])
            dl = np.linalg.norm(cen[i] - cen[j], axis=1)
            dc = np.linalg.norm(cen[ci] - cen[cj], axis=2)
            okc = inside & np.isfinite(dc)
            kc = okc.sum(axis=1)
            good = np.isfinite(dl) & (kc >= MIN_CONTROLS)
            dist[good] += dl[good]
            dctl[good] += np.nanmedian(np.where(okc, dc, np.nan)[good], axis=1)
            seen += good
        sim = coarsen(sim5.astype(np.float64), f)
        scnt, sctl = count_enrichment(sim, i, j, has)
        ratio = np.where(seen > 0, dist / np.maximum(dctl, 1e-12), np.nan)
        name = Path(d).name
        per_loop[name] = ratio[:real].tolist()
        null = ratio[~is_real]
        null_placed = float((null <= CLOSER).sum() / max(len(null), 1))
        null = null[np.isfinite(null)]
        for c, m in classes.items():
            ok = m & np.isfinite(ratio)
            okc = ok & np.isfinite(sctl) & (sctl > 0)
            r = ratio[ok]
            row = {
                "arm": name,
                "class": c,
                "n": len(cifs),
                "loops": int(m.sum()),
                "scored": int(ok.sum()),
                "contact_enrichment": float(scnt[okc].sum() / sctl[okc].sum()) if okc.any() else float("nan"),
                "distance_ratio_median": float(np.median(r)) if r.size else float("nan"),
                "closer_than_controls": float((r < 1.0).mean()) if r.size else float("nan"),
                "closer_by_a_fifth": float((r <= CLOSER).mean()) if r.size else float("nan"),
                "placed": float((r <= CLOSER).sum() / max(int(m.sum()), 1)),
                "placed_above_moved": float((r <= CLOSER).sum() / max(int(m.sum()), 1)) - null_placed,
                "under_a_moved_loop": chance_under(r, null),
                "rank_with_map": spearman(map_enr[ok], -np.log(r)) if r.size else float("nan"),
            }
            rows.append(row)
            print(
                f"  {name:<32} {c:<15} loops {row['loops']:>5} scored {row['scored']:>5}  contact {row['contact_enrichment']:6.2f}"
                f"  ratio {row['distance_ratio_median']:.3f}  closer {row['closer_than_controls']:.2f}"
                f"  placed {row['placed']:.2f}  above moved {row['placed_above_moved']:+.2f}"
                f"  under {row['under_a_moved_loop']:.2f}  rank {row['rank_with_map']:+.2f}",
                flush=True,
            )
    for r in rows[: len(classes)]:
        print(f"  map {r['class']:<15} loops {r['loops']:>5} scored {r['scored']:>5}  contact {r['contact_enrichment']:6.2f}")
    out.write_text(json.dumps({"binsize": b, "rows": rows, "loops": per_loop}, indent=1) + "\n")


if __name__ == "__main__":
    main()
