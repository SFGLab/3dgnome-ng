r"""Writes the anchor contact table of `gnome3d.anchor_contacts` from an mcool and an anchor file.

    python slurm/ensemble/prep_anchor_contacts.py --mcool MAP.mcool --anchors ANCHORS.bed \
        --chroms chr1-chr22,chrX --res 1000 --out data/GM12878/GM12878_anchor_contacts_1kb.npz

The map is read at full depth, raw counts. Each pixel is shared between the anchors it
overlaps in proportion to the overlap. The anchors must be the anchor file of the run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cooler
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gnome3d import anchor_contacts as ac  # noqa: E402
from gnome3d.io import parse_chrs_arg  # noqa: E402

ROWS_PER_READ = 20_000_000


def read_anchors(path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    by: dict[str, set[tuple[int, int]]] = {}
    for line in open(path):
        f = line.split()
        if len(f) >= 3:
            by.setdefault(f[0], set()).add((int(f[1]), int(f[2])))
    out = {}
    for chrom, iv in by.items():
        a = np.array(sorted(iv), dtype=np.int64)
        out[chrom] = (a[:, 0], a[:, 1])
    return out


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True))


def chromosome(
    c: cooler.Cooler, chrom: str, starts: np.ndarray, ends: np.ndarray
) -> ac.AnchorContacts:
    res = int(c.binsize)
    ext = c.extent(chrom)
    bins = c.bins()[ext[0] : ext[1]]
    bin_starts = bins["start"].to_numpy().astype(np.int64)
    w = ac.overlap_weights(bin_starts, res, starts, ends)
    hit = np.asarray(w.sum(1)).ravel() > 0
    offsets = c._load_dset("indexes/bin1_offset")
    pix = c.pixels()
    total = sp.coo_matrix((len(starts), len(starts)))
    acc: list[sp.coo_matrix] = []
    batch_b1: list[np.ndarray] = []
    batch_b2: list[np.ndarray] = []
    batch_c: list[np.ndarray] = []
    pending = 0

    def flush() -> None:
        nonlocal pending
        if not batch_b1:
            return
        acc.append(
            ac.aggregate_pixels(
                w, np.concatenate(batch_b1), np.concatenate(batch_b2), np.concatenate(batch_c)
            )
        )
        batch_b1.clear()
        batch_b2.clear()
        batch_c.clear()
        pending = 0

    for r0, r1 in runs(hit):
        lo, hi = int(offsets[ext[0] + r0]), int(offsets[ext[0] + r1])
        for p0 in range(lo, hi, ROWS_PER_READ):
            p = pix[p0 : min(hi, p0 + ROWS_PER_READ)]
            b1 = p["bin1_id"].to_numpy() - ext[0]
            b2 = p["bin2_id"].to_numpy() - ext[0]
            inside = (b2 >= 0) & (b2 < len(bin_starts))
            keep = inside & hit[np.clip(b2, 0, len(bin_starts) - 1)]
            batch_b1.append(b1[keep])
            batch_b2.append(b2[keep])
            batch_c.append(p["count"].to_numpy()[keep].astype(np.float64))
            pending += int(keep.sum())
            if pending > ROWS_PER_READ:
                flush()
    flush()
    for m in acc:
        total = total + m
    t = sp.coo_matrix(total)
    t.sum_duplicates()
    return ac.AnchorContacts(
        starts=starts,
        ends=ends,
        i=t.row.astype(np.int32),
        j=t.col.astype(np.int32),
        count=t.data.astype(np.float32),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mcool", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--chroms", default="")
    ap.add_argument("--res", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = cooler.Cooler(f"{a.mcool}::/resolutions/{a.res}")
    anchors = read_anchors(a.anchors)
    chroms = parse_chrs_arg(a.chroms)[0] if a.chroms else sorted(anchors)
    tables: ac.AnchorContactMap = {}
    for chrom in chroms:
        if chrom not in anchors or chrom not in c.chromnames:
            print(f"{chrom}: no anchors or not in the map, skipped")
            continue
        t0 = time.time()
        st, en = anchors[chrom]
        tables[chrom] = chromosome(c, chrom, st, en)
        t = tables[chrom]
        print(
            f"{chrom}: {len(st):,} anchors, {len(t.count):,} pairs with a count, "
            f"{float(t.count.sum()):,.0f} contacts, {time.time() - t0:.0f} s",
            flush=True,
        )
    ac.save(
        Path(a.out),
        tables,
        a.res,
        f"{Path(a.mcool).name} at {a.res} bp, anchors {Path(a.anchors).name}",
    )
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
