"""Cut one region of a cooler into the sparse npz the Micro-C readouts read, at a coarser bin.

    python playground/rnapii/cool_slice.py <cool URI> chr1:0-60000000 5000 out.npz

The cooler may be far finer than the bin asked for, 50 bp for the deep Micro-C maps. Its pixels
are read in chunks and summed into the coarse bins, so the fine matrix is never held. The npz
has the keys `hic_slice.py` writes, `bin1`, `bin2`, `count`, `start`, `binsize` and `nbins`,
raw counts in the upper triangle with bin indices from the region start.
"""

from __future__ import annotations

import sys

import numpy as np

CHUNK = 20_000_000


def main() -> None:
    import cooler

    uri, region, binsize, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    chrom, span = region.split(":")
    a, b = (int(x) for x in span.replace(",", "").split("-"))
    clr = cooler.Cooler(uri)
    name = chrom if chrom in clr.chromnames else chrom.removeprefix("chr")
    fine = int(clr.binsize)
    if binsize % fine or a % binsize:
        raise ValueError(f"bin size {binsize} and start {a} must be multiples of the cooler's {fine}")
    b = min(b, int(clr.chromsizes[name]))
    n = -(-(b - a) // binsize)
    lo, hi = clr.extent((name, a, b))
    with clr.open("r") as h:
        p0, p1 = int(h["indexes/bin1_offset"][lo]), int(h["indexes/bin1_offset"][hi])
        dense = np.zeros((n, n), dtype=np.float64)
        f = binsize // fine
        for s in range(p0, p1, CHUNK):
            e = min(s + CHUNK, p1)
            b1 = h["pixels/bin1_id"][s:e]
            b2 = h["pixels/bin2_id"][s:e]
            c = h["pixels/count"][s:e]
            keep = b2 < hi
            np.add.at(dense, ((b1[keep] - lo) // f, (b2[keep] - lo) // f), c[keep])
            print(f"  {e - p0:,} of {p1 - p0:,} pixels", flush=True)
    i, j = np.nonzero(np.triu(dense))
    np.savez_compressed(out, bin1=i, bin2=j, count=dense[i, j], start=a, binsize=binsize, nbins=n)
    print(f"{out}: {name}:{a}-{b} of {clr.chromsizes[name]:,} bp, {n} bins of {binsize}, {len(i):,} pixels, {dense[i, j].sum():,.0f} contacts")


if __name__ == "__main__":
    main()
