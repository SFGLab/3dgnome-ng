"""Cut one region of a Juicer `.hic`, local or a URL, into a sparse npz the readouts read.

    python playground/rnapii/hic_slice.py <hic path or URL> chr1:0-60000000 5000 out.npz

Runs under any Python that has `hic-straw`; the readouts themselves do not import it. The npz
holds `bin1`, `bin2` (upper triangle, bin indices from the region start), `count` (observed,
no normalisation), `start`, `binsize` and `nbins`.
"""

from __future__ import annotations

import sys

import numpy as np


def main() -> None:
    import hicstraw

    path, region, binsize, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    chrom, span = region.split(":")
    a, b = (int(x) for x in span.split("-"))
    # Juicer files name chromosomes with or without the chr prefix
    names = {c.name for c in hicstraw.HiCFile(path).getChromosomes()}
    name = chrom if chrom in names else chrom.removeprefix("chr")
    q = f"{name}:{a}:{b - 1}"
    recs = hicstraw.straw("observed", "NONE", path, q, q, "BP", binsize)
    x = np.array([r.binX for r in recs], dtype=np.int64)
    y = np.array([r.binY for r in recs], dtype=np.int64)
    c = np.array([r.counts for r in recs], dtype=np.float64)
    i, j = (np.minimum(x, y) - a) // binsize, (np.maximum(x, y) - a) // binsize
    nb = (b - a + binsize - 1) // binsize
    ok = (i >= 0) & (j < nb)
    np.savez_compressed(out, bin1=i[ok], bin2=j[ok], count=c[ok], start=a, binsize=binsize, nbins=nb)
    print(f"{out}: {ok.sum()} pixels, {nb} bins, total {c[ok].sum():.0f}")


if __name__ == "__main__":
    main()
