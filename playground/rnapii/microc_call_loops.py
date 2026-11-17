"""Call loops on a Micro-C slice with the dots caller of cooltools.

    python playground/rnapii/microc_call_loops.py <slice.npz> <chrom> <out prefix> [nproc]

The slice is the npz `hic_slice.py` cuts, raw counts at one bin size. It is written as a cooler,
balanced, and its loops are called at the slice's bin size and at twice that. The caller is the
HiCCUPS scheme of Rao et al. 2014 as cooltools implements it, with the tool's own defaults. The
expectation is the mean balanced count at each separation within the slice.

Writes `<prefix>_<bin size>.cool` and `<prefix>_<bin size>.bedpe` per bin size. A bedpe row is
the two pixel bins of a loop and its raw count, the seven columns `read_bedpe` reads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def write_cooler(slice_path: Path, chrom: str, out: Path) -> int:
    """Write the slice as a cooler of one chromosome. Returns the bin size."""
    import cooler

    z = np.load(slice_path)
    n, b, a = int(z["nbins"]), int(z["binsize"]), int(z["start"])
    starts = a + np.arange(n, dtype=np.int64) * b
    bins = pd.DataFrame({"chrom": chrom, "start": starts, "end": starts + b})
    pixels = pd.DataFrame({"bin1_id": z["bin1"], "bin2_id": z["bin2"], "count": z["count"].astype(np.int64)})
    pixels = pixels.sort_values(["bin1_id", "bin2_id"]).reset_index(drop=True)
    cooler.create_cooler(str(out), bins, pixels, ordered=True, dtypes={"count": np.int64})
    return b


def call(cool: Path, bedpe: Path, nproc: int) -> int:
    """Balance one cooler, call its loops and write them. Returns the number of loops."""
    import cooler
    import cooltools

    clr = cooler.Cooler(str(cool))
    cooler.balance_cooler(clr, store=True, cis_only=True, ignore_diags=2, mad_max=5)
    clr = cooler.Cooler(str(cool))
    b = clr.bins()[:]
    chrom = str(b["chrom"].iloc[0])
    view = pd.DataFrame({"chrom": [chrom], "start": [int(b["start"].min())], "end": [int(b["end"].max())], "name": [chrom]})
    expected = cooltools.expected_cis(clr, view_df=view, nproc=nproc)
    dots = cooltools.dots(clr, expected=expected, view_df=view, nproc=nproc)
    dots = dots.sort_values(["start1", "start2"])
    cols = ["chrom1", "start1", "end1", "chrom2", "start2", "end2", "count"]
    dots[cols].to_csv(bedpe, sep="\t", header=False, index=False)
    return len(dots)


def main() -> None:
    import cooler

    slice_path, chrom, prefix = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    fine = Path(f"{prefix}_tmp.cool")
    b = write_cooler(slice_path, chrom, fine)
    base = Path(f"{prefix}_{b}.cool")
    fine.rename(base)
    coarse = Path(f"{prefix}_{2 * b}.cool")
    cooler.coarsen_cooler(str(base), str(coarse), 2, chunksize=10_000_000)
    for cool in (base, coarse):
        bedpe = cool.with_suffix(".bedpe")
        n = call(cool, bedpe, nproc)
        print(f"{cool.name}: {n} loops -> {bedpe}", flush=True)


if __name__ == "__main__":
    main()
