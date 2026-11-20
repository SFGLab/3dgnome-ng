"""Split the counts of regions of a contact map into two independent halves.

    python playground/thin_cooler.py MCOOL OUT_DIR LABEL REGION [REGION ...]

Each raw count of a 1 kb cell inside a region is split by a binomial draw at one half. The two
halves of a Poisson count are independent Poisson counts. Half `a` is the input of a run and half
`b` is the map its structures are scored against, which the run did not read. The halves are
`OUT_DIR/a/LABEL/<name>` and `OUT_DIR/b/LABEL/<name>`, each an mcool at 1, 2, 5, 10 and 25 kb that
holds the regions alone, balanced over the bins of the regions. Half `a` also gets
`LABEL_hic_25kb_singletons.bedpe`, its counts at 25 kb as the singletons of a run. The seed is
fixed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cooler
import numpy as np
import pandas as pd

RES = [1000, 2000, 5000, 10000, 25000]


def thin(mcool: Path, out: Path, label: str, regions: list[str], seed: int = 20261120) -> None:
    clr = cooler.Cooler(f"{mcool}::/resolutions/{RES[0]}")
    bins = clr.bins()[:][["chrom", "start", "end"]]
    parts = [
        clr.matrix(balance=False, as_pixels=True, join=False).fetch(region)[
            ["bin1_id", "bin2_id", "count"]
        ]
        for region in regions
    ]
    pixels = pd.concat(parts).sort_values(["bin1_id", "bin2_id"]).reset_index(drop=True)
    total = pixels["count"].to_numpy().astype(np.int64)
    first = np.random.default_rng(seed).binomial(total, 0.5)
    for name, count in (("a", first), ("b", total - first)):
        half = pixels[["bin1_id", "bin2_id"]].copy()
        half["count"] = count
        half = half[half["count"] > 0]
        target = out / name / label
        target.mkdir(parents=True, exist_ok=True)
        base = target / (mcool.stem + ".1kb.cool")
        cooler.create_cooler(str(base), bins, half, ordered=True, dtypes={"count": np.int32})
        final = target / mcool.name
        cooler.zoomify_cooler(str(base), str(final), RES, chunksize=10_000_000, nproc=4)
        for res in RES:
            cooler.balance_cooler(
                cooler.Cooler(f"{final}::/resolutions/{res}"), store=True, cis_only=True
            )
        print(f"{name}: {int(half['count'].sum()):,} contacts in {len(half):,} cells -> {final}")
        if name == "a":
            coarse = cooler.Cooler(f"{final}::/resolutions/25000")
            rows = pd.concat(
                [
                    coarse.matrix(balance=False, as_pixels=True, join=True).fetch(region)
                    for region in regions
                ]
            )
            rows = rows[(rows["count"] > 0) & (rows["start1"] != rows["start2"])]
            rows[["chrom1", "start1", "end1", "chrom2", "start2", "end2", "count"]].to_csv(
                target / f"{label}_hic_25kb_singletons.bedpe", sep="\t", header=False, index=False
            )


if __name__ == "__main__":
    thin(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4:])
