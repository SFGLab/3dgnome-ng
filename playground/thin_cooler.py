"""Split the counts of regions of a contact map into two independent halves.

    python playground/thin_cooler.py MCOOL OUT_DIR LABEL REGION [REGION ...]
    python playground/thin_cooler.py --weights MCOOL HALF_MCOOL
    python playground/thin_cooler.py --chromosomes MCOOL OUT_DIR LABEL CHROM [CHROM ...]

Each raw count of a 1 kb cell inside a region is split by a binomial draw at one half. The two
halves of a Poisson count are independent Poisson counts. Half `a` is the input of a run and half
`b` is the map its structures are scored against, which the run did not read. The halves are
`OUT_DIR/a/LABEL/<name>` and `OUT_DIR/b/LABEL/<name>`, each an mcool at 1, 2, 5, 10 and 25 kb that
holds the regions alone with the balancing weights of MCOOL. A half has the bias of the whole map
per bin. Weights fitted on the regions alone raise the bins at the ends of a region, which lose
their contacts beyond it. Half `a` also gets `LABEL_hic_25kb_singletons.bedpe`, its counts at
25 kb as the singletons of a run. The seed is fixed.

The second form gives an existing half the weights of MCOOL in place.

The third form splits whole chromosomes for the ensembles of a published result. It splits the
counts of the 5 kb cells between two bins of one chromosome, the resolution of the contact cells
of a run, and reads the map in parts of 50 million cells. The halves are
`OUT_DIR/a/LABEL.mcool` and `OUT_DIR/b/LABEL.mcool` at 5, 10 and 25 kb with the weights of MCOOL.
A run reads the singletons of production beside half `a`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cooler
import h5py
import numpy as np
import pandas as pd

RES = [1000, 2000, 5000, 10000, 25000]


WHOLE_RES = [5000, 10000, 25000]
PART = 50_000_000


def copy_weights(mcool: Path, half: Path, resolutions: list[int] = RES) -> None:
    """Writes the balancing weights of the map into the half at every resolution."""
    for res in resolutions:
        weight = cooler.Cooler(f"{mcool}::/resolutions/{res}").bins()[:]["weight"].to_numpy()
        with h5py.File(half, "r+") as f:
            group = f[f"resolutions/{res}/bins"]
            assert group["start"].shape[0] == weight.shape[0], f"bins differ at {res} bp"
            if "weight" in group:
                del group["weight"]
            group.create_dataset("weight", data=weight)


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
        copy_weights(mcool, final)
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


def thin_chromosomes(mcool: Path, out: Path, label: str, chroms: list[str], seed: int = 20261120) -> None:
    """Splits the counts within each chromosome of `chroms` at 5 kb into the two halves."""
    clr = cooler.Cooler(f"{mcool}::/resolutions/{WHOLE_RES[0]}")
    bins = clr.bins()[:][["chrom", "start", "end"]]
    order = [c for c in clr.chromnames if c in set(chroms)]
    assert len(order) == len(set(chroms)), f"not in the map: {sorted(set(chroms) - set(order))}"
    with clr.open("r") as f:
        offset = f["indexes/bin1_offset"][:]

    def parts(name: str):  # noqa: ANN202
        # The two calls draw the same numbers: one generator per chromosome over the same parts.
        for chrom in order:
            lo, hi = clr.extent(chrom)
            rng = np.random.default_rng([seed, clr.chromnames.index(chrom)])
            for p0 in range(int(offset[lo]), int(offset[hi]), PART):
                part = clr.pixels()[p0 : min(p0 + PART, int(offset[hi]))]
                part = part[part["bin2_id"] < hi]
                total = part["count"].to_numpy().astype(np.int64)
                first = rng.binomial(total, 0.5)
                part = part[["bin1_id", "bin2_id"]].assign(count=first if name == "a" else total - first)
                yield part[part["count"] > 0]

    for name in ("a", "b"):
        target = out / name
        target.mkdir(parents=True, exist_ok=True)
        base = target / f"{label}.5kb.cool"
        final = target / f"{label}.mcool"
        assert not base.exists() and not final.exists(), f"{base} or {final} exists"
        cooler.create_cooler(str(base), bins, parts(name), ordered=True, dtypes={"count": np.int32})
        cooler.zoomify_cooler(str(base), str(final), WHOLE_RES, chunksize=10_000_000, nproc=4)
        copy_weights(mcool, final, WHOLE_RES)
        for res in WHOLE_RES:
            n = int(cooler.Cooler(f"{final}::/resolutions/{res}").info["sum"])
            print(f"{name} {label} at {res} bp: {n:,} contacts -> {final}")


if __name__ == "__main__":
    if sys.argv[1] == "--weights":
        copy_weights(Path(sys.argv[2]), Path(sys.argv[3]))
    elif sys.argv[1] == "--chromosomes":
        thin_chromosomes(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5:])
    else:
        thin(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4:])
