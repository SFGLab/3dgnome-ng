"""The bin pairs of one region of a .mcool as a seven column BEDPE.

    gnome3d-ng extract --mcool sample.mcool --region chr1:11500000-12600000 --out region.bedpe

Writes the region's intra chromosomal pairs at one resolution, raw counts, one line per pair,
without the diagonal, the way the hosted contact maps are written. This is the file the
website takes as a BEDPE of the region, made on the computer that holds the .mcool.
`--contacts-per-mb` thins the counts to that density by drawing contacts without replacement,
which keeps every separation's share of the decay.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from gnome3d.types import I64Array


class ExtractError(ValueError):
    """A region or resolution the file does not hold."""


def _length(bp: int) -> str:
    return f"{bp / 1e6:g} Mb" if bp >= 1_000_000 else f"{bp / 1e3:g} kb"


def _words(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else f"{', '.join(xs[:-1])} and {xs[-1]}"


def parse_region(text: str) -> tuple[str, int, int]:
    """chr:start-end with optional thousands commas."""
    m = re.fullmatch(r"\s*([^:\s]+):([\d,]+)-([\d,]+)\s*", text)
    if not m:
        raise ExtractError(f"{text!r} is not chr:start-end")
    start, end = int(m[2].replace(",", "")), int(m[3].replace(",", ""))
    if end <= start:
        raise ExtractError(f"{text!r} ends before it starts")
    return m[1], start, end


def thin(counts: I64Array, target: int, seed: int) -> I64Array:
    """Counts reduced to a target total by drawing contacts without replacement. Counts at or
    under the target are returned as they are."""
    if int(counts.sum()) <= target:
        return counts
    rng = np.random.default_rng(seed)
    return np.asarray(rng.multivariate_hypergeometric(counts, target), dtype=np.int64)


def _open(mcool: str, res: int) -> Any:  # noqa: ANN401
    import cooler  # pyright: ignore[reportMissingTypeStubs]

    paths: list[str] = cooler.fileops.list_coolers(mcool)  # pyright: ignore
    have = sorted(int(p.rsplit("/", 1)[-1]) for p in paths if p.rsplit("/", 1)[-1].isdigit())
    if have:
        if res not in have:
            raise ExtractError(
                f"{Path(mcool).name} has no {_length(res)} resolution. It has {_words([_length(r) for r in have])}."
            )
        return cooler.Cooler(f"{mcool}::/resolutions/{res}")  # pyright: ignore
    c = cooler.Cooler(mcool)  # pyright: ignore
    size = int(c.binsize or 0)  # pyright: ignore
    if size != res:
        raise ExtractError(f"{Path(mcool).name} is binned at {_length(size)}, not {_length(res)}.")
    return c


def extract(
    mcool: str,
    region: str,
    out: Path,
    res: int = 25_000,
    contacts_per_mb: float = 0.0,
    seed: int = 0,
) -> int:
    """Writes the region's off diagonal bin pairs as BEDPE and returns the number of lines.

    @param mcool a .mcool or single resolution .cool file
    @param region chr:start-end, the chromosome named with or without chr as the file has it
    @param res the bin size in bp, which the file must hold
    @param contacts_per_mb thin to this many contacts per Mb of the region, 0 keeps every contact
    @param seed the seed of the thinning draw
    @raises ExtractError naming a chromosome or resolution the file lacks
    """
    chrom, start, end = parse_region(region)
    c = _open(mcool, res)
    names: list[str] = [str(n) for n in c.chromnames]
    bare = chrom.removeprefix("chr")
    name = next((n for n in names if n in (chrom, bare, f"chr{bare}")), None)
    if name is None:
        raise ExtractError(
            f"{Path(mcool).name} has no chromosome {chrom}. It has {_words(names[:24])}."
        )
    span = f"{name}:{start}-{end}"
    pix = c.matrix(balance=False, as_pixels=True, join=True).fetch(span)
    pix = pix[(pix["start1"] >= start) & (pix["end2"] <= end) & (pix["start1"] != pix["start2"])]
    counts = np.rint(pix["count"].to_numpy()).astype(np.int64)
    if contacts_per_mb > 0:
        counts = thin(counts, round(contacts_per_mb * (end - start) / 1e6), seed)
    s1 = pix["start1"].to_numpy()
    e1 = pix["end1"].to_numpy()
    s2 = pix["start2"].to_numpy()
    e2 = pix["end2"].to_numpy()
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for a, b, x, y, k in zip(s1, e1, s2, e2, counts, strict=True):
            if k > 0:
                f.write(f"{chrom}\t{a}\t{b}\t{chrom}\t{x}\t{y}\t{k}\n")
                n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="gnome3d-ng extract",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mcool", required=True, help="a .mcool or .cool file")
    ap.add_argument("--region", required=True, help="chr:start-end")
    ap.add_argument("--res", type=int, default=25_000, help="bin size in bp, default 25000")
    ap.add_argument(
        "--contacts-per-mb",
        type=float,
        default=0.0,
        help="thin to this many contacts per Mb, 0 keeps every contact",
    )
    ap.add_argument("--seed", type=int, default=0, help="seed of the thinning draw")
    ap.add_argument("--out", required=True, help="the BEDPE to write")
    a = ap.parse_args(argv)
    try:
        n = extract(a.mcool, a.region, Path(a.out), a.res, a.contacts_per_mb, a.seed)
    except ExtractError as e:
        print(f"gnome3d-ng extract: {e}", file=sys.stderr)
        return 2
    print(f"{n:,} bin pairs written to {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
