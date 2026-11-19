"""Unit checks for gnome3d-ng extract, the bin pairs of one region of a .mcool as BEDPE.

    python tests/unit/test_extract.py

The output is what the website takes as a BEDPE of the region: intra chromosomal pairs at the
asked resolution, one line per pair, without the diagonal, as the hosted maps are written.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gnome3d.extract import ExtractError, extract  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}   {detail if not cond else ''}")


def write_mcool(out: Path) -> pd.DataFrame:
    """A cooler at 5 and 25 kb on chr1 and chr2 with known pixels. Returns the 25 kb pixels."""
    import cooler

    rng = np.random.default_rng(3)
    chroms = {"chr1": 3_000_000, "chr2": 1_000_000}
    keep = None
    for res in (5_000, 25_000):
        bins = pd.concat(
            [pd.DataFrame({"chrom": c, "start": np.arange(0, n, res)}) for c, n in chroms.items()],
            ignore_index=True,
        )
        bins["end"] = np.minimum(bins["start"] + res, bins["chrom"].map(chroms))
        n = len(bins)
        i = rng.integers(0, n, 4000)
        j = rng.integers(0, n, 4000)
        pix = pd.DataFrame(
            {
                "bin1_id": np.minimum(i, j),
                "bin2_id": np.maximum(i, j),
                "count": rng.integers(1, 50, 4000),
            }
        )
        pix = pix.groupby(["bin1_id", "bin2_id"], as_index=False)["count"].sum()
        cooler.create_cooler(f"{out}::resolutions/{res}", bins, pix, ordered=True, mode="a")
        if res == 25_000:
            keep = pix.merge(bins.reset_index(names="bin1_id"), on="bin1_id").merge(
                bins.reset_index(names="bin2_id"), on="bin2_id", suffixes=("1", "2")
            )
    assert keep is not None
    return keep


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        mcool = root / "sample.mcool"
        pix = write_mcool(mcool)
        lo, hi = 500_000, 2_000_000
        want = pix[
            (pix.chrom1 == "chr1")
            & (pix.chrom2 == "chr1")
            & (pix.start1 >= lo)
            & (pix.end2 <= hi)
            & (pix.bin1_id != pix.bin2_id)
        ]

        out = root / "region.bedpe"
        n = extract(str(mcool), "chr1:500,000-2,000,000", out)
        got = pd.read_csv(out, sep="\t", header=None)
        check(
            "one line per off diagonal pair of the window",
            n == len(want) == len(got),
            f"{n} {len(want)} {len(got)}",
        )
        pairs = set(zip(got[1], got[4], got[6], strict=True))
        check(
            "positions and counts match the matrix",
            pairs == set(zip(want.start1, want.start2, want["count"], strict=True)),
        )
        check(
            "bins are 25 kb and on the region's chromosome",
            set(got[2] - got[1]) == {25_000}
            and set(got[0]) == {"chr1"}
            and set(got[3]) == {"chr1"},
        )

        thin = root / "thin.bedpe"
        extract(str(mcool), "chr1:500000-2000000", thin, contacts_per_mb=100.0)
        t = pd.read_csv(thin, sep="\t", header=None)
        merged = (
            t[[1, 4, 6]]
            .set_axis(["a", "b", "kept"], axis=1)
            .merge(got[[1, 4, 6]].set_axis(["a", "b", "all"], axis=1), on=["a", "b"])
        )
        check("--contacts-per-mb thins to that count", int(t[6].sum()) == 150, str(int(t[6].sum())))
        check(
            "thinning keeps every pair at or under its count",
            len(merged) == len(t) and bool((merged["kept"] <= merged["all"]).all()),
        )

        try:
            extract(str(mcool), "chr7:0-1000000", root / "x.bedpe")
            check("an unknown chromosome names the ones the file has", False)
        except ExtractError as e:
            check(
                "an unknown chromosome names the ones the file has",
                "chr7" in str(e) and "chr1 and chr2" in str(e),
                str(e),
            )
        try:
            extract(str(mcool), "chr1:0-1000000", root / "x.bedpe", res=10_000)
            check("a missing resolution names the ones the file has", False)
        except ExtractError as e:
            check(
                "a missing resolution names the ones the file has",
                "5 kb and 25 kb" in str(e),
                str(e),
            )

        cli = root / "cli.bedpe"
        r = subprocess.run(
            [sys.executable, "-m", "gnome3d.cli", "extract", "--mcool", str(mcool), "--region", "chr1:500000-2000000", "--out", str(cli)],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )  # fmt: skip
        check(
            "gnome3d-ng extract runs from the command line",
            r.returncode == 0 and cli.read_text() == out.read_text(),
            r.stderr[-400:],
        )
        r = subprocess.run(
            [sys.executable, "-m", "gnome3d.cli", "extract", "--mcool", str(mcool), "--region", "chr7:0-1000", "--out", str(root / "y.bedpe")],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )  # fmt: skip
        check(
            "a refusal exits 2 with its message",
            r.returncode == 2 and "chr7" in r.stderr,
            r.stderr[-400:],
        )

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
