"""Unit checks for `gnome3d extract`."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gnome3d.extract import ExtractError, extract

ROOT = Path(__file__).resolve().parents[2]


def write_mcool(out: Path) -> pd.DataFrame:
    """Writes a cooler at 5 and 25 kb on chr1 and chr2 with known pixels. Returns the 25 kb pixels."""
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


@pytest.fixture(scope="module", autouse=True)
def shared(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    d = str(tmp_path_factory.mktemp("work"))
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
    return SimpleNamespace(n=n, want=want, got=got, root=root, mcool=mcool, out=out)


def test_one_line_per_off_diagonal_pair_of_the_window(shared: SimpleNamespace) -> None:
    n, want, got = shared.n, shared.want, shared.got
    assert n == len(want) == len(got), (
        f"one line per off diagonal pair of the window: {n} {len(want)} {len(got)}"
    )
    pairs = set(zip(got[1], got[4], got[6], strict=True))
    assert pairs == set(zip(want.start1, want.start2, want["count"], strict=True)), (
        "positions and counts match the matrix"
    )
    assert (
        set(got[2] - got[1]) == {25_000} and set(got[0]) == {"chr1"} and set(got[3]) == {"chr1"}
    ), "bins are 25 kb and on the region's chromosome"


def test_contacts_per_mb_thins_to_that_count(shared: SimpleNamespace) -> None:
    root, mcool, got = shared.root, shared.mcool, shared.got
    thin = root / "thin.bedpe"
    extract(str(mcool), "chr1:500000-2000000", thin, contacts_per_mb=100.0)
    t = pd.read_csv(thin, sep="\t", header=None)
    merged = (
        t[[1, 4, 6]]
        .set_axis(["a", "b", "kept"], axis=1)
        .merge(got[[1, 4, 6]].set_axis(["a", "b", "all"], axis=1), on=["a", "b"])
    )
    assert int(t[6].sum()) == 150, "--contacts-per-mb thins to that count: " + str(int(t[6].sum()))
    assert len(merged) == len(t) and bool((merged["kept"] <= merged["all"]).all()), (
        "thinning keeps every pair at or under its count"
    )


def test_an_unknown_chromosome_names_the_ones_the_file_has(shared: SimpleNamespace) -> None:
    mcool, root = shared.mcool, shared.root
    with pytest.raises(ExtractError) as unknown:
        extract(str(mcool), "chr7:0-1000000", root / "x.bedpe")
    assert "chr7" in str(unknown.value) and "chr1 and chr2" in str(unknown.value)
    with pytest.raises(ExtractError) as missing:
        extract(str(mcool), "chr1:0-1000000", root / "x.bedpe", res=10_000)
    assert "5 kb and 25 kb" in str(missing.value)


def test_gnome3d_extract_runs_from_the_command_line(shared: SimpleNamespace) -> None:
    root, mcool, out = shared.root, shared.mcool, shared.out
    cli = root / "cli.bedpe"
    r = subprocess.run(
        [sys.executable, "-m", "gnome3d.cli", "extract", "--mcool", str(mcool), "--region", "chr1:500000-2000000", "--out", str(cli)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert r.returncode == 0 and cli.read_text() == out.read_text(), (
        "gnome3d extract runs from the command line: " + str(r.stderr[-400:])
    )
    r = subprocess.run(
        [sys.executable, "-m", "gnome3d.cli", "extract", "--mcool", str(mcool), "--region", "chr7:0-1000", "--out", str(root / "y.bedpe")],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert r.returncode == 2 and "chr7" in r.stderr, "a refusal exits 2 with its message: " + str(
        r.stderr[-400:]
    )
