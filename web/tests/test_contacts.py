"""Contact maps of a job, ICE balancing and the stratum adjusted correlation.

    pytest web/tests/test_contacts.py
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fixture import make_data

from gnome3d_web.contacts import (
    MAX_BINS,
    bins_for,
    coarsest_fit,
    hic_map,
    ice,
    loop_enrichment,
    loop_map,
    median_bond,
    model_map,
    multimm_map,
    rank_agreement,
    rank_reach,
    scc,
    window_map,
)


def square(once: list[list[float]]) -> np.ndarray:
    """The whole map of an answer, which holds each pair of bins once: row i from column i on."""
    n = len(once)
    m = np.zeros((n, n))
    for i, row in enumerate(once):
        m[i, i:] = row
        m[i:, i] = row
    return m


def battery_scc(A: np.ndarray, B: np.ndarray, max_sep_bins: int) -> float:
    """legacy/validation/metrics/hic.py _scc, copied: whole diagonals weighted by length and both spreads."""
    num = den = 0.0
    for dd in range(1, min(max_sep_bins, A.shape[0] - 1) + 1):
        a, b = np.diagonal(A, dd), np.diagonal(B, dd)
        if a.size < 2 or a.std() < 1e-12 or b.std() < 1e-12:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        w = a.size * a.std() * b.std()
        num += w * r
        den += w
    return num / den if den > 0 else float("nan")


def test_a_map_on_bins_of_its_own_keeps_its_total_on_a_coarser_grid() -> None:
    small = np.arange(16.0).reshape(4, 4)
    assert abs(window_map(small, 0, 10, 0, 40, 20).sum() - small.sum()) < 1e-9, "a map on bins of its own keeps its total on a coarser grid"
    assert window_map(small, 0, 10, 0, 40, 20)[0, 1] == small[:2, 2:].sum(), "a coarser bin holds the bins inside it"
    fine = window_map(small, 0, 10, 0, 40, 5)
    assert fine.shape == (8, 8) and fine[0, 2] == small[0, 1] / 4, "a finer bin holds its share of the bin around it: " + str(fine[0, :4])
    assert abs(window_map(small, 0, 10, 0, 45, 15)[0, 0] - (small[0, 0] + 0.5 * small[0, 1] + 0.5 * small[1, 0] + 0.25 * small[1, 1])) < 1e-9, "a bin across two bins of the map takes a part of each"
    assert window_map(small, 100, 10, 0, 50, 10).sum() == 0, "the grid outside the map is empty"


def test_two_beads_1_unit_apart_in_one_bin_count_once_per_structure() -> None:
    pos = [np.array([[0.0, 0, 0], [1.0, 0, 0], [9.0, 0, 0]])] * 2
    mids = np.array([1000, 3000, 70000])
    m = model_map(pos, mids, 0, 100000, 25000)
    assert m[0, 0] == 2 and m.sum() == 2, "two beads 1 unit apart in one bin count once per structure: " + str(m[:3, :3])


@pytest.fixture(scope="module")
def sums_and_rng() -> SimpleNamespace:
    rng = np.random.default_rng(1)
    raw = rng.random((30, 30)) + 0.1
    raw = raw + raw.T
    b = ice(raw)
    sums = b.sum(axis=1)
    return SimpleNamespace(sums=sums, rng=rng)


def test_ice_gives_rows_of_equal_sum(sums_and_rng: SimpleNamespace) -> None:
    sums = sums_and_rng.sums
    assert np.allclose(sums, sums.mean(), rtol=1e-3), "ICE gives rows of equal sum: " + str(sums[:4])


def test_a_map_agrees_with_itself(sums_and_rng: SimpleNamespace) -> None:
    rng = sums_and_rng.rng
    x = np.array([[1.0 / (1 + abs(i - j)) + 0.3 * rng.random() for j in range(60)] for i in range(60)])
    x = (x + x.T) / 2
    assert abs(scc(x, x, 25000) - 1) < 1e-9, "a map agrees with itself"

    shuffled = x.copy()
    for k in range(1, 60):
        d = np.diagonal(x, k).copy()
        rng.shuffle(d)
        idx = np.arange(60 - k)
        shuffled[idx, idx + k] = d
        shuffled[idx + k, idx] = d
    assert scc(x, shuffled, 25000) < 0.5, "a map with each diagonal shuffled agrees poorly: " + str(scc(x, shuffled, 25000))
    assert scc(x, np.zeros_like(x), 25000) is None, "an empty map has no agreement"
    assert abs(rank_agreement(x, x, 25000) - 1) < 1e-9, "a map ranks as itself"
    assert abs(rank_agreement(x, 7 * x**3, 25000) - 1) < 1e-9, "the rank agreement ignores scale and any rising curve"
    assert abs(rank_agreement(x, shuffled, 25000)) < 0.3, "a map with each diagonal shuffled ranks poorly: " + str(rank_agreement(x, shuffled, 25000))
    assert rank_agreement(x, np.zeros_like(x), 25000) is None, "an empty map has no rank agreement"

    holed = x.copy()
    holed[20:30, :] = 0
    holed[:, 20:30] = 0
    kept = np.ones(60, dtype=bool)
    kept[20:30] = False
    assert abs(rank_agreement(x, holed, 25000, kept) - 1) < 1e-9 and rank_agreement(x, holed, 25000) < 1 - 1e-6, "bins left out do not count"

    thin = x.copy()
    for k in range(9, 60):
        idx = np.arange(60 - k)
        thin[idx, idx + k] = thin[idx + k, idx] = 0
        thin[0, k] = thin[k, 0] = 1
    assert rank_reach(x, thin, 25000) == 8 * 25000 and abs(rank_agreement(x, thin, 25000) - 1) < 1e-9, "the rank agreement stops at the first distance under half filled: " + str(rank_reach(x, thin, 25000))
    assert rank_reach(x, x, 25000) == 57 * 25000, "a full map is read to its end: " + str(rank_reach(x, x, 25000))

    sparse = np.zeros((40, 40))
    other = np.zeros((40, 40))
    for k in range(1, 40):
        for i in range(40 - k):
            if rng.random() < 0.3:
                sparse[i, i + k] = sparse[i + k, i] = 1 + rng.integers(3)
            if rng.random() < 0.3:
                other[i, i + k] = other[i + k, i] = 1 + rng.integers(3)
    assert abs(scc(sparse, other, 25000) - battery_scc(sparse, other, 80)) < 1e-12, "the agreement is the battery's on whole diagonals: " + f"{scc(sparse, other, 25000)} {battery_scc(sparse, other, 80)}"


def test_the_multimm_map_is_1_d_1_3_of_bin_centroids_in_units_of_the_median(sums_and_rng: SimpleNamespace) -> None:
    rng = sums_and_rng.rng
    chain = [np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])]
    mm = multimm_map(chain, np.array([12_500, 37_500, 62_500]), 0, 75_000, 25_000)
    assert np.allclose(mm, [[1, 1 / 8, 1 / 27], [1 / 8, 1, 1 / 8], [1 / 27, 1 / 8, 1]]), "the MultiMM map is 1/(d+1)^3 of bin centroids in units of the median step: " + str(mm)
    wide = multimm_map([rng.random((50, 3)) * 10 for _ in range(3)], np.arange(50) * 1000 + 500, 0, 50_000, 1000)
    assert bool((wide[~np.eye(50, dtype=bool)] > 0).all()), "the MultiMM map has no zero off the diagonal"
    flat = np.ones((30, 30))
    loop_px = np.zeros((30, 30))
    loop_px[5, 15] = loop_px[15, 5] = 4
    raised = flat.copy()
    raised[5, 15] = raised[15, 5] = 3
    assert abs(loop_enrichment(loop_px, raised) - 3) < 1e-9, "a map raised at the loop pixels has an enrichment above 1: " + str(loop_enrichment(loop_px, raised))
    assert abs(loop_enrichment(loop_px, flat) - 1) < 1e-9, "a map flat at the loop pixels has an enrichment of 1"
    assert abs(median_bond([np.array([[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0], [4.0, 0, 0]])]) - 1.0) < 1e-12, "the median bond of a chain"


def test_every_kind_of_map_is_symmetric(sums_and_rng: SimpleNamespace) -> None:
    rng = np.random.default_rng(3)
    pos = [rng.normal(size=(40, 3)) * 3 for _ in range(3)]
    mids = np.arange(40) * 25_000 + 12_500
    sing = pd.DataFrame({0: "chr1", 1: rng.integers(0, 1_000_000, 300), 2: 0, 3: "chr1", 4: rng.integers(0, 1_000_000, 300), 5: 0, 6: rng.integers(1, 9, 300)})
    half = rng.random((30, 30))
    maps = {
        "the model as 1/(d+1)^3": multimm_map(pos, mids, 0, 1_000_000, 50_000),
        "the model as counted contacts": model_map(pos, mids, 0, 1_000_000, 50_000),
        "the contacts of a BEDPE": hic_map(sing, "chr1", 0, 1_000_000, 50_000),
        "the loops": loop_map([(120_000, 480_000, 7), (300_000, 910_000, 3)], 0, 1_000_000, 50_000),
        "a map on bins of its own": window_map(half + half.T, 0, 8_192, 0, 240_000, 25_000),
    }
    for name, m in maps.items():
        assert m.sum() > 0 and np.allclose(m, m.T), f"{name} is symmetric and one half of it is the whole map"
        assert np.allclose(ice(m), ice(m).T), f"{name} stays symmetric when balanced"


def test_the_longest_job_fits_a_map_at_25_kb() -> None:
    longest = 20_000_000
    # A start off the grid of the bins adds a bin, and insertions add their bases to the region.
    for start, extra in ((120_184_079, 0), (120_184_079, 5_253), (0, 0), (37_615_000, 500_000)):
        n = len(bins_for(start, start + longest + extra, 25_000))
        assert n <= MAX_BINS, f"a job of {longest + extra:,} bp from {start:,} has {n} bins at 25 kb, over {MAX_BINS}"
        assert coarsest_fit(start, start + longest + extra) <= 25_000, "the finest fit of the longest job is 25 kb or finer"
    assert coarsest_fit(0, 40_000_000) == 50_000, "a window of 40 Mb fits at 50 kb"


def test_the_endpoint_returns_two_square_maps_on_the_bins_of_the_resolution(tmp_path: Path) -> None:
    d = str(tmp_path)
    root = Path(d)
    make_data(root / "data")
    os.environ.update(
        DATA_DIR=str(root / "data"), RESULTS_DIR=str(root / "results"), QUEUE="inline", FRONTEND_DIST=str(root / "missing"),
        ENGINE_CMD=f"{sys.executable} {Path(__file__).resolve().parent / 'fake_engine.py'}",
    )  # fmt: skip
    from fastapi.testclient import TestClient

    from gnome3d_web.app import create_app

    c = TestClient(create_app())
    url = c.post("/api/jobs", data={"cell_type": "GMX", "region": "chr1:0-3000000", "factors": "CTCF", "n_models": "2"}).json()["url"]
    r = c.get(f"/api{url}/contacts", params={"region": "chr1:0-3000000", "res": 25000, "upper": "hic", "lower": "model", "norm": "raw"}).json()
    n = len(r["bins"])
    assert n == 120 and len(r["upper"]) == n and len(r["lower"][0]) == n and r["bins"][1] - r["bins"][0] == 25000, "the endpoint returns two square maps on the bins of the resolution: " + str(n)
    assert [len(row) for row in r["upper"]] == list(range(n, 0, -1)) and [len(row) for row in r["lower"]] == list(range(n, 0, -1)), "each map holds each pair of bins once, row i from column i on"
    assert r["upper_label"].startswith("Hi-C") and "2 structures" in r["lower_label"] and "1/(d+1)" in r["lower_label"], "the halves are labelled: " + str((r["upper_label"], r["lower_label"]))
    assert "Hi-C of GMX" in r["scc_about"] and "1/(d+1)" in r["scc_about"] and "raw" in r["scc_about"], "the agreement names the two halves it compares: " + str(r.get("scc_about"))
    assert "rank" in r and (r["rank"] is None or -1 <= r["rank"] <= 1) and "ranked" in r["rank_about"] and "balanced" in r["rank_about"] and isinstance(r["rank_reach"], int), "the rank agreement comes with what it compares: " + str(r.get("rank_about"))

    counted = c.get(f"/api{url}/contacts", params={"region": "chr1:0-3000000", "res": 25000, "upper": "hic", "lower": "model_count", "norm": "raw"}).json()
    assert "within" in counted["lower_label"], "the counted model is still offered: " + str(counted.get("lower_label", counted))

    battery = c.get(f"/api{url}/contacts", params={"region": "chr1:0-3000000", "res": 25000, "upper": "hic", "lower": "model_count", "norm": "ice"}).json()
    assert counted["scc"] != r["scc"] or "1/(d+1)" in r["scc_about"], "the agreement follows the halves shown: " + f"{counted['scc']} {r['scc']}"
    assert battery["scc"] == counted["scc"] and "validation battery" in battery["scc_about"], "Hi-C against the counted model is the battery's number, whatever the display: " + str(battery.get("scc_about"))

    lp = c.get(f"/api{url}/contacts", params={"region": "chr1:0-3000000", "res": 25000, "upper": "loops_ctcf", "lower": "model", "norm": "raw"}).json()
    up = square(lp["upper"])
    assert up.sum() > 0 and "CTCF loops" in lp["upper_label"], "a loop half bins each CTCF loop at its anchors: " + str(lp.get("upper_label", lp))

    lp_ice = c.get(f"/api{url}/contacts", params={"region": "chr1:0-3000000", "res": 25000, "upper": "loops_ctcf", "lower": "model", "norm": "ice"}).json()
    assert lp_ice["upper"] == lp["upper"], "a loop half is never balanced: " + "ice changed the loop half"
    assert isinstance(lp.get("loop_enrichment"), float), "with loops on one side the card has the enrichment: " + str(lp.get("loop_enrichment"))

    from gnome3d_web import explore as ex

    reads: list[tuple] = []

    def fake(url: str, chrom: str, start: int, end: int, res: int, allow: bool):  # noqa: ANN202
        reads.append((url, start, end, res))
        b0 = np.arange(start - start % res, end, res)
        return b0[:-1], b0[1:], np.full(len(b0) - 1, 7)

    ex._read_deep = fake
    job = root / "results" / "jobs" / url.split("/")[-1]
    deep = ex.contact_maps(job, "chr1:0-3000000", 25000, "hic", "model", "raw", hic_url="https://example.org/deep.mcool", hic_name="4DN deep map")
    ex.contact_maps(job, "chr1:0-3000000", 25000, "hic", "model", "raw", hic_url="https://example.org/deep.mcool", hic_name="4DN deep map")
    ex.contact_maps(job, "chr1:500000-1500000", 25000, "hic", "model", "raw", hic_url="https://example.org/deep.mcool", hic_name="4DN deep map")
    assert len(reads) == 1 and reads[0][1:3] == (0, 3000000), "a deep map is read once per resolution over the job's region and windows are cut from it: " + str(reads)
    assert "4DN deep map" in deep["upper_label"] and "4DN deep map" in deep["scc_about"], "the Hi-C half names the deep map: " + str(deep["upper_label"])
    assert r["scc"] is None or -1 <= r["scc"] <= 1, "the agreement is a number or null: " + str(r["scc"])

    # The answer is gzipped for a client that takes gzip and plain for one that does not.
    ask = {"region": "chr1:0-3000000", "res": 25000, "upper": "hic", "lower": "model", "norm": "raw"}
    plain = c.get(f"/api{url}/contacts", params=ask, headers={"Accept-Encoding": "identity"})
    assert "content-encoding" not in plain.headers and plain.json() == r, "a client that takes no gzip gets the answer plain: " + str(dict(plain.headers))
    with c.stream("GET", f"/api{url}/contacts", params=ask, headers={"Accept-Encoding": "gzip, deflate"}) as zipped:
        sent = b"".join(zipped.iter_raw())
    assert zipped.headers["content-encoding"] == "gzip" and zipped.headers["vary"] == "Accept-Encoding", "a client that takes gzip gets gzip: " + str(dict(zipped.headers))
    assert zipped.headers["content-type"] == "application/json", "the answer is still JSON: " + zipped.headers["content-type"]
    assert json.loads(gzip.decompress(sent)) == r, "the gzipped answer holds the same maps"
    assert len(sent) < len(plain.content) / 2, f"the gzipped answer is under half of the plain one: {len(sent)} against {len(plain.content)}"

    bad = c.get(f"/api{url}/contacts", params={"region": "chr1:0-3000000", "res": 2000, "upper": "hic", "lower": "model"})
    assert bad.status_code == 400 and "5 kb" in bad.json()["error"], "too many bins is a 400 naming a resolution that fits: " + bad.text

    out = c.get(f"/api{url}/contacts", params={"region": "chr1:3500000-4000000", "res": 25000, "upper": "hic", "lower": "model", "norm": "ice"})
    assert out.status_code == 400 and "chr1:0-3,000,000" in out.json()["error"], "a window outside the job's region is a 400 naming the region: " + out.text

    part = c.get(f"/api{url}/contacts", params={"region": "chr1:2900000-3400000", "res": 25000, "upper": "hic", "lower": "model", "norm": "ice"})
    assert part.status_code == 200 and part.json()["bins"][-1] < 3_000_000, "a window reaching past the region is cut to it: " + part.text[:200]
    assert float(ice(np.zeros((4, 4))).sum()) == 0.0, "ICE of an empty map is an empty map"
