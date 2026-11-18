"""Contact maps of a job, ICE balancing and the stratum adjusted correlation.

    python web/tests/test_contacts.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from _check import check, done
from fixture import make_data

from gnome3d_web.contacts import ice, loop_enrichment, median_bond, model_map, multimm_map, rank_agreement, rank_reach, scc


def battery_scc(A: np.ndarray, B: np.ndarray, max_sep_bins: int) -> float:
    """validation/metrics/hic.py _scc, copied: whole diagonals weighted by length and both spreads."""
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


def main() -> int:
    pos = [np.array([[0.0, 0, 0], [1.0, 0, 0], [9.0, 0, 0]])] * 2
    mids = np.array([1000, 3000, 70000])
    m = model_map(pos, mids, 0, 100000, 25000)
    check("two beads 1 unit apart in one bin count once per structure", m[0, 0] == 2 and m.sum() == 2, str(m[:3, :3]))

    rng = np.random.default_rng(1)
    raw = rng.random((30, 30)) + 0.1
    raw = raw + raw.T
    b = ice(raw)
    sums = b.sum(axis=1)
    check("ICE gives rows of equal sum", np.allclose(sums, sums.mean(), rtol=1e-3), str(sums[:4]))

    x = np.array([[1.0 / (1 + abs(i - j)) + 0.3 * rng.random() for j in range(60)] for i in range(60)])
    x = (x + x.T) / 2
    check("a map agrees with itself", abs(scc(x, x, 25000) - 1) < 1e-9)
    shuffled = x.copy()
    for k in range(1, 60):
        d = np.diagonal(x, k).copy()
        rng.shuffle(d)
        idx = np.arange(60 - k)
        shuffled[idx, idx + k] = d
        shuffled[idx + k, idx] = d
    check("a map with each diagonal shuffled agrees poorly", scc(x, shuffled, 25000) < 0.5, str(scc(x, shuffled, 25000)))
    check("an empty map has no agreement", scc(x, np.zeros_like(x), 25000) is None)
    check("a map ranks as itself", abs(rank_agreement(x, x, 25000) - 1) < 1e-9)
    check("the rank agreement ignores scale and any rising curve", abs(rank_agreement(x, 7 * x**3, 25000) - 1) < 1e-9)
    check("a map with each diagonal shuffled ranks poorly", abs(rank_agreement(x, shuffled, 25000)) < 0.3, str(rank_agreement(x, shuffled, 25000)))
    check("an empty map has no rank agreement", rank_agreement(x, np.zeros_like(x), 25000) is None)
    holed = x.copy()
    holed[20:30, :] = 0
    holed[:, 20:30] = 0
    kept = np.ones(60, dtype=bool)
    kept[20:30] = False
    check("bins left out do not count", abs(rank_agreement(x, holed, 25000, kept) - 1) < 1e-9 and rank_agreement(x, holed, 25000) < 1 - 1e-6)
    thin = x.copy()
    for k in range(9, 60):
        idx = np.arange(60 - k)
        thin[idx, idx + k] = thin[idx + k, idx] = 0
        thin[0, k] = thin[k, 0] = 1
    check("the rank agreement stops at the first distance under half filled", rank_reach(x, thin, 25000) == 8 * 25000 and abs(rank_agreement(x, thin, 25000) - 1) < 1e-9, str(rank_reach(x, thin, 25000)))
    check("a full map is read to its end", rank_reach(x, x, 25000) == 57 * 25000, str(rank_reach(x, x, 25000)))
    sparse = np.zeros((40, 40))
    other = np.zeros((40, 40))
    for k in range(1, 40):
        for i in range(40 - k):
            if rng.random() < 0.3:
                sparse[i, i + k] = sparse[i + k, i] = 1 + rng.integers(3)
            if rng.random() < 0.3:
                other[i, i + k] = other[i + k, i] = 1 + rng.integers(3)
    check("the agreement is the battery's on whole diagonals", abs(scc(sparse, other, 25000) - battery_scc(sparse, other, 80)) < 1e-12, f"{scc(sparse, other, 25000)} {battery_scc(sparse, other, 80)}")

    chain = [np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])]
    mm = multimm_map(chain, np.array([12_500, 37_500, 62_500]), 0, 75_000, 25_000)
    check("the MultiMM map is 1/(d+1)^3 of bin centroids in units of the median step", np.allclose(mm, [[1, 1 / 8, 1 / 27], [1 / 8, 1, 1 / 8], [1 / 27, 1 / 8, 1]]), str(mm))
    wide = multimm_map([rng.random((50, 3)) * 10 for _ in range(3)], np.arange(50) * 1000 + 500, 0, 50_000, 1000)
    check("the MultiMM map has no zero off the diagonal", bool((wide[~np.eye(50, dtype=bool)] > 0).all()))
    flat = np.ones((30, 30))
    loop_px = np.zeros((30, 30))
    loop_px[5, 15] = loop_px[15, 5] = 4
    raised = flat.copy()
    raised[5, 15] = raised[15, 5] = 3
    check("a map raised at the loop pixels has an enrichment above 1", abs(loop_enrichment(loop_px, raised) - 3) < 1e-9, str(loop_enrichment(loop_px, raised)))
    check("a map flat at the loop pixels has an enrichment of 1", abs(loop_enrichment(loop_px, flat) - 1) < 1e-9)
    check("the median bond of a chain", abs(median_bond([np.array([[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0], [4.0, 0, 0]])]) - 1.0) < 1e-12)

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        make_data(root / "data")
        os.environ.update(
            DATA_DIR=str(root / "data"), RESULTS_DIR=str(root / "results"), QUEUE="inline", FRONTEND_DIST=str(root / "missing"),
            ENGINE_CMD=f"{Path(__file__).resolve().parents[2] / '.venv' / 'bin' / 'python'} {Path(__file__).resolve().parent / 'fake_engine.py'}",
        )  # fmt: skip
        from fastapi.testclient import TestClient

        from gnome3d_web.app import create_app

        c = TestClient(create_app())
        url = c.post("/api/jobs", data={"cell_type": "GMX", "region": "chr1:0-3000000", "factors": "CTCF", "n_models": "2"}).json()["url"]
        r = c.get(f"/api{url}/contacts", params={"region": "chr1:0-3000000", "res": 25000, "upper": "hic", "lower": "model", "norm": "raw"}).json()
        n = len(r["bins"])
        check("the endpoint returns two square maps on the bins of the resolution", n == 120 and len(r["upper"]) == n and len(r["lower"][0]) == n and r["bins"][1] - r["bins"][0] == 25000, str(n))
        check("the halves are labelled", r["upper_label"].startswith("Hi-C") and "2 structures" in r["lower_label"] and "1/(d+1)" in r["lower_label"], str((r["upper_label"], r["lower_label"])))
        check("the agreement names the two halves it compares", "Hi-C of GMX" in r["scc_about"] and "1/(d+1)" in r["scc_about"] and "raw" in r["scc_about"], str(r.get("scc_about")))
        check("the rank agreement comes with what it compares", "rank" in r and (r["rank"] is None or -1 <= r["rank"] <= 1) and "ranked" in r["rank_about"] and "balanced" in r["rank_about"], str(r.get("rank_about")))
        counted = c.get(f"/api{url}/contacts", params={"region": "chr1:0-3000000", "res": 25000, "upper": "hic", "lower": "model_count", "norm": "raw"}).json()
        check("the counted model is still offered", "within" in counted["lower_label"], counted.get("lower_label", counted))
        battery = c.get(f"/api{url}/contacts", params={"region": "chr1:0-3000000", "res": 25000, "upper": "hic", "lower": "model_count", "norm": "ice"}).json()
        check("the agreement follows the halves shown", counted["scc"] != r["scc"] or "1/(d+1)" in r["scc_about"], f"{counted['scc']} {r['scc']}")
        check("Hi-C against the counted model is the battery's number, whatever the display", battery["scc"] == counted["scc"] and "validation battery" in battery["scc_about"], str(battery.get("scc_about")))
        lp = c.get(f"/api{url}/contacts", params={"region": "chr1:0-3000000", "res": 25000, "upper": "loops_ctcf", "lower": "model", "norm": "raw"}).json()
        up = np.array(lp["upper"])
        check("a loop half bins each CTCF loop at its anchors", up.sum() > 0 and "CTCF loops" in lp["upper_label"], lp.get("upper_label", lp))
        lp_ice = c.get(f"/api{url}/contacts", params={"region": "chr1:0-3000000", "res": 25000, "upper": "loops_ctcf", "lower": "model", "norm": "ice"}).json()
        check("a loop half is never balanced", lp_ice["upper"] == lp["upper"], "ice changed the loop half")
        check("with loops on one side the card has the enrichment", isinstance(lp.get("loop_enrichment"), float), str(lp.get("loop_enrichment")))

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
        check("a deep map is read once per resolution over the job's region and windows are cut from it", len(reads) == 1 and reads[0][1:3] == (0, 3000000), str(reads))
        check("the Hi-C half names the deep map", "4DN deep map" in deep["upper_label"] and "4DN deep map" in deep["scc_about"], deep["upper_label"])
        check("the agreement is a number or null", r["scc"] is None or -1 <= r["scc"] <= 1, str(r["scc"]))
        bad = c.get(f"/api{url}/contacts", params={"region": "chr1:0-3000000", "res": 2000, "upper": "hic", "lower": "model"})
        check("too many bins is a 400 naming a resolution that fits", bad.status_code == 400 and "5 kb" in bad.json()["error"], bad.text)
        out = c.get(f"/api{url}/contacts", params={"region": "chr1:3500000-4000000", "res": 25000, "upper": "hic", "lower": "model", "norm": "ice"})
        check("a window outside the job's region is a 400 naming the region", out.status_code == 400 and "chr1:0-3,000,000" in out.json()["error"], out.text)
        part = c.get(f"/api{url}/contacts", params={"region": "chr1:2900000-3400000", "res": 25000, "upper": "hic", "lower": "model", "norm": "ice"})
        check("a window reaching past the region is cut to it", part.status_code == 200 and part.json()["bins"][-1] < 3_000_000, part.text[:200])
        check("ICE of an empty map is an empty map", float(ice(np.zeros((4, 4))).sum()) == 0.0)
    return done()


if __name__ == "__main__":
    sys.exit(main())
