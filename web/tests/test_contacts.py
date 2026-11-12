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

from gnome3d_web.contacts import ice, model_map, scc


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
        check("the halves are labelled", r["upper_label"].startswith("Hi-C") and "2 structures" in r["lower_label"], str((r["upper_label"], r["lower_label"])))
        check("the agreement is a number or null", r["scc"] is None or -1 <= r["scc"] <= 1, str(r["scc"]))
        bad = c.get(f"/api{url}/contacts", params={"region": "chr1:0-3000000", "res": 2000, "upper": "hic", "lower": "model"})
        check("too many bins is a 400 naming a resolution that fits", bad.status_code == 400 and "5 kb" in bad.json()["error"], bad.text)
    return done()


if __name__ == "__main__":
    sys.exit(main())
