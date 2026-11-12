"""The job data endpoints of the explorer.

    python web/tests/test_explore.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

from _check import check, done
from fixture import make_data


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        make_data(root / "data")
        os.environ.update(
            DATA_DIR=str(root / "data"),
            RESULTS_DIR=str(root / "results"),
            QUEUE="inline",
            FRONTEND_DIST=str(root / "missing"),
            ENGINE_CMD=f"{Path(__file__).resolve().parents[2] / '.venv' / 'bin' / 'python'} {Path(__file__).resolve().parent / 'fake_engine.py'}",
        )
        from fastapi.testclient import TestClient

        from gnome3d_web.app import create_app

        c = TestClient(create_app())
        form = {"cell_type": "GMX", "region": "chr1:0-3000000", "factors": "CTCF+RNAPII", "n_models": "3", "background": "hic"}
        three = "/api" + c.post("/api/jobs", data=form).json()["url"]
        one = "/api" + c.post("/api/jobs", data=form | {"n_models": "1"}).json()["url"]

        s = c.get(three + "/summary").json()
        names = {f["name"] for f in s["files"]}
        check("the summary has the state, structures and files", s["state"] == "done" and s["structures"] == 3 and {"ep.tsv", "marks.json", "engine.log"} <= names, str(s)[:200])
        check("the summary has the loop counts and the gene count", s["loops"] == {"CTCF": 14, "RNAPII": 1} and s["genes"] >= 1, str(s)[:300])
        check("a done job is not in the queue", s["ahead"] is None)

        st = c.get(three + "/structure/2").json()
        check("a structure has a chain with genomic positions", len(st["chain"]) > 100 and len(st["chain"][0]) == 4, str(st["chain"][:2]))
        check("a structure has its beads", len(st["beads"]) > 100 and st["beads"][0][5] in ("anchor", "subanchor"))
        anchored = [a for a in st["anchors"] if a["loops"]]
        check("anchors carry their loops", len(anchored) >= 2 and anchored[0]["loops"][0]["factor"] in ("CTCF", "RNAPII"), str(anchored[:1]))
        check("structure 4 of 3 is a 404", c.get(three + "/structure/4").status_code == 404)

        ann = c.get(three + "/annotations").json()
        check("annotations list the genes, enhancers and loops of the region", {g["name"] for g in ann["genes"]} == {"ALPHA", "BETA"} and len(ann["enhancers"]) == 2 and len(ann["loops"]) == 15, str(ann)[:300])

        p = c.get(three + "/picks").json()
        check("the picks are structure numbers", all(1 <= p[k] <= 3 for k in ("medoid", "compact", "random")), str(p))

        r = c.get(three + "/distance", params={"a": 1310000, "b": 2600000}).json()
        check("a distance has one value per structure, a mean and an SD", r["n"] == 3 and len(r["values"]) == 3 and r["sd"] > 0 and abs(r["mean"] - sum(r["values"]) / 3) < 1e-9, str(r))
        r1 = c.get(one + "/distance", params={"a": 1310000, "b": 2600000}).json()
        check("one structure has an SD of 0", r1["n"] == 1 and r1["sd"] == 0, str(r1))
        far = c.get(three + "/distance", params={"a": 3_200_000, "b": 2600000})
        check("a position far from every bead is a 400 with the distance", far.status_code == 400 and "kb from the nearest bead" in far.json()["error"], far.text)

        g = c.get(three + "/genes").json()
        check("gene rows carry statistics over the structures and the loop joining them", g and {"min", "q1", "median", "q3", "max", "joined_by", "n"} <= set(g[0]), str(g[:1]))

        loops = c.get(three + "/loops").json()
        check("the loops list factor, ends, span and PETs", len(loops) == 15 and {"factor", "a", "b", "span", "pets"} <= set(loops[0]), str(loops[:1]))
        check("the log is text", "wrote 3 structures" in c.get(three + "/log").text)
        z = zipfile.ZipFile(io.BytesIO(c.get(three + "/files.zip").content))
        check("the zip holds the job's files", "ep.tsv" in z.namelist() and any(n.endswith(".cif") for n in z.namelist()), str(z.namelist()[:5]))

        vcf = (root / "data" / "svs" / "demo.vcf.gz").read_bytes()
        priv = c.post("/api/jobs", data=form | {"n_models": "1"}, files={"vcf": ("mine.vcf.gz", vcf)}).json()["url"]
        token = priv.rsplit("/", 1)[1]
        check("a private job answers under /api/p", c.get(f"/api/p/{token}/summary").status_code == 200)
        check("a private job is not under /api/jobs", c.get(f"/api/jobs/{token}/summary").status_code == 404)
    return done()


if __name__ == "__main__":
    sys.exit(main())
