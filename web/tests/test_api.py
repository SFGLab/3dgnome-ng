"""The JSON endpoints the frontend reads: catalog, region, loop preview, settings and job submission.

    python web/tests/test_api.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
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
        os.environ.pop("HICDIFFUSION_DIR", None)
        from fastapi.testclient import TestClient

        from gnome3d_web.app import create_app

        c = TestClient(create_app())
        cat = c.get("/api/catalog").json()
        gmx = cat["cell_types"]["GMX"]
        check("the catalog lists the cell type with its data", gmx["data"] == ["CTCF", "RNAPII", "Hi-C", "enhancers"], str(gmx))
        check("the catalog gives the limits and the factor choices", cat["max_structures"] == 100 and cat["max_region_mb"] == 20 and cat["factors"] == ["CTCF", "RNAPII", "CTCF+RNAPII"], str(cat)[:200])
        check("the catalog says HiCDiffusion is not set up", cat["hicdiffusion"] is False)
        check("the catalog says which call sets have a sample table", cat["sv_sets"]["DEMO"]["sample_table"] is True and cat["sv_sets"]["BARE"]["sample_table"] is False, str(cat["sv_sets"]))
        t = c.get("/api/sv_sets/DEMO/samples").json()
        check("a call set's sample table lists each sample with its parents", t["columns"][0] == "sample" and t["rows"][0] == ["P1", "female", "PUR", "AMR", "", "P2"], str(t))
        check("a call set without a table is a 404", c.get("/api/sv_sets/BARE/samples").status_code == 404 and c.get("/api/sv_sets/NOPE/samples").status_code == 404)

        r = c.get("/api/region", params={"cell": "GMX", "q": "alpha"}).json()
        check("a gene name in lower case resolves to its window", r["gene"] == "ALPHA" and r["chrom"] == "chr1" and r["end"] - r["start"] == 2_000_000, str(r))
        check("a region gets a label with its length", r["label"].endswith("2.0 Mb"), str(r))
        g = c.get("/api/genes", params={"cell": "GMX", "q": "al"}).json()
        check("typed letters give the genes that start with them", g == [{"name": "ALPHA", "chrom": "chr1", "tss": 1310000}], str(g))
        g = c.get("/api/genes", params={"cell": "GMX", "q": "ta"}).json()
        check("a gene that holds the text comes after those that start with it", [x["name"] for x in g] == ["BETA"], str(g))
        check("one letter gives no genes and an unknown cell type is a 404", c.get("/api/genes", params={"cell": "GMX", "q": "a"}).json() == [] and c.get("/api/genes", params={"cell": "NOPE", "q": "al"}).status_code == 404)
        r = c.get("/api/region", params={"cell": "GMX", "q": "chr1:0-25000000"})
        check("a region over the limit is a 400 naming it", r.status_code == 400 and "20,000,000" in r.json()["error"], r.text)

        p = c.get("/api/preview", params={"cell": "GMX", "region": "chr1:0-3000000"}).json()
        check("the preview lists the CTCF and RNAPII loops", p["counts"] == {"CTCF": 14, "RNAPII": 1} and len(p["loops"]["CTCF"]) == 14, str(p["counts"]))
        check("a preview loop is two anchor midpoints and a count", p["loops"]["RNAPII"] == [[1300500, 1500500, 4]], str(p["loops"]["RNAPII"]))
        check("the preview lists the genes", sorted(g["name"] for g in p["genes"]) == ["ALPHA", "BETA"], str(p["genes"]))

        sch = c.get("/api/settings", params={"cell": "GMX"}).json()
        check("the settings schema has the design's first group", sch[0]["name"] == "Distance law", str(sch[0])[:120])
        r = c.post("/api/settings/ini", files={"file": ("c.ini", b"[distance]\npolymer_exponent = 0.2\n[data]\nanchors = x\n")}).json()
        check("an uploaded ini is read into changes and refusals", r["overrides"] == {"distance.polymer_exponent": "0.2"} and len(r["refused"]) == 1, str(r))

        form = {"cell_type": "GMX", "region": "chr1:0-2000000", "factors": "CTCF", "n_models": "2", "background": "hic", "overrides": json.dumps({"distance.polymer_exponent": "0.3"})}
        r = c.post("/api/jobs", data=form)
        check("a job post returns 201 and its url", r.status_code == 201 and r.json()["url"].startswith("/jobs/"), r.text)
        again = c.post("/api/jobs", data=form)
        check("the same post finds the same job", again.json().get("url") == r.json().get("url"))
        st = json.loads((root / "results" / "jobs" / r.json()["url"].rsplit("/", 1)[1] / "request.json").read_text())
        check("the job keeps its setting changes", st["overrides"] == {"distance.polymer_exponent": "0.3"}, str(st))
        bad = c.post("/api/jobs", data=form | {"overrides": json.dumps({"distance.polymer_exponent": "9"})})
        check("a refused setting is a 400 on the overrides field", bad.status_code == 400 and bad.json()["field"] == "overrides", bad.text)
        bad = c.post("/api/jobs", data=form | {"background": "hicdiffusion"})
        check("HiCDiffusion without its setup is a 400 on the background field", bad.status_code == 400 and bad.json()["field"] == "background", bad.text)
        bad = c.post("/api/jobs", data=form | {"region": "nowhere"})
        check("an unknown region is a 400 on the region field", bad.status_code == 400 and bad.json()["field"] == "region", bad.text)
    return done()


if __name__ == "__main__":
    sys.exit(main())
