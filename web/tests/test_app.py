"""The app's routes with the inline queue and the stand in engine.

    python web/tests/test_app.py
"""

from __future__ import annotations

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
            ENGINE_CMD=f"{Path(__file__).resolve().parents[2] / '.venv' / 'bin' / 'python'} {Path(__file__).resolve().parent / 'fake_engine.py'}",
        )
        os.environ.pop("HICDIFFUSION_DIR", None)
        from fastapi.testclient import TestClient

        from gnome3d_web.app import create_app

        c = TestClient(create_app())
        r = c.get("/model/classic")
        check("the form lists the cell type and the retention period", r.status_code == 200 and "GMX, lymphoblastoid" in r.text and "14 days" in r.text)
        form = {"cell_type": "GMX", "region": "chr1:0-3000000", "factors": "CTCF", "n_models": "2"}
        check("the form states the 20 Mb limit", "up to 20 Mb" in c.get("/model/classic").text)
        r = c.post("/jobs", data={**form, "region": "chr1:0-25000000"})
        check("a region over 20 Mb is a form error", r.status_code == 400 and "20,000,000" in r.text)
        os.environ["MAX_REGION_MB"] = "2.5"
        small = TestClient(create_app())
        r = small.post("/jobs", data={**form, "region": "chr1:0-3000000"})
        check("MAX_REGION_MB sets the limit", r.status_code == 400 and "2,500,000" in r.text and "up to 2.5 Mb" in small.get("/model/classic").text)
        del os.environ["MAX_REGION_MB"]
        r = c.post("/jobs", data={**form, "hicdiffusion": "on"})
        check("HiCDiffusion on a region over 2 Mb is a form error naming the limit", r.status_code == 400 and "2,000,000" in r.text and "HICDIFFUSION_DIR" not in r.text)
        r = c.post("/jobs", data={**form, "region": "chr1:0-2000000", "hicdiffusion": "on"})
        check("HiCDiffusion without its setup is a form error naming it", r.status_code == 400 and "HICDIFFUSION_DIR" in r.text)
        r = c.post("/jobs", data=form, follow_redirects=False)
        check("a valid request redirects to its job", r.status_code == 303 and r.headers["location"].startswith("/jobs/"))
        job = r.headers["location"]
        page = c.get(job)
        check("the job page shows the finished job with its viewer", page.status_code == 200 and "done" in page.text and "ngl" in page.text.lower())
        st = c.get("/api" + job).json()
        check("the status API reports done", st["state"] == "done", str(st)[:120])
        r2 = c.post("/jobs", data=form, follow_redirects=False)
        check("the same request finds the same job", r2.headers["location"] == job)
        names = c.get("/api" + job + "/files").json()
        check("the files API lists the structures and the table", "ep.tsv" in names and any(n.endswith(".cif") for n in names), str(names))
        check("a listed file downloads", c.get(job + "/files/ep.tsv").status_code == 200)
        cif = next(n for n in names if n.endswith(".cif"))
        view = c.get(job + "/view/" + cif.replace(".cif", ".json"))
        payload = view.json() if view.status_code == 200 else {}
        check(
            "the viewer gets the smoothed chain and boxes around the marked elements",
            view.status_code == 200 and len(payload.get("chain", [])) > 100 and {b["kind"] for b in payload.get("boxes", [])} == {"promoter", "enhancer"},
            f"{view.status_code} {len(payload.get('chain', []))} points {len(payload.get('boxes', []))} boxes",
        )
        page = c.get(job).text
        check("the gene table shows distances to two decimals", "<td class=\"num\">" in page and ".0000" not in page)
        check("a path outside the job is refused", c.get(job + "/files/..%2F..%2Fcatalog.json").status_code == 404)
        vcf = (root / "data" / "svs" / "demo.vcf.gz").read_bytes()
        r = c.post("/jobs", data={**form, "sample": "P1"}, files={"vcf": ("mine.vcf.gz", vcf)}, follow_redirects=False)
        loc = r.headers.get("location", "")
        check("an upload makes a private job behind a secret link", r.status_code == 303 and loc.startswith("/p/") and len(loc) > 20, loc)
        check("the private job is not under the shared jobs", c.get("/jobs/" + loc.split("/")[-1]).status_code == 404)
    return done()


if __name__ == "__main__":
    sys.exit(main())
