"""The app's routes with the inline queue and the stand in engine.

    pytest web/tests/test_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fixture import make_data


@pytest.fixture(scope="module", autouse=True)
def site(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    d = str(tmp_path_factory.mktemp("work"))
    root = Path(d)
    make_data(root / "data")
    os.environ.update(
        DATA_DIR=str(root / "data"),
        RESULTS_DIR=str(root / "results"),
        QUEUE="inline",
        ENGINE_CMD=f"{sys.executable} {Path(__file__).resolve().parent / 'fake_engine.py'}",
    )
    os.environ.pop("HICDIFFUSION_DIR", None)
    from fastapi.testclient import TestClient

    from gnome3d_web.app import create_app

    c = TestClient(create_app())
    r = c.get("/model/classic")
    return SimpleNamespace(r=r, c=c, TestClient=TestClient, create_app=create_app, root=root)


def test_the_form_lists_the_cell_type_and_the_retention_period(site: SimpleNamespace) -> None:
    r = site.r
    assert r.status_code == 200 and "GMX, lymphoblastoid" in r.text and "14 days" in r.text, "the form lists the cell type and the retention period"


@pytest.fixture(scope="module")
def form() -> Any:
    form = {"cell_type": "GMX", "region": "chr1:0-3000000", "factors": "CTCF", "n_models": "2"}
    return form


def test_the_form_states_the_20_mb_limit(site: SimpleNamespace) -> None:
    c = site.c
    assert "up to 20 Mb" in c.get("/model/classic").text, "the form states the 20 Mb limit"


def test_a_region_over_20_mb_is_a_form_error(form: Any, site: SimpleNamespace) -> None:
    c = site.c
    r = c.post("/jobs", data={**form, "region": "chr1:0-25000000"})
    assert r.status_code == 400 and "20,000,000" in r.text, "a region over 20 Mb is a form error"


def test_max_region_mb_sets_the_limit(form: Any, site: SimpleNamespace) -> None:
    TestClient, create_app = site.TestClient, site.create_app
    os.environ["MAX_REGION_MB"] = "2.5"
    small = TestClient(create_app())
    r = small.post("/jobs", data={**form, "region": "chr1:0-3000000"})
    assert r.status_code == 400 and "2,500,000" in r.text and "up to 2.5 Mb" in small.get("/model/classic").text, "MAX_REGION_MB sets the limit"


def test_hicdiffusion_on_a_region_over_2_mb_is_a_form_error_naming_the_limit(form: Any, site: SimpleNamespace) -> None:
    c = site.c
    del os.environ["MAX_REGION_MB"]
    r = c.post("/jobs", data={**form, "hicdiffusion": "on"})
    assert r.status_code == 400 and "2,000,000" in r.text and "HICDIFFUSION_DIR" not in r.text, "HiCDiffusion on a region over 2 Mb is a form error naming the limit"


def test_hicdiffusion_without_its_setup_is_a_form_error_naming_it(form: Any, site: SimpleNamespace) -> None:
    c = site.c
    r = c.post("/jobs", data={**form, "region": "chr1:0-2000000", "hicdiffusion": "on"})
    assert r.status_code == 400 and "HICDIFFUSION_DIR" in r.text, "HiCDiffusion without its setup is a form error naming it"


def test_a_valid_request_redirects_to_its_job(form: Any, site: SimpleNamespace) -> None:
    c = site.c
    r = c.post("/jobs", data=form, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/jobs/"), "a valid request redirects to its job"

    job = r.headers["location"]
    page = c.get(job + "/classic")
    assert page.status_code == 200 and "done" in page.text and "ngl" in page.text.lower(), "the job page shows the finished job with its viewer"

    st = c.get("/api" + job).json()
    assert st["state"] == "done", "the status API reports done: " + str(st)[:120]

    r2 = c.post("/jobs", data=form, follow_redirects=False)
    assert r2.headers["location"] == job, "the same request finds the same job"

    names = c.get("/api" + job + "/files").json()
    assert "ep.tsv" in names and any(n.endswith(".cif") for n in names), "the files API lists the structures and the table: " + str(names)
    assert c.get(job + "/files/ep.tsv").status_code == 200, "a listed file downloads"

    cif = next(n for n in names if n.endswith(".cif"))
    view = c.get(job + "/view/" + cif.replace(".cif", ".json"))
    payload = view.json() if view.status_code == 200 else {}
    assert view.status_code == 200 and len(payload.get("chain", [])) > 100 and {b["kind"] for b in payload.get("boxes", [])} == {"promoter", "enhancer"}, "the viewer gets the smoothed chain and boxes around the marked elements: " + f"{view.status_code} {len(payload.get('chain', []))} points {len(payload.get('boxes', []))} boxes"

    page = c.get(job + "/classic").text
    assert "<td class=\"num\">" in page and ".0000" not in page, "the gene table shows distances to two decimals"
    assert c.get(job + "/files/..%2F..%2Fcatalog.json").status_code == 404, "a path outside the job is refused"


def test_an_upload_makes_a_private_job_behind_a_secret_link(form: Any, site: SimpleNamespace) -> None:
    root, c = site.root, site.c
    vcf = (root / "data" / "svs" / "demo.vcf.gz").read_bytes()
    r = c.post("/jobs", data={**form, "sample": "P1"}, files={"vcf": ("mine.vcf.gz", vcf)}, follow_redirects=False)
    loc = r.headers.get("location", "")
    assert r.status_code == 303 and loc.startswith("/p/") and len(loc) > 20, "an upload makes a private job behind a secret link: " + str(loc)

    token = loc.split("/")[-1]
    assert c.get(f"/jobs/{token}/classic").status_code == 404 and c.get(f"/api/jobs/{token}/summary").status_code == 404, "the private job is not under the shared jobs"
