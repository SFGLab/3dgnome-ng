"""The routes of the app with the inline queue and the stand in engine."""

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
    r = c.get("/api/catalog")
    return SimpleNamespace(r=r, c=c, TestClient=TestClient, create_app=create_app, root=root)


def test_the_catalog_lists_the_cell_type(site: SimpleNamespace) -> None:
    r = site.r
    cell = r.json()["cell_types"].get("GMX", {})
    assert r.status_code == 200 and cell.get("description") == "lymphoblastoid", "the catalog lists the cell type with its description: " + str(cell)


@pytest.fixture(scope="module")
def form() -> Any:
    form = {"cell_type": "GMX", "region": "chr1:0-3000000", "factors": "CTCF", "n_models": "2"}
    return form


def test_the_catalog_states_the_20_mb_limit(site: SimpleNamespace) -> None:
    c = site.c
    assert c.get("/api/catalog").json()["max_region_mb"] == 20, "the catalog states the 20 Mb limit"


def test_a_region_over_20_mb_is_refused_at_its_field(form: Any, site: SimpleNamespace) -> None:
    c = site.c
    r = c.post("/api/jobs", data={**form, "region": "chr1:0-25000000"})
    assert r.status_code == 400 and "20,000,000" in r.json()["error"] and r.json()["field"] == "region", "a region over 20 Mb is refused at the region field: " + r.text


def test_max_structures_caps_the_structures_of_a_request(form: Any, site: SimpleNamespace) -> None:
    TestClient, create_app = site.TestClient, site.create_app
    os.environ["MAX_STRUCTURES"] = "1"
    try:
        small = TestClient(create_app())
        five = small.post("/api/jobs", data={**form, "n_models": "5"}).json()["url"]
        one = small.post("/api/jobs", data={**form, "n_models": "1"}).json()["url"]
    finally:
        del os.environ["MAX_STRUCTURES"]
    asked = small.get("/api" + five + "/summary").json()["request"]["n_models"]
    assert five == one and asked == 1, "a request for five structures on a site of one is the request for one: " + str((five, one, asked))


def test_max_region_mb_sets_the_limit(form: Any, site: SimpleNamespace) -> None:
    TestClient, create_app = site.TestClient, site.create_app
    os.environ["MAX_REGION_MB"] = "2.5"
    small = TestClient(create_app())
    r = small.post("/api/jobs", data={**form, "region": "chr1:0-3000000"})
    assert r.status_code == 400 and "2,500,000" in r.json()["error"] and small.get("/api/catalog").json()["max_region_mb"] == 2.5, "MAX_REGION_MB sets the limit"


def test_hicdiffusion_on_a_region_over_2_mb_is_refused_naming_the_limit(form: Any, site: SimpleNamespace) -> None:
    c = site.c
    del os.environ["MAX_REGION_MB"]
    r = c.post("/api/jobs", data={**form, "background": "hicdiffusion"})
    assert r.status_code == 400 and "2,000,000" in r.json()["error"] and "HICDIFFUSION_DIR" not in r.text, "HiCDiffusion on a region over 2 Mb is refused naming the limit: " + r.text


def test_hicdiffusion_without_its_setup_is_refused_naming_it(form: Any, site: SimpleNamespace) -> None:
    c = site.c
    r = c.post("/api/jobs", data={**form, "region": "chr1:0-2000000", "background": "hicdiffusion"})
    assert r.status_code == 400 and "HICDIFFUSION_DIR" in r.json()["error"], "HiCDiffusion without its setup is refused naming it: " + r.text


def test_a_valid_request_answers_with_its_job(form: Any, site: SimpleNamespace) -> None:
    c = site.c
    r = c.post("/api/jobs", data=form)
    assert r.status_code == 201 and r.json()["url"].startswith("/jobs/"), "a valid request answers with its job: " + r.text

    job = r.json()["url"]
    st = c.get("/api" + job).json()
    assert st["state"] == "done", "the status API reports done: " + str(st)[:120]

    r2 = c.post("/api/jobs", data=form)
    assert r2.json()["url"] == job, "the same request finds the same job"

    summary = c.get("/api" + job + "/summary").json()
    names = [f["name"] for f in summary["files"] + summary["models"]]
    assert "ep.tsv" in names and any(n.endswith(".cif") for n in names), "the summary lists the structures and the table: " + str(names)
    assert c.get(job + "/files/ep.tsv").status_code == 200, "a listed file downloads"

    assert c.get(job + "/files/..%2F..%2Fcatalog.json").status_code == 404, "a path outside the job is refused"
    # The site has no page rendered by the server and no form that posts to /jobs.
    for old in (job + "/classic", job + "/view/x.json"):
        assert c.get(old).status_code in (404, 503), f"{old} is no page of the server"
    assert c.post("/jobs", data=form).status_code in (404, 405), "the old form has no address to post to"


def test_an_upload_makes_a_private_job_behind_a_secret_link(form: Any, site: SimpleNamespace) -> None:
    root, c = site.root, site.c
    vcf = (root / "data" / "svs" / "demo.vcf.gz").read_bytes()
    upload = c.post("/api/uploads", data={"kind": "vcf"}, files={"file": ("mine.vcf.gz", vcf)}).json()["id"]
    r = c.post("/api/jobs", data={**form, "sample": "P1", "vcf_upload": upload})
    loc = r.json().get("url", "") if r.status_code == 201 else ""
    assert r.status_code == 201 and loc.startswith("/p/") and len(loc) > 20, "an upload makes a private job behind a secret link: " + r.text[:120]

    token = loc.split("/")[-1]
    assert c.get(f"/api/p/{token}/summary").status_code == 200 and c.get(f"/api/jobs/{token}/summary").status_code == 404, "the private job is not under the shared jobs"
