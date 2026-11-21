"""The files a visitor downloads from a job state no path of the server.

    pytest web/tests/test_downloads.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fixture import make_data

FORM = {"cell_type": "GMX", "region": "chr1:0-3000000", "factors": "CTCF", "n_models": "2"}


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    root = Path(str(tmp_path_factory.mktemp("work")))
    data = make_data(root / "data")
    (data / "expression").mkdir()
    (data / "expression" / "lines.tsv").write_text("gene\tlength\ta1\ta2\nALPHA\t1000\t40\t44\nBETA\t1200\t77\t79\nGAMMA\t900\t5\t9\n")
    catalog = json.loads((data / "catalog.json").read_text())
    catalog["expression"] = {"LINES": {"label": "Lines RNA-seq", "file": "expression/lines.tsv", "kind": "counts", "cell_types": {"GMX": ["a1", "a2"]}}}
    (data / "catalog.json").write_text(json.dumps(catalog))
    os.environ.update(
        DATA_DIR=str(data), RESULTS_DIR=str(root / "results"), QUEUE="inline", FRONTEND_DIST=str(root / "missing"),
        ENGINE_CMD=f"{sys.executable} {Path(__file__).resolve().parent / 'fake_engine.py'}",
    )  # fmt: skip
    from fastapi.testclient import TestClient

    from gnome3d_web.app import create_app

    c = TestClient(create_app())
    vcf = c.post("/api/uploads", data={"kind": "vcf"}, files={"file": ("demo.vcf.gz", (data / "svs" / "demo.vcf.gz").read_bytes())}).json()["id"]
    private = c.post("/api/jobs", data={**FORM, "sample": "P1", "haplotype": "0", "vcf_upload": vcf}).json()["url"]
    shared = c.post("/api/jobs", data={**FORM, "expression": "LINES"}).json()["url"]
    # An engine states the files it reads in its log, and a failed job keeps the end of the log
    # and the error in its status.
    job = root / "results" / "private" / private.split("/")[-1]
    with (job / "engine.log").open("a") as log:
        log.write(f"config {job}/inputs/config.ini\n")
    status = json.loads((job / "status.json").read_text())
    (job / "status.json").write_text(json.dumps({**status, "log_tail": f"config {job}/inputs/config.ini", "error": f"no file {job}/models"}))
    return SimpleNamespace(c=c, root=root, private=private, shared=shared, job=job)


def test_a_job_with_an_upload_is_private_and_finished(site: SimpleNamespace) -> None:
    c = site.c
    assert site.private.startswith("/p/") and site.shared.startswith("/jobs/"), "the job with a VCF is private and the other is shared: " + str((site.private, site.shared))
    assert c.get(f"/api{site.shared}/summary").json()["state"] == "done", "the shared job is done"
    assert str(site.root) in (site.job / "request.json").read_text(), "the request on disk has the place of the upload on the server"


@pytest.mark.parametrize("name", ["request.json", "status.json", "engine.log"])
def test_a_downloaded_file_of_a_private_job_has_no_path_of_the_server(site: SimpleNamespace, name: str) -> None:
    r = site.c.get(f"{site.private}/files/{name}")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"] and name in r.headers["content-disposition"], "the file is a download by its name: " + str(r.headers)
    assert str(site.root) not in r.text, f"{name} states no path of the server: " + r.text[:400]


def test_the_downloaded_request_is_the_one_the_page_states(site: SimpleNamespace) -> None:
    c = site.c
    request = c.get(f"{site.private}/files/request.json").json()
    assert request == c.get(f"/api{site.private}/summary").json()["request"] and "upload_vcf" not in request and request["sample"] == "P1", "the request is the summary's: " + str(request)
    log = c.get(f"{site.private}/files/engine.log").text
    assert "config inputs/config.ini" in log and log == c.get(f"/api{site.private}/log").text, "the log is the one of the page, its paths from the job's directory: " + log


def test_the_zip_of_a_private_job_has_no_path_of_the_server(site: SimpleNamespace) -> None:
    z = zipfile.ZipFile(io.BytesIO(site.c.get(f"/api{site.private}/files.zip").content))
    names = sorted(z.namelist())
    assert {"request.json", "status.json", "engine.log"} <= set(names) and not any(n.startswith(("upload.", "inputs/")) for n in names), "the zip has the three files and neither the upload nor the inputs: " + str(names)
    told = [n for n in names if not n.endswith(".cif") and str(site.root) in z.read(n).decode(errors="replace")]
    assert told == [], "no file of the zip states a path of the server: " + str(told)


def test_the_status_and_the_summary_of_a_job_have_no_path_of_the_server(site: SimpleNamespace) -> None:
    c = site.c
    status = c.get(f"/api{site.private}").json()
    assert status["error"] == "no file models" and status["log_tail"] == "config inputs/config.ini", "the status has its error and its log from the job's directory: " + str(status)
    summary = c.get(f"/api{site.private}/summary").json()
    assert summary["error"] == "no file models" and str(site.root) not in json.dumps(summary), "the summary too: " + str(summary["error"])


def test_a_request_with_hosted_expression_has_the_source_and_no_path(site: SimpleNamespace) -> None:
    c = site.c
    job = site.root / "results" / "jobs" / site.shared.split("/")[-1]
    on_disk = json.loads((job / "request.json").read_text())
    assert on_disk["expression"]["source"] == "LINES" and "path" not in on_disk["expression"] and str(site.root) not in json.dumps(on_disk), "the request on disk has the source by its id: " + str(on_disk["expression"])
    assert c.get(f"{site.shared}/files/request.json").json()["expression"] == "LINES", "the downloaded request has the source"
    assert c.get(f"/api{site.shared}/summary").json()["expression"]["samples"] == 2, "the job read the table of that source"



def test_the_summary_a_download_and_the_zip_have_the_same_files(site: SimpleNamespace) -> None:
    c = site.c
    summary = c.get(f"/api{site.shared}/summary").json()
    files, structures = [f["name"] for f in summary["files"]], [f["name"] for f in summary["models"]]
    assert {"expression.json", "expression.tsv", "request.json"} <= set(files) and structures, "the job lists its files and its structures: " + str(files)
    refused = [n for n in files + structures if c.get(f"{site.shared}/files/{n}").status_code != 200]
    assert refused == [], "every listed file downloads: " + str(refused)
    held = sorted(zipfile.ZipFile(io.BytesIO(c.get(f"/api{site.shared}/files.zip").content)).namelist())
    assert held == sorted(files + ["models/" + n for n in structures]), "the zip holds the listed files and no other: " + str(held)


@pytest.mark.parametrize("name", ["upload.vcf.gz", "explore_genes_v3.json", "config.ini", "status.json.tmp"])
def test_a_file_of_the_site_is_no_download(site: SimpleNamespace, name: str) -> None:
    (site.job / "explore_genes_v3.json").write_text("{}")
    (site.job / "status.json.tmp").write_text("{}")
    assert (site.job / "upload.vcf.gz").is_file(), "the job holds its upload"
    assert site.c.get(f"{site.private}/files/{name}").status_code == 404, f"{name} is not served"
    listed = [f["name"] for f in site.c.get(f"/api{site.private}/summary").json()["files"]]
    held = zipfile.ZipFile(io.BytesIO(site.c.get(f"/api{site.private}/files.zip").content)).namelist()
    assert name not in listed and name not in held, f"{name} is neither listed nor in the zip: " + str((listed, held))
