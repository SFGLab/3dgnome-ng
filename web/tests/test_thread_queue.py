"""The app with the queue of threads in the web process, the queue of the single container.

    pytest web/tests/test_thread_queue.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fixture import make_data

FORM = {"cell_type": "GMX", "region": "chr1:0-3000000", "factors": "CTCF", "n_models": "2"}


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    root = Path(str(tmp_path_factory.mktemp("work")))
    make_data(root / "data")
    os.environ.update(
        DATA_DIR=str(root / "data"), RESULTS_DIR=str(root / "results"), QUEUE="thread", GPUS="0", WORKERS_PER_GPU="1", FRONTEND_DIST=str(root / "missing"),
        ENGINE_CMD=f"{sys.executable} {Path(__file__).resolve().parent / 'fake_engine.py'}",
    )  # fmt: skip
    from fastapi.testclient import TestClient

    from gnome3d_web.app import create_app

    return SimpleNamespace(c=TestClient(create_app()), root=root)


def finished(c: Any, url: str) -> dict[str, Any]:
    """The summary of a job once it is done or failed, within a minute."""
    until = time.time() + 60
    while True:
        summary = c.get(f"/api{url}/summary").json()
        if summary["state"] in ("done", "failed") or time.time() > until:
            return summary
        time.sleep(0.05)


def test_a_shared_job_runs_in_a_thread_of_the_web_process(site: SimpleNamespace) -> None:
    c = site.c
    r = c.post("/api/jobs", data=FORM)
    assert r.status_code == 201 and r.json()["url"].startswith("/jobs/"), "the job is taken: " + r.text
    summary = finished(c, r.json()["url"])
    assert summary["state"] == "done" and summary["structures"] == 2, "the job finishes with its structures: " + str((summary["state"], summary["error"]))


def test_a_job_waits_behind_the_one_that_runs(site: SimpleNamespace) -> None:
    c = site.c
    os.environ["FAKE_ENGINE_SLEEP"] = "1.5"
    try:
        first = c.post("/api/jobs", data={**FORM, "region": "chr1:0-2000000"}).json()["url"]
        until = time.time() + 30
        while c.get(f"/api{first}/summary").json()["state"] == "queued" and time.time() < until:
            time.sleep(0.01)
        second = c.post("/api/jobs", data={**FORM, "region": "chr1:0-2500000"}).json()["url"]
        third = c.post("/api/jobs", data={**FORM, "region": "chr1:0-2600000"}).json()["url"]
        waiting = c.get(f"/api{third}/summary").json()
        assert waiting["state"] == "queued" and waiting["ahead"] == 1, "the third job waits with one job ahead of it: " + str((waiting["state"], waiting["ahead"]))
        states = [finished(c, url)["state"] for url in (first, second, third)]
    finally:
        del os.environ["FAKE_ENGINE_SLEEP"]
    assert states == ["done", "done", "done"], "the three jobs finish one after another: " + str(states)


def test_a_private_job_runs_in_a_thread_too(site: SimpleNamespace) -> None:
    c = site.c
    table = "gene\tmine\nALPHA\t12.5\nBETA\t3.0\n"
    up = c.post("/api/uploads", data={"kind": "expression", "cell_type": "GMX"}, files={"file": ("mine.tsv", io.BytesIO(table.encode()), "text/plain")})
    assert up.status_code == 201, "the table is taken: " + up.text
    r = c.post("/api/jobs", data={**FORM, "expression": "own", "expression_upload": up.json()["id"], "expression_columns": "[\"mine\"]", "expression_kind": "values"})
    assert r.status_code == 201 and r.json()["url"].startswith("/p/"), "a job with an own table is private: " + r.text
    summary = finished(c, r.json()["url"])
    assert summary["state"] == "done" and summary["expression"], "the private job finishes with its expression: " + str((summary["state"], summary["error"]))


def test_a_private_job_that_waits_has_its_status_on_disk(site: SimpleNamespace) -> None:
    c = site.c
    table = "gene\tmine\nALPHA\t12.5\nBETA\t3.0\n"
    up = c.post("/api/uploads", data={"kind": "expression", "cell_type": "GMX"}, files={"file": ("mine.tsv", io.BytesIO(table.encode()), "text/plain")}).json()["id"]
    os.environ["FAKE_ENGINE_SLEEP"] = "1.5"
    try:
        first = c.post("/api/jobs", data={**FORM, "region": "chr1:0-2100000"}).json()["url"]
        until = time.time() + 30
        while c.get(f"/api{first}/summary").json()["state"] == "queued" and time.time() < until:
            time.sleep(0.01)
        private = c.post("/api/jobs", data={**FORM, "expression": "own", "expression_upload": up, "expression_columns": "[\"mine\"]", "expression_kind": "values"}).json()["url"]
        on_disk = site.root / "results" / "private" / private.split("/")[-1] / "status.json"
        # A restart of the site finds the jobs to queue again by this file.
        assert on_disk.is_file() and json.loads(on_disk.read_text())["state"] == "queued", "the waiting private job has the status queued on disk"
        assert (on_disk.parent / "request.json").is_file(), "and its request"
        r = c.get(f"/api{private}")
        assert r.status_code == 200 and r.json()["state"] == "queued", "its status route answers: " + r.text[:200]
        states = [finished(c, url)["state"] for url in (first, private)]
    finally:
        del os.environ["FAKE_ENGINE_SLEEP"]
    assert states == ["done", "done"], "both jobs finish: " + str(states)


def test_the_summary_of_a_job_being_written_lists_no_half_written_file(tmp_path: Path) -> None:
    from gnome3d_web.results import views

    # A job writes a file under a second name and renames it. A page that asks for the summary at
    # that moment got an error from the file that was gone.
    (tmp_path / "status.json").write_text(json.dumps({"state": "running"}))
    (tmp_path / "status.json.tmp").write_text("{")
    assert [f["name"] for f in views.summary(tmp_path, None)["files"]] == ["status.json"], "the summary lists the files and not the one being written"  # type: ignore[union-attr]
    assert views._sizes([tmp_path / "gone.json"]) == [], "a file that is gone when its size is read is left out"
