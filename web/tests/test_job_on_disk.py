"""Jobs a site already holds on its disk, read with the code of the day.

    pytest web/tests/test_job_on_disk.py

web/tests/job_on_disk has two job directories and the answers the site gave on them when they
were written, before the refactor of the site. One job has variants of a sample. The other has
expression and the files a visit of its page leaves. web/scripts/make_job_fixture.py wrote them.
"""

from __future__ import annotations

import io
import json
import math
import os
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fixture import make_data

HELD = Path(__file__).resolve().parent / "job_on_disk"
JOBS = ("with_variants", "with_expression")
REGION = "chr1:800000-2800000"
ROUTES = ("summary", "structure/1", "structure/2", "annotations", "picks", "distance?a=1310000&b=1490000", "genes", "loops", f"contacts?region={REGION}&res=25000")


def thin(value: Any) -> Any:
    """The value with every list of over 100 items cut to its length and each twentieth item."""
    if isinstance(value, dict):
        return {k: thin(v) for k, v in value.items()}
    if isinstance(value, list):
        return {"length": len(value), "each_twentieth": [thin(v) for v in value[::20]]} if len(value) > 100 else [thin(v) for v in value]
    return value


def difference(a: Any, b: Any, at: str = "") -> str:
    """The place of the first difference of two answers, empty when they agree. Numbers agree to six digits."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return f"{at}: keys {sorted(set(a) ^ set(b))}"
        return next((d for d in (difference(a[k], b[k], f"{at}/{k}") for k in sorted(a)) if d), "")
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{at}: {len(a)} and {len(b)} items"
        return next((d for d in (difference(x, y, f"{at}[{i}]") for i, (x, y) in enumerate(zip(a, b, strict=True))) if d), "")
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return "" if math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9) else f"{at}: {a} and {b}"
    return "" if a == b else f"{at}: {a!r} and {b!r}"


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    root = Path(str(tmp_path_factory.mktemp("work")))
    make_data(root / "data")
    jobs = root / "results" / "jobs"
    for name in JOBS:
        shutil.copytree(HELD / name, jobs / name)
    # A job of the time before overrides, the background, own data and expression were in a request.
    shutil.copytree(HELD / "with_variants", jobs / "old_request")
    old = {"cell_type": "GMX", "region": REGION, "factors": "CTCF+RNAPII", "sv_set": "DEMO", "sample": "P1", "haplotype": 0, "n_models": 2, "hicdiffusion": True, "upload_vcf": ""}
    (jobs / "old_request" / "request.json").write_text(json.dumps(old))
    os.environ.update(DATA_DIR=str(root / "data"), RESULTS_DIR=str(root / "results"), QUEUE="inline", FRONTEND_DIST=str(root / "missing"))
    from fastapi.testclient import TestClient

    from gnome3d_web.app import create_app

    said = json.loads((HELD / "answers.json").read_text())["answers"]
    return SimpleNamespace(c=TestClient(create_app()), said=said, jobs=jobs, old=old)


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("name", JOBS)
def test_a_job_from_disk_answers_as_it_did(site: SimpleNamespace, name: str, route: str) -> None:
    r = site.c.get(f"/api/jobs/{name}/{route}")
    assert r.status_code == 200, f"{route} of {name} answers: {r.status_code} {r.text[:300]}"
    now = thin(r.json())
    if route == "summary":
        now.pop("now")
    assert difference(site.said[name][route], now) == "", f"{route} of {name} is the answer the job gave when it was written"


@pytest.mark.parametrize("name", JOBS)
def test_the_status_the_log_and_the_zip_of_a_job_from_disk(site: SimpleNamespace, name: str) -> None:
    c, said = site.c, site.said[name]
    assert c.get(f"/api/jobs/{name}").json() == said["status"], "the status is the file on disk"
    assert c.get(f"/api/jobs/{name}/log").text == said["log"], "the log is the one of the job"
    names = sorted(zipfile.ZipFile(io.BytesIO(c.get(f"/api/jobs/{name}/files.zip").content)).namelist())
    assert names == said["zip"], "the zip has the files it had: " + str(names)


def test_a_file_of_a_job_from_disk_is_served(site: SimpleNamespace) -> None:
    c = site.c
    r = c.get("/jobs/with_variants/files/chr1_800000_2740000_s1.cif")
    assert r.status_code == 200 and r.content == (HELD / "with_variants" / "models" / "chr1_800000_2740000_s1.cif").read_bytes(), "a structure is served as it is on disk"
    assert c.get("/jobs/with_expression/files/expression.tsv").status_code == 200, "the expression table of a job is served"


def test_two_jobs_from_disk_compare_as_they_did(site: SimpleNamespace) -> None:
    c, said = site.c, site.said["compare"]
    pair = "a=/jobs/with_variants&b=/jobs/with_expression"
    for q in ("", "&shared=1"):
        now = c.get(f"/api/compare?{pair}{q}").json()
        assert difference(said[q], now) == "", "the comparison of the distances is the one of the time, with " + repr(q)
    # Both genes have the ratio 1 here. Rows of one ratio are in the order of their names.
    assert [r["gene"] for r in c.get(f"/api/compare?{pair}").json()["rows"]] == ["ALPHA", "BETA"], "rows of one ratio are ordered by gene"
    assert difference(said["gene"], c.get(f"/api/compare/gene?{pair}&gene=ALPHA").json()) == "", "the comparison of one gene is the one of the time"


def test_a_request_of_the_old_form_is_read(site: SimpleNamespace) -> None:
    from gnome3d_web.jobs import JobRequest

    req = JobRequest(**site.old)
    assert req.background == "hicdiffusion" and req.overrides == {} and req.own == {} and req.expression == {}, "the old flag is the predicted background and the later fields are empty: " + str(req)
    r = site.c.get("/api/jobs/old_request/summary")
    assert r.status_code == 200 and r.json()["request"]["hicdiffusion"] is True and r.json()["structures"] == 2, "the page of such a job has its summary: " + r.text[:300]
    assert site.c.get("/api/jobs/old_request/genes").status_code == 200, "and its genes"


def test_the_files_a_visit_left_in_a_job_are_read_again(site: SimpleNamespace) -> None:
    # The page of a job leaves what it computed in the job's directory. A site reads those files
    # of its older jobs by these names.
    job = site.jobs / "with_expression"
    left = sorted(p.name for p in (HELD / "with_expression").glob("explore_*"))
    assert left == ["explore_genes_v3.json", "explore_picks.json"], "the visited job has the files of its visit: " + str(left)
    (job / "explore_picks.json").write_text(json.dumps({"a": 1_000_001, "b": 1_000_002}))
    assert site.c.get("/api/jobs/with_expression/picks").json() == {"a": 1_000_001, "b": 1_000_002}, "the picks are read from the file of the visit"
