"""The HiCDiffusion step: what the server lacks and one job through it with a stand in model.

    pytest web/tests/test_hicdiffusion.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fixture import make_data

PY = sys.executable
FAKE_ENGINE = [PY, str(Path(__file__).resolve().parent / "fake_engine.py")]
FAKE_MODEL = f"{PY} {Path(__file__).resolve().parent / 'fake_hicdiffusion.py'}"


def make_parts(root: Path) -> dict[str, str]:
    """A checkout, two checkpoints and a 3 Mb chr1 with its index, as the four variables."""
    repo, models, ref = root / "hicdiffusion", root / "models", root / "ref"
    for d in (repo, models, ref):
        d.mkdir()
    (repo / "hicdiffusion_model.py").write_text("")
    for c in ("hicdiffusion.ckpt", "hicdiffusion_encoder_decoder.ckpt"):
        (models / c).write_text("")
    n = 3_000_000
    (ref / "hg38.fa").write_text(">chr1\n" + ("ACGTACGTAC" * 5 + "\n") * (n // 50))
    (ref / "hg38.fa.fai").write_text(f"chr1\t{n}\t6\t50\t51\n")
    return {"HICDIFFUSION_DIR": str(repo), "HICDIFFUSION_PYTHON": FAKE_MODEL, "HICDIFFUSION_MODELS": str(models), "REFERENCE_FASTA": str(ref / "hg38.fa")}


@pytest.fixture(scope="module", autouse=True)
def shared(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    from gnome3d_web.engine import hicdiffusion
    from gnome3d_web.catalog.model import Catalog
    from gnome3d_web.jobs.request import JobRequest
    from gnome3d_web.jobs.run import run_job

    d = str(tmp_path_factory.mktemp("work"))
    root = Path(d)
    parts = make_parts(root)
    return SimpleNamespace(hicdiffusion=hicdiffusion, parts=parts, root=root, Catalog=Catalog, JobRequest=JobRequest, run_job=run_job)


def test_nothing_is_missing_with_every_part_in_place(shared: SimpleNamespace) -> None:
    hicdiffusion, parts = shared.hicdiffusion, shared.parts
    assert hicdiffusion.missing(parts) == [], "nothing is missing with every part in place: " + str(hicdiffusion.missing(parts))
    assert hicdiffusion.missing({}) == [f"{k} is not set" for k in hicdiffusion.VARIABLES], "with no variable set, each is named"
    assert hicdiffusion.missing(parts | {"REFERENCE_FASTA": ""}) == ["REFERENCE_FASTA is not set"], "one unset variable is named alone"


def test_a_checkpoint_that_is_not_there_is_named(shared: SimpleNamespace) -> None:
    root, hicdiffusion, parts = shared.root, shared.hicdiffusion, shared.parts
    (root / "models" / "hicdiffusion.ckpt").rename(root / "models" / "aside.ckpt")
    assert hicdiffusion.missing(parts) == ["HICDIFFUSION_MODELS has no hicdiffusion.ckpt"], "a checkpoint that is not there is named: " + str(hicdiffusion.missing(parts))


def test_a_fasta_without_its_index_is_named(shared: SimpleNamespace) -> None:
    root, hicdiffusion, parts = shared.root, shared.hicdiffusion, shared.parts
    (root / "models" / "aside.ckpt").rename(root / "models" / "hicdiffusion.ckpt")
    assert hicdiffusion.missing(parts | {"REFERENCE_FASTA": str(root / "models" / "hicdiffusion.ckpt")}) == ["REFERENCE_FASTA has no .fai index"], "a FASTA without its index is named"
    assert hicdiffusion.missing(parts | {"REFERENCE_FASTA": str(root / "none.fa")}) == ["REFERENCE_FASTA is not a file"], "a FASTA that is not there is named"
    assert hicdiffusion.missing(parts | {"HICDIFFUSION_PYTHON": str(root / "no-python")}) == ["HICDIFFUSION_PYTHON does not start"], "an interpreter that does not start is named"
    assert hicdiffusion.missing(parts | {"HICDIFFUSION_DIR": str(root / "models")}) == ["HICDIFFUSION_DIR has no hicdiffusion_model.py"], "a directory that is not the checkout is named"
    assert all(str(root) not in m for m in hicdiffusion.missing(parts | {"REFERENCE_FASTA": str(root / "none.fa"), "HICDIFFUSION_DIR": str(root / "models")})), "the messages hold no path of the server"


@pytest.fixture(scope="module")
def status_and_res_and_req(shared: SimpleNamespace) -> SimpleNamespace:
    parts, Catalog, root, JobRequest, run_job = shared.parts, shared.Catalog, shared.root, shared.JobRequest, shared.run_job
    os.environ.update(parts)
    cat = Catalog.load(make_data(root / "data"))
    req = JobRequest(cell_type="GMX", region="chr1:0-2000000", factors="CTCF", n_models=2, background="hicdiffusion")
    res = run_job(req, cat, root / "results" / "a", engine=FAKE_ENGINE, env_extra={"CUDA_VISIBLE_DEVICES": "1"})
    status = json.loads((res / "status.json").read_text())
    return SimpleNamespace(status=status, res=res, req=req, cat=cat)


def test_a_hicdiffusion_job_finishes(status_and_res_and_req: SimpleNamespace) -> None:
    status = status_and_res_and_req.status
    assert status["state"] == "done", "a HiCDiffusion job finishes: " + str(status)[:300]
    assert [s["stage"] for s in status["stages"]] == ["inputs", "hicdiffusion", "modelling", "marks", "done"], "its stages are inputs, hicdiffusion, modelling, marks, done: " + str(status["stages"])


@pytest.fixture(scope="module")
def log(status_and_res_and_req: SimpleNamespace) -> Any:
    res = status_and_res_and_req.res
    log = (res / "engine.log").read_text()
    return log


def test_the_model_read_the_whole_window(log: Any) -> None:
    assert "2097152 bases" in log, "the model read the whole window: " + str(log[:200])
    assert "runner runner.py" in log, "the model ran the site's runner"


def test_the_runner_is_alone_in_its_directory_so_no_module_of_the_site_shadows(log: Any, shared: SimpleNamespace) -> None:
    hicdiffusion = shared.hicdiffusion
    beside = sorted(p.name for p in hicdiffusion.RUNNER.parent.iterdir() if p.name != "__pycache__")
    assert beside == ["runner.py"], "the runner is alone in its directory, so no module of the site shadows one the model imports: " + str(beside)
    assert "GPU 1" in log, "the model ran on the GPU of the job's slot: " + str(log[:200])
    assert "PYTHONPATH unset" in log, "the model did not get the site's PYTHONPATH"
    assert log.index("fake HiCDiffusion") < log.index("wrote 2 structures"), "the job's log holds the model's lines and then the engine's: " + str(log)


def test_the_job_has_inputs(status_and_res_and_req: SimpleNamespace) -> None:
    res = status_and_res_and_req.res
    sing = (res / "inputs").glob("*")
    assert any(sing), "the job has inputs"


@pytest.fixture(scope="module")
def rows_and_kept(status_and_res_and_req: SimpleNamespace, shared: SimpleNamespace) -> SimpleNamespace:
    res = status_and_res_and_req.res
    hicdiffusion = shared.hicdiffusion
    kept = res / hicdiffusion.MAP_FILE
    rows = kept.read_text().splitlines() if kept.is_file() else []
    return SimpleNamespace(rows=rows, kept=kept)


def test_the_job_keeps_the_predicted_map_one_row_per_pair_of_bins(rows_and_kept: SimpleNamespace) -> None:
    rows = rows_and_kept.rows
    assert len(rows) == 256 * 257 // 2, "the job keeps the predicted map, one row per pair of bins: " + str(len(rows))
    assert rows[:1] == ["chr1\t0\t8192\tchr1\t0\t8192\t3.482"], "a row is a pair of 8,192 bp bins with the predicted count: " + str(rows[:1])


def test_the_map_reads_back_as_the_square_it_was(shared: SimpleNamespace, rows_and_kept: SimpleNamespace) -> None:
    hicdiffusion = shared.hicdiffusion
    kept = rows_and_kept.kept
    on, first, step, counts = hicdiffusion.read_map(kept)
    assert (on, first, step, counts.shape) == ("chr1", 0, 8192, (256, 256)) and bool((counts == 3.482).all()), "the map reads back as the square it was: " + str((on, first, step, counts.shape))


def test_a_model_that_fails_fails_the_job_at_its_stage(shared: SimpleNamespace, status_and_res_and_req: SimpleNamespace) -> None:
    run_job, root = shared.run_job, shared.root
    req, cat = status_and_res_and_req.req, status_and_res_and_req.cat
    os.environ["FAKE_HICDIFFUSION_FAIL"] = "1"
    res = run_job(req, cat, root / "results" / "b", engine=FAKE_ENGINE)
    status = json.loads((res / "status.json").read_text())
    assert status["state"] == "failed" and status["stage"] == "hicdiffusion" and "HiCDiffusion exited with 4" in status["error"], "a model that fails fails the job at its stage: " + str(status)[:300]
    assert "the model exploded" in status["log_tail"], "the failure shows the model's own words: " + str(status.get("log_tail", "")[:200])
    os.environ.pop("FAKE_HICDIFFUSION_FAIL")


def test_the_catalog_offers_hicdiffusion(shared: SimpleNamespace) -> None:
    root, hicdiffusion = shared.root, shared.hicdiffusion
    os.environ.update(DATA_DIR=str(root / "data"), RESULTS_DIR=str(root / "site"), QUEUE="inline", FRONTEND_DIST=str(root / "missing"), ENGINE_CMD=" ".join(FAKE_ENGINE))
    from fastapi.testclient import TestClient

    from gnome3d_web.app import create_app

    c = TestClient(create_app())
    assert c.get("/api/catalog").json()["hicdiffusion"] is True, "the catalog offers HiCDiffusion"

    form = {"cell_type": "GMX", "region": "chr1:0-2000000", "factors": "CTCF", "n_models": "2", "background": "hicdiffusion"}
    r = c.post("/api/jobs", data=form)
    assert r.status_code == 201, "a HiCDiffusion job is accepted: " + r.text[:200]

    s = c.get("/api" + r.json()["url"]).json()
    assert s["state"] == "done", "and it finishes through the site: " + str(s)[:300]

    url = r.json()["url"]
    listed = [f["name"] for f in c.get(f"/api{url}/summary").json()["files"]]
    assert hicdiffusion.MAP_FILE in listed, "the job lists the predicted map among its files: " + str(listed)

    got = c.get(f"{url}/files/{hicdiffusion.MAP_FILE}")
    assert got.status_code == 200 and got.text.startswith("chr1\t0\t8192\t"), "the predicted map downloads: " + str(got.status_code)
    with zipfile.ZipFile(io.BytesIO(c.get(f"/api{url}/files.zip").content)) as z:
        assert hicdiffusion.MAP_FILE in z.namelist(), "the zip of the job holds the predicted map: " + str(z.namelist())

    m = c.get(f"/api{url}/contacts", params={"region": "chr1:0-2000000", "res": 25000, "upper": "hicdiffusion", "lower": "model", "norm": "raw"})
    j = m.json()
    assert m.status_code == 200 and j["upper_label"].startswith("HiCDiffusion"), "the contact map takes the prediction as a half: " + m.text[:200]
    # A flat prediction of 3.482 a pixel puts (25000 / 8192) squared pixels into a 25 kb bin.
    # Row 3 of the half starts at column 3: the pair of bins 3 and 10 is its eighth cell.
    assert abs(j["upper"][3][7] - 3.482 * (25000 / 8192) ** 2) < 1e-3, "a bin of the half holds the predicted counts of its pixels: " + str(j["upper"][3][7])

    plain = c.post("/api/jobs", data=form | {"background": "hic", "region": "chr1:0-1900000"})
    bad = c.get(f"/api{plain.json()['url']}/contacts", params={"region": "chr1:0-1900000", "upper": "hicdiffusion"})
    assert bad.status_code == 400 and "hicdiffusion" not in bad.json()["error"].split("is not one of")[1], "a job without a prediction refuses the half: " + bad.text[:200]

    os.environ["REFERENCE_FASTA"] = str(root / "none.fa")
    assert c.get("/api/catalog").json()["hicdiffusion"] is False, "the catalog withdraws it once a part is gone"

    r = c.post("/api/jobs", data=form | {"region": "chr1:100000-2000000"})
    assert r.status_code == 400 and r.json()["field"] == "background" and "REFERENCE_FASTA is not a file" in r.json()["error"], "a job is then refused with the part named: " + r.text[:200]
