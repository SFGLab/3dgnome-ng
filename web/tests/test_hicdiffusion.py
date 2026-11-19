"""The HiCDiffusion step: what the server lacks and one job through it with a stand in model.

    python web/tests/test_hicdiffusion.py
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

PY = str(Path(__file__).resolve().parents[2] / ".venv" / "bin" / "python")
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


def main() -> int:
    from gnome3d_web import hicdiffusion
    from gnome3d_web.catalog import Catalog
    from gnome3d_web.jobs import JobRequest, run_job

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        parts = make_parts(root)
        check("nothing is missing with every part in place", hicdiffusion.missing(parts) == [], str(hicdiffusion.missing(parts)))
        check("with no variable set, each is named", hicdiffusion.missing({}) == [f"{k} is not set" for k in hicdiffusion.VARIABLES])
        check("one unset variable is named alone", hicdiffusion.missing(parts | {"REFERENCE_FASTA": ""}) == ["REFERENCE_FASTA is not set"])
        (root / "models" / "hicdiffusion.ckpt").rename(root / "models" / "aside.ckpt")
        check("a checkpoint that is not there is named", hicdiffusion.missing(parts) == ["HICDIFFUSION_MODELS has no hicdiffusion.ckpt"], str(hicdiffusion.missing(parts)))
        (root / "models" / "aside.ckpt").rename(root / "models" / "hicdiffusion.ckpt")
        check("a FASTA without its index is named", hicdiffusion.missing(parts | {"REFERENCE_FASTA": str(root / "models" / "hicdiffusion.ckpt")}) == ["REFERENCE_FASTA has no .fai index"])
        check("a FASTA that is not there is named", hicdiffusion.missing(parts | {"REFERENCE_FASTA": str(root / "none.fa")}) == ["REFERENCE_FASTA is not a file"])
        check("an interpreter that does not start is named", hicdiffusion.missing(parts | {"HICDIFFUSION_PYTHON": str(root / "no-python")}) == ["HICDIFFUSION_PYTHON does not start"])
        check("a directory that is not the checkout is named", hicdiffusion.missing(parts | {"HICDIFFUSION_DIR": str(root / "models")}) == ["HICDIFFUSION_DIR has no hicdiffusion_model.py"])
        check("the messages hold no path of the server", all(str(root) not in m for m in hicdiffusion.missing(parts | {"REFERENCE_FASTA": str(root / "none.fa"), "HICDIFFUSION_DIR": str(root / "models")})))

        os.environ.update(parts)
        cat = Catalog.load(make_data(root / "data"))
        req = JobRequest(cell_type="GMX", region="chr1:0-2000000", factors="CTCF", n_models=2, background="hicdiffusion")
        res = run_job(req, cat, root / "results" / "a", engine=FAKE_ENGINE, env_extra={"CUDA_VISIBLE_DEVICES": "1"})
        status = json.loads((res / "status.json").read_text())
        check("a HiCDiffusion job finishes", status["state"] == "done", str(status)[:300])
        check("its stages are inputs, hicdiffusion, modelling, marks, done", [s["stage"] for s in status["stages"]] == ["inputs", "hicdiffusion", "modelling", "marks", "done"], str(status["stages"]))
        log = (res / "engine.log").read_text()
        check("the model read the whole window", "2097152 bases" in log, log[:200])
        check("the model ran the site's runner", "runner runner.py" in log)
        beside = sorted(p.name for p in hicdiffusion.RUNNER.parent.iterdir() if p.name != "__pycache__")
        check("the runner is alone in its directory, so no module of the site shadows one the model imports", beside == ["runner.py"], str(beside))
        check("the model ran on the GPU of the job's slot", "GPU 1" in log, log[:200])
        check("the model did not get the site's PYTHONPATH", "PYTHONPATH unset" in log)
        check("the job's log holds the model's lines and then the engine's", log.index("fake HiCDiffusion") < log.index("wrote 2 structures"), log)
        sing = (res / "inputs").glob("*")
        check("the job has inputs", any(sing))
        kept = res / hicdiffusion.MAP_FILE
        rows = kept.read_text().splitlines() if kept.is_file() else []
        check("the job keeps the predicted map, one row per pair of bins", len(rows) == 256 * 257 // 2, str(len(rows)))
        check("a row is a pair of 8,192 bp bins with the predicted count", rows[:1] == ["chr1\t0\t8192\tchr1\t0\t8192\t3.482"], str(rows[:1]))
        on, first, step, counts = hicdiffusion.read_map(kept)
        check("the map reads back as the square it was", (on, first, step, counts.shape) == ("chr1", 0, 8192, (256, 256)) and bool((counts == 3.482).all()), str((on, first, step, counts.shape)))

        os.environ["FAKE_HICDIFFUSION_FAIL"] = "1"
        res = run_job(req, cat, root / "results" / "b", engine=FAKE_ENGINE)
        status = json.loads((res / "status.json").read_text())
        check("a model that fails fails the job at its stage", status["state"] == "failed" and status["stage"] == "hicdiffusion" and "HiCDiffusion exited with 4" in status["error"], str(status)[:300])
        check("the failure shows the model's own words", "the model exploded" in status["log_tail"], status.get("log_tail", "")[:200])
        os.environ.pop("FAKE_HICDIFFUSION_FAIL")

        os.environ.update(DATA_DIR=str(root / "data"), RESULTS_DIR=str(root / "site"), QUEUE="inline", FRONTEND_DIST=str(root / "missing"), ENGINE_CMD=" ".join(FAKE_ENGINE))
        from fastapi.testclient import TestClient
        from gnome3d_web.app import create_app

        c = TestClient(create_app())
        check("the catalog offers HiCDiffusion", c.get("/api/catalog").json()["hicdiffusion"] is True)
        form = {"cell_type": "GMX", "region": "chr1:0-2000000", "factors": "CTCF", "n_models": "2", "background": "hicdiffusion"}
        r = c.post("/api/jobs", data=form)
        check("a HiCDiffusion job is accepted", r.status_code == 201, r.text[:200])
        s = c.get("/api" + r.json()["url"]).json()
        check("and it finishes through the site", s["state"] == "done", str(s)[:300])
        url = r.json()["url"]
        listed = [f["name"] for f in c.get(f"/api{url}/summary").json()["files"]]
        check("the job lists the predicted map among its files", hicdiffusion.MAP_FILE in listed, str(listed))
        got = c.get(f"{url}/files/{hicdiffusion.MAP_FILE}")
        check("the predicted map downloads", got.status_code == 200 and got.text.startswith("chr1\t0\t8192\t"), str(got.status_code))
        with zipfile.ZipFile(io.BytesIO(c.get(f"/api{url}/files.zip").content)) as z:
            check("the zip of the job holds the predicted map", hicdiffusion.MAP_FILE in z.namelist(), str(z.namelist()))
        m = c.get(f"/api{url}/contacts", params={"region": "chr1:0-2000000", "res": 25000, "upper": "hicdiffusion", "lower": "model", "norm": "raw"})
        j = m.json()
        check("the contact map takes the prediction as a half", m.status_code == 200 and j["upper_label"].startswith("HiCDiffusion"), m.text[:200])
        # A flat prediction of 3.482 a pixel puts (25000 / 8192) squared pixels into a 25 kb bin.
        check("a bin of the half holds the predicted counts of its pixels", abs(j["upper"][3][10] - 3.482 * (25000 / 8192) ** 2) < 1e-3, str(j["upper"][3][10]))
        plain = c.post("/api/jobs", data=form | {"background": "hic", "region": "chr1:0-1900000"})
        bad = c.get(f"/api{plain.json()['url']}/contacts", params={"region": "chr1:0-1900000", "upper": "hicdiffusion"})
        check("a job without a prediction refuses the half", bad.status_code == 400 and "hicdiffusion" not in bad.json()["error"].split("is not one of")[1], bad.text[:200])
        os.environ["REFERENCE_FASTA"] = str(root / "none.fa")
        check("the catalog withdraws it once a part is gone", c.get("/api/catalog").json()["hicdiffusion"] is False)
        r = c.post("/api/jobs", data=form | {"region": "chr1:100000-2000000"})
        check("a job is then refused with the part named", r.status_code == 400 and r.json()["field"] == "background" and "REFERENCE_FASTA is not a file" in r.json()["error"], r.text[:200])
    return done()


if __name__ == "__main__":
    sys.exit(main())
