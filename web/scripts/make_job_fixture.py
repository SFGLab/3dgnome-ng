"""Writes the job directories of web/tests/job_on_disk and the answers the site gives on them.

    python web/scripts/make_job_fixture.py OUT WORK

The directories under web/tests/job_on_disk were written with this script at commit 7b8affb7,
before the refactor of the site. They stand for the jobs a running site holds on its disk.
web/tests/test_job_on_disk.py reads them with the code of the day. A run of this script on later
code makes directories of that later code and proves nothing about the old ones. OUT must not
exist. WORK is an empty scratch directory.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

WEB = Path(__file__).resolve().parents[1]
for extra in (WEB.parent, WEB / "vendor", WEB, WEB / "tests"):
    sys.path.insert(0, str(extra))

from fixture import make_data  # noqa: E402
from test_job_on_disk import REGION, ROUTES, thin  # noqa: E402

# A server keeps its data and its results under paths of its own. The fixture states one.
SERVER = "/srv/site"


def expression_table(path: Path) -> None:
    names = ["ALPHA", "BETA"] + [f"GENE{i}" for i in range(48)]
    rows = ["gene\tlength\ta1\ta2"] + [f"{n}\t{1000 + 10 * i}\t{40 + 37 * i}\t{44 + 35 * i}" for i, n in enumerate(names)]
    path.write_text("\n".join(rows) + "\n")


def site(root: Path):  # noqa: ANN201
    os.environ.update(
        DATA_DIR=str(root / "data"), RESULTS_DIR=str(root / "results"), QUEUE="inline", FRONTEND_DIST=str(root / "missing"),
        ENGINE_CMD=f"{sys.executable} {WEB / 'tests' / 'fake_engine.py'}",
    )  # fmt: skip
    from fastapi.testclient import TestClient

    from gnome3d_web.app import create_app

    return TestClient(create_app())


def answers(c, name: str) -> dict[str, object]:  # noqa: ANN001
    api = f"/api/jobs/{name}"
    out: dict[str, object] = {}
    for route in ROUTES:
        r = c.get(f"{api}/{route}")
        assert r.status_code == 200, (name, route, r.text[:300])
        out[route] = thin(r.json())
    out["summary"].pop("now")  # type: ignore[union-attr]
    out["status"] = c.get(api).json()
    out["log"] = c.get(f"{api}/log").text
    out["zip"] = sorted(zipfile.ZipFile(io.BytesIO(c.get(f"{api}/files.zip").content)).namelist())
    return out


def main() -> None:
    out, work = Path(sys.argv[1]), Path(sys.argv[2])
    assert not out.exists(), f"{out} exists"
    data = make_data(work / "first" / "data")
    (data / "expression").mkdir()
    expression_table(data / "expression" / "lines.tsv")
    catalog = json.loads((data / "catalog.json").read_text())
    catalog["expression"] = {"LINES": {"label": "Lines RNA-seq", "file": "expression/lines.tsv", "kind": "counts", "cell_types": {"GMX": ["a1", "a2"]}}}
    (data / "catalog.json").write_text(json.dumps(catalog))
    c = site(work / "first")
    form = {"cell_type": "GMX", "region": REGION, "n_models": "2"}
    jobs = {
        "with_variants": {**form, "factors": "CTCF+RNAPII", "sv_set": "DEMO", "sample": "P1", "haplotype": "0"},
        "with_expression": {**form, "factors": "CTCF", "expression": "LINES"},
    }
    out.mkdir(parents=True)
    keys = {}
    for name, fields in jobs.items():
        r = c.post("/api/jobs", data=fields)
        assert r.status_code == 201 and r.json()["url"].startswith("/jobs/"), r.text
        keys[name] = r.json()["url"].split("/")[-1]
        made = work / "first" / "results" / "jobs" / keys[name]
        if name == "with_expression":
            # This one is kept as a visited job, with the files the job page leaves in it.
            for route in ("summary", "picks", "genes", "loops"):
                assert c.get(f"/api/jobs/{keys[name]}/{route}").status_code == 200, route
        shutil.copytree(made, out / name)
    first = str((work / "first").resolve())
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.suffix in (".json", ".log", ".ini", ".tsv", ".bed", ".bedpe", ".txt"):
            text = p.read_text()
            if first in text or str(work / "first") in text:
                p.write_text(text.replace(first, SERVER).replace(str(work / "first"), SERVER))
    # The answers come from the copies, on a second site with another place on disk.
    make_data(work / "second" / "data")
    for name in jobs:
        shutil.copytree(out / name, work / "second" / "results" / "jobs" / name)
    c = site(work / "second")
    said = {name: answers(c, name) for name in jobs}
    pair = "a=/jobs/with_variants&b=/jobs/with_expression"
    said["compare"] = {q: c.get(f"/api/compare?{pair}{q}").json() for q in ("", "&shared=1")}
    said["compare"]["gene"] = c.get(f"/api/compare/gene?{pair}&gene=ALPHA").json()
    (out / "answers.json").write_text(json.dumps({"keys": keys, "answers": said}, indent=1, sort_keys=True) + "\n")
    print(keys, sum(p.stat().st_size for p in out.rglob("*") if p.is_file()))


main()
