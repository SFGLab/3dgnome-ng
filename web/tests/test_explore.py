"""The job data endpoints of the explorer.

    pytest web/tests/test_explore.py
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


@pytest.fixture(scope="module", autouse=True)
def site(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    d = str(tmp_path_factory.mktemp("work"))
    root = Path(d)
    make_data(root / "data")
    os.environ.update(
        DATA_DIR=str(root / "data"),
        RESULTS_DIR=str(root / "results"),
        QUEUE="inline",
        FRONTEND_DIST=str(root / "missing"),
        ENGINE_CMD=f"{sys.executable} {Path(__file__).resolve().parent / 'fake_engine.py'}",
    )
    from fastapi.testclient import TestClient

    from gnome3d_web.app import create_app

    c = TestClient(create_app())
    form = {"cell_type": "GMX", "region": "chr1:0-3000000", "factors": "CTCF+RNAPII", "n_models": "3", "background": "hic"}
    three = "/api" + c.post("/api/jobs", data=form).json()["url"]
    one = "/api" + c.post("/api/jobs", data=form | {"n_models": "1"}).json()["url"]

    s = c.get(three + "/summary").json()
    names = {f["name"] for f in s["files"]}
    return SimpleNamespace(s=s, names=names, c=c, form=form, three=three, one=one, root=root)


def test_the_summary_has_the_state_structures_and_files(site: SimpleNamespace) -> None:
    s, names, c, form = site.s, site.names, site.c, site.form
    assert s["state"] == "done" and s["structures"] == 3 and {"ep.tsv", "marks.json", "engine.log"} <= names, "the summary has the state, structures and files: " + str(s)[:200]
    assert s["loops"] == {"CTCF": 14, "RNAPII": 1} and s["genes"] >= 1, "the summary has the loop counts and the gene count: " + str(s)[:300]
    assert s["ahead"] is None, "a done job is not in the queue"
    assert s["sample_note"] == "", "a job without a hosted sample has no sample note: " + str(s.get("sample_note"))
    hosted = "/api" + c.post("/api/jobs", data=form | {"n_models": "1", "sv_set": "DEMO", "sample": "P1"}).json()["url"]
    note = c.get(hosted + "/summary").json()["sample_note"]
    assert note == "PUR, AMR, female", "a hosted sample's job gives its population and sex: " + str(note)


def test_a_structure_has_a_chain_with_genomic_positions(site: SimpleNamespace) -> None:
    c, three = site.c, site.three
    st = c.get(three + "/structure/2").json()
    assert len(st["chain"]) > 100 and len(st["chain"][0]) == 4, "a structure has a chain with genomic positions: " + str(st["chain"][:2])
    assert len(st["beads"]) > 100 and st["beads"][0][5] in ("anchor", "subanchor"), "a structure has its beads"
    anchored = [a for a in st["anchors"] if a["loops"]]
    assert len(anchored) >= 2 and anchored[0]["loops"][0]["factor"] in ("CTCF", "RNAPII"), "anchors carry their loops: " + str(anchored[:1])
    assert c.get(three + "/structure/4").status_code == 404, "structure 4 of 3 is a 404"


def test_annotations_list_the_genes_enhancers_and_loops_of_the_region(site: SimpleNamespace) -> None:
    c, three = site.c, site.three
    ann = c.get(three + "/annotations").json()
    assert {g["name"] for g in ann["genes"]} == {"ALPHA", "BETA"} and len(ann["enhancers"]) == 2 and len(ann["loops"]) == 15, "annotations list the genes, enhancers and loops of the region: " + str(ann)[:300]


def test_the_picks_are_structure_numbers(site: SimpleNamespace) -> None:
    c, three = site.c, site.three
    p = c.get(three + "/picks").json()
    assert all(1 <= p[k] <= 3 for k in ("medoid", "compact", "random")), "the picks are structure numbers: " + str(p)


def test_a_distance_has_one_value_per_structure_a_mean_and_an_sd(site: SimpleNamespace) -> None:
    c, three, one = site.c, site.three, site.one
    r = c.get(three + "/distance", params={"a": 1310000, "b": 2600000}).json()
    assert r["n"] == 3 and len(r["values"]) == 3 and r["sd"] > 0 and abs(r["mean"] - sum(r["values"]) / 3) < 1e-9, "a distance has one value per structure, a mean and an SD: " + str(r)
    r1 = c.get(one + "/distance", params={"a": 1310000, "b": 2600000}).json()
    assert r1["n"] == 1 and r1["sd"] == 0, "one structure has an SD of 0: " + str(r1)
    far = c.get(three + "/distance", params={"a": 3_200_000, "b": 2600000})
    assert far.status_code == 400 and "kb from the nearest bead" in far.json()["error"], "a position far from every bead is a 400 with the distance: " + far.text


def test_gene_rows_carry_statistics_over_the_structures_and_the_loop_joining(site: SimpleNamespace) -> None:
    c, three = site.c, site.three
    g = c.get(three + "/genes").json()
    assert g and {"min", "q1", "median", "q3", "max", "joined_by", "n"} <= set(g[0]), "gene rows carry statistics over the structures and the loop joining them: " + str(g[:1])
    assert all(isinstance(r.get("promoter_loops"), int) and r["promoter_loops"] >= 0 for r in g), "gene rows count the loops the promoter bead anchors: " + str(g[:1])


def test_the_loops_list_factor_ends_span_and_pets(site: SimpleNamespace) -> None:
    c, three = site.c, site.three
    loops = c.get(three + "/loops").json()
    assert len(loops) == 15 and {"factor", "a", "b", "span", "pets"} <= set(loops[0]), "the loops list factor, ends, span and PETs: " + str(loops[:1])
    assert "wrote 3 structures" in c.get(three + "/log").text, "the log is text"
    z = zipfile.ZipFile(io.BytesIO(c.get(three + "/files.zip").content))
    assert "ep.tsv" in z.namelist() and any(n.endswith(".cif") for n in z.namelist()), "the zip holds the job's files: " + str(z.namelist()[:5])


def test_two_jobs_on_one_region_compare_on_the_same_region(site: SimpleNamespace) -> None:
    c, form, three = site.c, site.form, site.three
    ctcf = c.post("/api/jobs", data=form | {"factors": "CTCF"}).json()["url"]
    both = three.removeprefix("/api")
    cmp = c.get("/api/compare", params={"a": both, "b": ctcf}).json()
    assert cmp["same_region"] is True, "two jobs on one region compare on the same region: " + str(cmp)[:200]
    assert cmp["rows"] and {"gene", "linear_bp", "a", "b", "ratio"} <= set(cmp["rows"][0]), "compare rows carry both means and a ratio: " + str(cmp["rows"][:1])
    fold = [max(r["ratio"], 1 / r["ratio"]) for r in cmp["rows"]]
    assert fold == sorted(fold, reverse=True), "compare rows are sorted by the size of the change: " + str(fold)
    other = c.post("/api/jobs", data=form | {"region": "chr1:500000-3000000"}).json()["url"]
    cmp2 = c.get("/api/compare", params={"a": both, "b": other}).json()
    shared = {r["gene"] for r in c.get(three + "/genes").json()} & {r["gene"] for r in c.get("/api" + other + "/genes").json()}
    assert cmp2["same_region"] is False and {r["gene"] for r in cmp2["rows"]} <= shared, "jobs on different regions say so and list the shared genes only: " + str(cmp2)[:200]
    assert c.get("/api/compare", params={"a": both, "b": "/jobs/nothing123"}).status_code == 404, "a url that is not a job is a 404"


def test_a_private_job_answers_under_api_p(site: SimpleNamespace) -> None:
    root, c, form = site.root, site.c, site.form
    vcf = (root / "data" / "svs" / "demo.vcf.gz").read_bytes()
    priv = c.post("/api/jobs", data=form | {"n_models": "1"}, files={"vcf": ("mine.vcf.gz", vcf)}).json()["url"]
    token = priv.rsplit("/", 1)[1]
    assert c.get(f"/api/p/{token}/summary").status_code == 200, "a private job answers under /api/p"
    assert c.get(f"/api/jobs/{token}/summary").status_code == 404, "a private job is not under /api/jobs"
    from gnome3d_web import explore as ex
    from gnome3d_web.jobs import _status

    run = root / "results" / "jobs" / "running0000000001"
    (run / "models").mkdir(parents=True)
    (run / "request.json").write_text(json.dumps({"cell_type": "GMX", "region": "chr1:0-3000000", "factors": "CTCF", "n_models": 5}))
    _status(run, state="queued", stage="queued")
    _status(run, state="running", stage="inputs")
    _status(run, stage="modelling")
    _status(run, stage="modelling")
    for k in (1, 2):
        (run / "models" / f"chr1_0_3000000_s{k}.cif").write_text("data_x\n")
    (run / "engine.log").write_text("".join(f"line {k}\n" for k in range(40)))
    live = ex.summary(run, None)
    assert [st["stage"] for st in live["stages"]] == ["queued", "inputs", "modelling"] and all("at" in st for st in live["stages"]), "a running job lists its stages once each, in order: " + str(live.get("stages"))
    assert live["structures_done"] == 2 and live["structures_asked"] == 5, "a running job counts the structures written: " + str(live)
    (run / "engine.log").write_text("".join(f"line {k}\n" for k in range(39)) + f"wrote {run}/models/s3.cif\n")
    tail = ex.summary(run, None)["log_recent"]
    assert str(run) not in tail and "wrote models/s3.cif" in tail and str(run) not in ex.log_text(run), "the log shows no server path: " + str(tail[-80:])
    (run / "engine.log").write_text("".join(f"line {k}\n" for k in range(450)))
    live = ex.summary(run, None)
    assert live["log_recent"].splitlines()[-1] == "line 449" and len(live["log_recent"].splitlines()) == 200, "a running job carries the end of its engine log: " + str(live.get("log_recent"))[-60:]
