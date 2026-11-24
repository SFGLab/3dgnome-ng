"""The JSON endpoints the frontend reads, which are the catalog, the region, the loop preview, the
settings and the job submission.
"""

from __future__ import annotations

import json
import os
import sys
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
    os.environ.pop("HICDIFFUSION_DIR", None)
    from fastapi.testclient import TestClient

    from gnome3d_web.app import create_app

    c = TestClient(create_app())
    cat = c.get("/api/catalog").json()
    gmx = cat["cell_types"]["GMX"]
    return SimpleNamespace(gmx=gmx, cat=cat, c=c, root=root)


def test_the_catalog_lists_the_cell_type_with_its_data(site: SimpleNamespace) -> None:
    gmx, cat, c = site.gmx, site.cat, site.c
    assert gmx["data"] == ["CTCF", "RNAPII", "Hi-C", "enhancers"], "the catalog lists the cell type with its data: " + str(gmx)
    assert cat["max_structures"] == 100 and cat["max_region_mb"] == 20 and cat["factors"] == ["CTCF", "RNAPII", "CTCF+RNAPII"], "the catalog gives the limits and the factor choices: " + str(cat)[:200]
    assert cat["hicdiffusion"] is False, "the catalog says HiCDiffusion is not set up"
    assert cat["sv_sets"]["DEMO"]["sample_table"] is True and cat["sv_sets"]["BARE"]["sample_table"] is False, "the catalog says which call sets have a sample table: " + str(cat["sv_sets"])
    t = c.get("/api/sv_sets/DEMO/samples").json()
    assert t["columns"][0] == "sample" and t["rows"][0] == ["P1", "female", "PUR", "AMR", "", "P2"], "a call set's sample table lists each sample with its parents: " + str(t)
    assert c.get("/api/sv_sets/BARE/samples").status_code == 404 and c.get("/api/sv_sets/NOPE/samples").status_code == 404, "a call set without a table is a 404"


def test_a_gene_name_in_lower_case_resolves_to_its_window(site: SimpleNamespace) -> None:
    c = site.c
    r = c.get("/api/region", params={"cell": "GMX", "q": "alpha"}).json()
    assert r["gene"] == "ALPHA" and r["chrom"] == "chr1" and r["end"] - r["start"] == 2_000_000, "a gene name in lower case resolves to its window: " + str(r)
    assert r["label"].endswith("2.0 Mb"), "a region gets a label with its length: " + str(r)
    g = c.get("/api/genes", params={"cell": "GMX", "q": "al"}).json()
    assert g == [{"name": "ALPHA", "chrom": "chr1", "tss": 1310000}], "typed letters give the genes that start with them: " + str(g)
    g = c.get("/api/genes", params={"cell": "GMX", "q": "ta"}).json()
    assert [x["name"] for x in g] == ["BETA"], "a gene that holds the text comes after those that start with it: " + str(g)
    assert c.get("/api/genes", params={"cell": "GMX", "q": "a"}).json() == [] and c.get("/api/genes", params={"cell": "NOPE", "q": "al"}).status_code == 404, "one letter gives no genes and an unknown cell type is a 404"
    r = c.get("/api/region", params={"cell": "GMX", "q": "chr1:0-25000000"})
    assert r.status_code == 400 and "20,000,000" in r.json()["error"], "a region over the limit is a 400 naming it: " + r.text


def test_the_preview_has_each_gene_that_reaches_into_the_region_with_its_span(site: SimpleNamespace) -> None:
    c = site.c
    whole = c.get("/api/preview", params={"cell": "GMX", "region": "chr1:0-3000000"}).json()["genes"]
    assert {"name": "ALPHA", "tss": 1310000, "start": 1310000, "end": 1352000, "strand": "+"} in whole, "a gene has its TSS, its span and its strand: " + str(whole)
    # BETA spans 2,571,000 to 2,600,001 and its TSS is outside this region.
    part = c.get("/api/preview", params={"cell": "GMX", "region": "chr1:2000000-2590000"}).json()["genes"]
    assert [g["name"] for g in part] == ["BETA"], "a gene whose body reaches into the region is listed: " + str(part)


def test_the_preview_lists_the_ctcf_and_rnapii_loops(site: SimpleNamespace) -> None:
    c = site.c
    p = c.get("/api/preview", params={"cell": "GMX", "region": "chr1:0-3000000"}).json()
    assert p["counts"] == {"CTCF": 14, "RNAPII": 1} and len(p["loops"]["CTCF"]) == 14, "the preview lists the CTCF and RNAPII loops: " + str(p["counts"])
    assert p["loops"]["RNAPII"][0][:3] == [1300500, 1500500, 4] and len(p["loops"]["RNAPII"][0]) == 5 and p["loops"]["RNAPII"][0][3] < 1300500 < 1500500 < p["loops"]["RNAPII"][0][4], "a preview loop is two anchor midpoints, a count and the outer edges of the anchors: " + str(p["loops"]["RNAPII"])
    assert sorted(g["name"] for g in p["genes"]) == ["ALPHA", "BETA"], "the preview lists the genes: " + str(p["genes"])


def test_the_settings_schema_has_the_design_s_first_group(site: SimpleNamespace) -> None:
    c = site.c
    sch = c.get("/api/settings", params={"cell": "GMX"}).json()
    assert sch[0]["name"] == "Distance law", "the settings schema has the design's first group: " + str(sch[0])[:120]
    r = c.post("/api/settings/ini", files={"file": ("c.ini", b"[distance]\npolymer_exponent = 0.2\n[data]\nanchors = x\n")}).json()
    assert r["overrides"] == {"distance.polymer_exponent": "0.2"} and len(r["refused"]) == 1, "an uploaded ini is read into changes and refusals: " + str(r)


def test_a_job_post_returns_201_and_its_url(site: SimpleNamespace) -> None:
    c, root = site.c, site.root
    form = {"cell_type": "GMX", "region": "chr1:0-2000000", "factors": "CTCF", "n_models": "2", "background": "hic", "overrides": json.dumps({"distance.polymer_exponent": "0.3"})}
    r = c.post("/api/jobs", data=form)
    assert r.status_code == 201 and r.json()["url"].startswith("/jobs/"), "a job post returns 201 and its url: " + r.text

    again = c.post("/api/jobs", data=form)
    assert again.json().get("url") == r.json().get("url"), "the same post finds the same job"

    fresh = c.post("/api/jobs", data=form | {"fresh": "1"})
    repeat = fresh.json().get("url", "")
    assert fresh.status_code == 201 and repeat.startswith("/jobs/") and repeat != r.json()["url"], "the same post marked fresh starts a job of its own: " + fresh.text
    assert json.loads((root / "results" / "jobs" / repeat.rsplit("/", 1)[1] / "request.json").read_text()) == json.loads((root / "results" / "jobs" / r.json()["url"].rsplit("/", 1)[1] / "request.json").read_text()), "the repeat has the request of the first job"
    assert c.get("/api" + repeat).json().get("state") == "done", "the repeat finishes: " + c.get("/api" + repeat).text[:200]
    assert c.post("/api/jobs", data=form).json().get("url") == r.json()["url"], "the first job stays the one a plain post finds"

    other = c.post("/api/jobs", data=form | {"fresh": "1", "n_models": "3"}).json()["url"]
    assert c.post("/api/jobs", data=form | {"n_models": "3"}).json()["url"] == other, "a fresh post of a request the site does not hold takes the key of the request"

    st = json.loads((root / "results" / "jobs" / r.json()["url"].rsplit("/", 1)[1] / "request.json").read_text())
    assert st["overrides"] == {"distance.polymer_exponent": "0.3"}, "the job keeps its setting changes: " + str(st)

    bad = c.post("/api/jobs", data=form | {"overrides": json.dumps({"distance.polymer_exponent": "9"})})
    assert bad.status_code == 400 and bad.json()["field"] == "overrides", "a refused setting is a 400 on the overrides field: " + bad.text

    bad = c.post("/api/jobs", data=form | {"background": "hicdiffusion"})
    assert bad.status_code == 400 and bad.json()["field"] == "background", "HiCDiffusion without its setup is a 400 on the background field: " + bad.text

    bad = c.post("/api/jobs", data=form | {"region": "nowhere"})
    assert bad.status_code == 400 and bad.json()["field"] == "region", "an unknown region is a 400 on the region field: " + bad.text
