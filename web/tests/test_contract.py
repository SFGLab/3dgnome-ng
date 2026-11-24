"""The answers of the site against the shapes its interface declares.

The app describes every route and the shape of its answers at /openapi.json. The types of the
frontend are generated from that description. Each test here asks a route and compares the answer
with the declared shape. Every key is declared, no key is undeclared and every value has its type.
The last test fails for a route no test of this file has asked.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fixture import make_data

HELD = Path(__file__).resolve().parent / "job_on_disk"
REGION = "chr1:800000-2800000"
FORM = {"cell_type": "GMX", "region": "chr1:0-3000000", "factors": "CTCF", "n_models": "2"}
# Answers that are files or text, not JSON of a declared shape.
NOT_JSON = {("get", "/jobs/{key}/files/{name}"), ("get", "/p/{key}/files/{name}"), ("get", "/api/jobs/{key}/log"), ("get", "/api/p/{key}/log"), ("get", "/api/jobs/{key}/files.zip"), ("get", "/api/p/{key}/files.zip")}


def mismatch(value: Any, schema: dict[str, Any], spec: dict[str, Any], at: str = "") -> str:
    """Returns the first place a value departs from a JSON schema of the description, empty when it conforms."""
    if "$ref" in schema:
        return mismatch(value, spec["components"]["schemas"][schema["$ref"].rsplit("/", 1)[-1]], spec, at)
    if "anyOf" in schema:
        found = [mismatch(value, s, spec, at) for s in schema["anyOf"]]
        return "" if "" in found else f"{at}: none of {len(found)} shapes, the first says {found[0]}"
    if "const" in schema or "enum" in schema:
        allowed = schema.get("enum", [schema.get("const")])
        return "" if value in allowed else f"{at}: {value!r} is not one of {allowed}"
    kind = schema.get("type")
    if kind == "null":
        return "" if value is None else f"{at}: {value!r} is not null"
    if kind == "boolean":
        return "" if isinstance(value, bool) else f"{at}: {value!r} is not a boolean"
    if kind in ("integer", "number"):
        whole = isinstance(value, int) and not isinstance(value, bool)
        fits = whole or (kind == "number" and isinstance(value, float))
        return "" if fits else f"{at}: {value!r} is not {kind}"
    if kind == "string":
        return "" if isinstance(value, str) else f"{at}: {value!r} is not a string"
    if kind == "array":
        if not isinstance(value, list):
            return f"{at}: {type(value).__name__} is not an array"
        if "prefixItems" in schema:
            parts = schema["prefixItems"]
            if len(value) != len(parts):
                return f"{at}: {len(value)} items, the shape has {len(parts)}"
            return next((m for m in (mismatch(v, s, spec, f"{at}[{i}]") for i, (v, s) in enumerate(zip(value, parts, strict=True))) if m), "")
        return next((m for m in (mismatch(v, schema.get("items", {}), spec, f"{at}[{i}]") for i, v in enumerate(value[:300])) if m), "")
    if kind == "object":
        if not isinstance(value, dict):
            return f"{at}: {type(value).__name__} is not an object"
        if "properties" not in schema:
            each = schema.get("additionalProperties", {})
            return next((m for m in (mismatch(v, each, spec, f"{at}/{k}") for k, v in value.items()) if m), "") if isinstance(each, dict) else ""
        missing = sorted(set(schema.get("required", [])) - set(value))
        extra = sorted(set(value) - set(schema["properties"]))
        if missing or extra:
            return f"{at}: lacks {missing} and has undeclared {extra}"
        return next((m for m in (mismatch(v, schema["properties"][k], spec, f"{at}/{k}") for k, v in value.items()) if m), "")
    return f"{at}: the shape declares nothing" if not schema else ""


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    root = Path(str(tmp_path_factory.mktemp("work")))
    data = make_data(root / "data")
    (data / "expression").mkdir()
    (data / "expression" / "lines.tsv").write_text("gene\tlength\ta1\ta2\n" + "".join(f"G{i}\t1000\t{40 + i}\t{44 + i}\n" for i in range(30)) + "ALPHA\t1000\t40\t44\nBETA\t1200\t77\t79\n")
    catalog = json.loads((data / "catalog.json").read_text())
    catalog["expression"] = {"LINES": {"label": "Lines RNA-seq", "file": "expression/lines.tsv", "kind": "counts", "cell_types": {"GMX": ["a1", "a2"]}}}
    (data / "catalog.json").write_text(json.dumps(catalog))
    for name in ("with_variants", "with_expression"):
        shutil.copytree(HELD / name, root / "results" / "jobs" / name)
    os.environ.update(
        DATA_DIR=str(data), RESULTS_DIR=str(root / "results"), QUEUE="inline", FRONTEND_DIST=str(root / "missing"),
        ENGINE_CMD=f"{sys.executable} {Path(__file__).resolve().parent / 'fake_engine.py'}",
    )  # fmt: skip
    from fastapi.testclient import TestClient

    from gnome3d_web.app import create_app

    c = TestClient(create_app())
    spec = c.get("/openapi.json").json()
    asked: set[tuple[str, str]] = set()

    def ask(method: str, route: str, url: str, expect: int, **more: Any) -> Any:
        """Asks a route and holds its answer against the shape declared for its status."""
        r = c.request(method, url, **more)
        assert r.status_code == expect, f"{method} {url} answers {expect}: {r.status_code} {r.text[:200]}"
        declared = spec["paths"][route][method]["responses"].get(str(expect), {}).get("content", {}).get("application/json", {}).get("schema")
        assert declared, f"{method} {route} declares the shape of its answer {expect}"
        found = mismatch(r.json(), declared, spec)
        assert found == "", f"the answer {expect} of {method} {route} has its declared shape, at {found}"
        if expect < 300:
            asked.add((method, route))
        return r.json()

    return SimpleNamespace(c=c, spec=spec, ask=ask, asked=asked, data=data, ct=data / "cell_types" / "GMX")


def test_the_form_routes_answer_in_their_shapes(site: SimpleNamespace) -> None:
    ask = site.ask
    ask("get", "/api/catalog", "/api/catalog", 200)
    assert len(ask("get", "/api/genes", "/api/genes?cell=GMX&q=AL", 200)) == 1, "the search finds ALPHA"
    ask("get", "/api/genes", "/api/genes?cell=NOSUCH&q=AL", 404)
    assert "gene" in ask("get", "/api/region", "/api/region?cell=GMX&q=ALPHA", 200), "a gene name gives its region with the gene"
    assert "gene" not in ask("get", "/api/region", f"/api/region?cell=GMX&q={REGION}", 200), "a region has no gene"
    ask("get", "/api/region", "/api/region?cell=GMX&q=nothing", 400)
    assert ask("get", "/api/preview", f"/api/preview?cell=GMX&region={REGION}", 200)["counts"]["CTCF"] > 0, "the preview has loops"
    assert ask("get", "/api/expression", "/api/expression?cell=GMX", 200)["options"], "the cell type has an expression option"
    assert ask("get", "/api/settings", "/api/settings?cell=GMX", 200), "the settings have groups"
    ini = ask("post", "/api/settings/ini", "/api/settings/ini", 200, files={"file": ("c.ini", b"[main]\nmax_pet_length = 500000\nnot_a_key = 1\n")})
    assert ini["refused"], "a key users may not change is refused: " + str(ini)
    assert ask("get", "/api/sv_sets/{name}/samples", "/api/sv_sets/DEMO/samples", 200)["rows"], "the call set has its samples"
    ask("get", "/api/sv_sets/{name}/samples", "/api/sv_sets/BARE/samples", 404)


@pytest.fixture(scope="module")
def uploads(site: SimpleNamespace) -> dict[str, str]:
    ct, data = site.ct, site.data
    files = {
        "loops": ("ctcf.bedpe", (ct / "ctcf.bedpe").read_bytes()),
        "anchors": ("a.bed", (ct / "anchors_ctcf.bed").read_bytes()),
        "contacts": ("c.bedpe", (ct / "singletons.bedpe").read_bytes()),
        "enhancers": ("e.bed", (ct / "enhancers.bed").read_bytes()),
        "vcf": ("demo.vcf.gz", (data / "svs" / "demo.vcf.gz").read_bytes()),
        "expression": ("mine.tsv", b"gene\tmine\tother\nALPHA\t12\t3\nBETA\t3\t9\n"),
    }
    return {kind: site.ask("post", "/api/uploads", "/api/uploads", 201, data={"kind": kind, "cell": "GMX", "region": REGION}, files={"file": file})["id"] for kind, file in files.items()}


def test_the_upload_and_variant_routes_answer_in_their_shapes(site: SimpleNamespace, uploads: dict[str, str]) -> None:
    ask = site.ask
    old = b"chr1\t100000\t101000\tchr1\t300000\t301000\tR\tL\tCTCF\t5\n"
    refused = ask("post", "/api/uploads", "/api/uploads", 201, data={"kind": "loops"}, files={"file": ("old.txt", old)})
    assert not refused["ok"] and refused["error"]["line"] == 1, "a file that fails its check has the line, what was found and the fix: " + str(refused)
    ask("post", "/api/uploads", "/api/uploads", 400, data={"kind": "nothing"}, files={"file": ("x", b"x")})
    ask("post", "/api/uploads/link", "/api/uploads/link", 400, data={"url": "http://127.0.0.1:9/x.hic", "cell": "GMX"})
    assert ask("get", "/api/uploads/{uid}/preview", f"/api/uploads/{uploads['loops']}/preview?region={REGION}&cell=GMX", 200), "the loops of the upload are drawn"
    check = {"cell": "GMX", "region": "chr1:0-3000000", "sample": "P1", "haplotype": "0", "sv_set": "DEMO"}
    assert ask("post", "/api/variants/check", "/api/variants/check", 200, data=check)["applied"] == {"DEL": 1}, "the first haplotype has the deletion"
    ask("post", "/api/variants/check", "/api/variants/check", 200, data={**check, "haplotype": "1", "upload": uploads["vcf"], "sv_set": ""})
    ask("post", "/api/variants/check", "/api/variants/check", 400, data={**check, "sample": "NOBODY"})


@pytest.fixture(scope="module")
def jobs(site: SimpleNamespace, uploads: dict[str, str]) -> SimpleNamespace:
    ask = site.ask
    post = lambda **more: ask("post", "/api/jobs", "/api/jobs", 201, data={**FORM, **more})["url"]  # noqa: E731
    ask("post", "/api/jobs", "/api/jobs", 400, data={**FORM, "region": "chr1:0-99000000"})
    states = ask("post", "/api/jobs/states", "/api/jobs/states", 200, json={"jobs": ["/jobs/with_variants", "/jobs/absent"]})["states"]
    assert states["/jobs/with_variants"]["state"] == "done" and states["/jobs/absent"] is None, "the states of a list name each job asked: " + str(states)
    own = {"loops": [{"factor": "CTCF", "upload": uploads["loops"]}], "contacts": uploads["contacts"], "enhancers": uploads["enhancers"], "anchors": uploads["anchors"]}
    return SimpleNamespace(
        plain=post(),
        failed=post(region="chr9:0-1000000"),
        own=post(cell_type="own", own=json.dumps(own), expression="own", expression_upload=uploads["expression"], expression_columns="[]", expression_kind="values"),
        hosted=post(expression="LINES", region="chr1:0-2900000"),
        variants=post(sv_set="DEMO", sample="P1", haplotype="1"),
        held=["/jobs/with_variants", "/jobs/with_expression"],
    )


def test_the_routes_of_a_job_page_answer_in_their_shapes(site: SimpleNamespace, jobs: SimpleNamespace) -> None:
    ask = site.ask
    for url in (jobs.plain, jobs.failed, jobs.own, jobs.hosted, jobs.variants, *jobs.held):
        kind = url.split("/")[1]
        route, api = f"/api/{kind}/{{key}}", f"/api{url}"
        status = ask("get", route, api, 200)
        summary = ask("get", f"{route}/summary", f"{api}/summary", 200)
        assert summary["state"] == status["state"] == ("failed" if url == jobs.failed else "done"), f"{url} is {summary['state']}"
        ask("get", f"{route}/structure/{{i}}", f"{api}/structure/9", 404)
        ask("get", f"{route}/distance", f"{api}/distance?a=900000000&b=900000001", 400)
        ask("get", f"{route}/annotations", f"{api}/annotations", 200)
        ask("get", f"{route}/loops", f"{api}/loops", 200)
        ask("get", f"{route}/contacts", f"{api}/contacts?region={REGION}&res=7", 400)
        if url == jobs.failed:
            continue
        ask("get", f"{route}/structure/{{i}}", f"{api}/structure/1", 200)
        ask("get", f"{route}/picks", f"{api}/picks", 200)
        ask("get", f"{route}/distance", f"{api}/distance?a=1310000&b=1490000", 200)
        assert ask("get", f"{route}/genes", f"{api}/genes", 200), f"{url} has genes"
        ask("get", f"{route}/contacts", f"{api}/contacts?region={REGION}&res=25000", 200)
        ask("get", f"{route}/contacts", f"{api}/contacts?region={REGION}&res=25000&upper=model_count&lower=loops_ctcf&norm=raw", 200)
    assert site.c.get(f"/api{jobs.plain}/log").status_code == 200 and site.c.get(f"/api{jobs.own}/files.zip").status_code == 200, "the log and the zip answer"


def test_a_job_without_structures_is_refused_where_they_are_needed(site: SimpleNamespace, jobs: SimpleNamespace) -> None:
    ask = site.ask
    api = f"/api{jobs.failed}"
    picks = ask("get", "/api/jobs/{key}/picks", f"{api}/picks", 404)
    distance = ask("get", "/api/jobs/{key}/distance", f"{api}/distance?a=1310000&b=1490000", 400)
    assert "no structure" in picks["error"] and "no structure" in distance["error"], "a failed job says it has no structures: " + str((picks, distance))
    assert ask("get", "/api/jobs/{key}/genes", f"{api}/genes", 200) == [], "its gene table is empty"


def test_the_comparison_of_two_jobs_answers_in_its_shapes(site: SimpleNamespace, jobs: SimpleNamespace) -> None:
    ask = site.ask
    modes = set()
    for a, b in ((jobs.held[0], jobs.held[1]), (jobs.own, jobs.hosted), (jobs.hosted, jobs.held[1]), (jobs.plain, jobs.variants)):
        for shared in ("", "&shared=1"):
            ask("get", "/api/compare", f"/api/compare?a={a}&b={b}{shared}", 200)
            modes.add(ask("get", "/api/compare/expression", f"/api/compare/expression?a={a}&b={b}{shared}", 200)["mode"])
        ask("get", "/api/compare/gene", f"/api/compare/gene?a={a}&b={b}&gene=ALPHA", 200)
        ask("get", "/api/compare/gene", f"/api/compare/gene?a={a}&b={b}&gene=NOPE", 404)
    assert "none" in modes and len(modes) > 1, "the pairs cover a comparison with a test and one without: " + str(modes)


def test_the_list_routes_answer_in_their_shape(site: SimpleNamespace) -> None:
    ask = site.ask
    made = ask("post", "/api/lists", "/api/lists", 201, json={"jobs": [{"url": "/jobs/with_variants", "label": "x", "name": "n", "at": 5}], "private": False})
    ask("get", "/api/lists/{code}", f"/api/lists/{made['code']}", 200)
    ask("get", "/api/lists/{code}", "/api/lists/nope", 400)
    merged = ask("put", "/api/lists/{code}", f"/api/lists/{made['code']}", 200, json={"jobs": [{"url": "/jobs/with_expression", "label": "y", "name": "", "at": 9.5, "edited": 12}]})
    assert len(merged["jobs"]) == 2, "the list has both jobs: " + str(merged)


def test_every_route_of_the_interface_was_asked(site: SimpleNamespace) -> None:
    declared = {(method, path) for path, methods in site.spec["paths"].items() for method in methods}
    # A link is read from another server. Its answer is built as the answer of an upload is.
    link, upload = (site.spec["paths"][path]["post"]["responses"]["201"] for path in ("/api/uploads/link", "/api/uploads"))
    assert link == upload, "the answer of a link read is declared as the answer of an upload"
    left = sorted(declared - site.asked - NOT_JSON - {("post", "/api/uploads/link")})
    assert left == [], "each route with a JSON answer has a test of its shape in this file: " + str(left)
    for method, path in sorted(NOT_JSON):
        assert (method, path) in declared, f"{method} {path} is a route of the interface"


def test_the_description_the_types_of_the_frontend_come_from_is_the_one_of_the_server() -> None:
    from gnome3d_web.app import interface

    kept = Path(__file__).resolve().parents[1] / "frontend" / "src" / "api" / "openapi.json"
    now = json.dumps(interface().openapi(), indent=1, sort_keys=True) + "\n"
    assert kept.read_text() == now, "web/frontend/src/api/openapi.json is the description the server gives. Run web/scripts/write_openapi.py, then npm run api in web/frontend."
