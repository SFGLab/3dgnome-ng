"""The app serves the frontend build for client routes and keeps its API and job routes."""

from __future__ import annotations

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
    dist = root / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html><body>FRONTEND-OK</body></html>")
    (dist / "assets" / "app.js").write_text("console.log('app')")
    os.environ.update(
        DATA_DIR=str(root / "data"),
        RESULTS_DIR=str(root / "results"),
        QUEUE="inline",
        FRONTEND_DIST=str(dist),
        ENGINE_CMD=f"{sys.executable} {Path(__file__).resolve().parent / 'fake_engine.py'}",
    )
    from fastapi.testclient import TestClient

    from gnome3d_web.app import create_app

    c = TestClient(create_app())
    return SimpleNamespace(c=c, dist=dist, root=root, TestClient=TestClient, create_app=create_app)


def test_returns_the_app(site: SimpleNamespace) -> None:
    c, dist, root, TestClient, create_app = site.c, site.dist, site.root, site.TestClient, site.create_app
    for path in ("/", "/method", "/cite", "/model", "/jobs"):
        r = c.get(path)
        assert r.status_code == 200 and "FRONTEND-OK" in r.text, f"{path} returns the app" + ": " + str(r.status_code)
    # An address the site does not have gets the app too, which shows its page for it, with the status 404.
    for path in ("/some/deep/link", "/nothing", "/jobs/sdfsdfd", "/p/werewwerwe", "/jobs/0000000000000000"):
        r = c.get(path)
        assert r.status_code == 404 and "FRONTEND-OK" in r.text, f"{path} is a missing address with the app as its page" + ": " + str(r.status_code)

    r = c.get("/assets/app.js")
    assert r.status_code == 200 and "console.log" in r.text, "an asset is served: " + str(r.status_code)

    assert "FRONTEND-OK" in c.get("/model").text, "/model returns the app"

    r = c.get("/api/catalog")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json"), "the catalog API stays JSON"

    form = {"cell_type": "GMX", "region": "chr1:0-3000000", "factors": "CTCF", "n_models": "2"}
    r = c.post("/api/jobs", data=form)
    assert r.status_code == 201 and r.json()["url"].startswith("/jobs/"), "a job post answers with its job: " + r.text[:80]
    job = r.json()["url"]

    assert c.get("/api/nothing").status_code == 404, "an unknown API path is a 404, not the app"
    assert c.get(job + "/files/..%2F..%2Fcatalog.json").status_code == 404, "a path under a job is a 404, not the app"
    assert c.get("/p/nothing/files/x").status_code == 404, "a path under a private job is a 404, not the app"
    assert "FRONTEND-OK" in c.get(job).text, "a job's page is the app"
    assert c.get(job + "/files/ep.tsv").status_code == 200, "a job's file is the server's"

    assert "site-hide" not in c.get("/").text, "a page without a head carries no flag"
    (dist / "index.html").write_text("<html><head></head><body>FRONTEND-OK</body></html>")
    config = root / "data" / "config.json"
    config.write_text('{"features": {"trios": false, "results": false, "method": true}}')
    flagged = TestClient(create_app())
    r = flagged.get("/")
    assert '<meta name="site-hide" content="results,trios,uniform_chain">' in r.text, "a feature config.json turns off reaches the app in the page: " + r.text[:120]
    config.write_text('{"features": {"trios": false}}')
    os.utime(config, ns=(1, 1))
    assert '<meta name="site-hide" content="trios,uniform_chain">' in flagged.get("/").text, "an edit of config.json is read without a restart"
    config.write_text('{"features": {"trois": false}}')
    os.utime(config, ns=(2, 2))
    assert '<meta name="site-hide" content="trios,uniform_chain">' in flagged.get("/").text, "a broken edit keeps the configuration the site had"
    config.write_text('{"features": {"uniform_chain": true}}')
    os.utime(config, ns=(3, 3))
    assert "site-hide" not in flagged.get("/").text, "the choice of the chain is on when config.json turns it on"
    for text, said in (('{"features": {"trois": false}}', "trois"), ('{"features": {"trios": "no"}}', "true or false"), ('{"feature": {}}', "unknown key"), ("{", "not JSON")):
        config.write_text(text)
        try:
            create_app()
            assert False, f"a site does not start on a config.json with {said}"
        except ValueError as e:
            assert said in str(e), f"a site does not start on a config.json with {said}" + ": " + str(e)
    config.unlink()
    (dist / "index.html").write_text("<html><head></head><body>FRONTEND-OK</body></html>")
    assert '<meta name="site-hide" content="uniform_chain">' in TestClient(create_app()).get("/").text, (
        "a site without config.json shows everything but the choice of the chain"
    )
    (dist / "index.html").write_text("<html><body>FRONTEND-OK</body></html>")

    os.environ["FRONTEND_DIST"] = str(root / "missing")
    bare = TestClient(create_app())
    r = bare.get("/")
    assert r.status_code == 503 and "The frontend is not built" in r.text, "without a build the app says so: " + str(r.status_code)
    assert bare.get("/api/catalog").status_code == 200, "without a build the API still answers"

    # The site under a prefix, with BASE_PATH on the server and a build made for the same path.
    (dist / "index.html").write_text('<html><body>FRONTEND-OK<script src="/new/assets/app.js"></script></body></html>')
    os.environ.update(FRONTEND_DIST=str(dist), BASE_PATH="/new/")
    under = TestClient(create_app())
    for path in ("/new", "/new/", "/new/method", "/new/jobs", "/", "/method"):
        r = under.get(path)
        assert r.status_code == 200 and "FRONTEND-OK" in r.text, f"under a prefix {path} returns the app" + ": " + str(r.status_code)
    for path in ("/newer", "/new/nothing"):
        r = under.get(path)
        assert r.status_code == 404 and "FRONTEND-OK" in r.text, f"under a prefix {path} is a missing address" + ": " + str(r.status_code)
    for path in ("/new/assets/app.js", "/assets/app.js"):
        assert "console.log" in under.get(path).text, f"under a prefix {path} is the asset"
    for path in ("/new/api/catalog", "/api/catalog"):
        r = under.get(path)
        assert r.status_code == 200 and r.headers["content-type"].startswith("application/json"), f"under a prefix {path} is JSON" + ": " + str(r.status_code)
    assert under.get("/new/api/nothing").status_code == 404, "under a prefix an unknown API path is a 404"

    r = under.post("/new/api/jobs", data=form)
    assert r.status_code == 201 and r.json()["url"].startswith("/jobs/"), "under a prefix the API gives the job's client route, without the prefix: " + r.text[:80]

    assert under.get("/new" + r.json()["url"] + "/files/ep.tsv").status_code == 200, "under a prefix a job's file is the server's"


def test_a_build_for_another_path_is_refused_with_both_paths_named(site: SimpleNamespace) -> None:
    dist, TestClient, create_app = site.dist, site.TestClient, site.create_app
    (dist / "index.html").write_text('<html><body>FRONTEND-OK<script src="/assets/app.js"></script></body></html>')
    r = TestClient(create_app()).get("/new/")
    assert r.status_code == 503 and "built for the path /" in r.text and "BASE_PATH /new" in r.text, "a build for another path is refused with both paths named: " + r.text[:120]
    os.environ.pop("BASE_PATH")
    (dist / "index.html").write_text('<html><body>FRONTEND-OK<script src="/new/assets/app.js"></script></body></html>')
    r = TestClient(create_app()).get("/")
    assert r.status_code == 503 and "built for the path /new" in r.text, "a prefixed build at the root is refused: " + r.text[:120]


def test_config_json_sets_the_longest_region_and_the_longest_hicdiffusion_region(site: SimpleNamespace) -> None:
    TestClient, create_app, root = site.TestClient, site.create_app, site.root
    config = root / "data" / "config.json"
    plain = TestClient(create_app()).get("/api/catalog").json()
    assert (plain["max_region_mb"], plain["hicdiffusion_mb"]) == (20, 2), "without the file the limits are 20 Mb and 2 Mb"

    config.write_text('{"limits": {"region_mb": 3, "hicdiffusion_mb": 1}}')
    limited = TestClient(create_app())
    cat = limited.get("/api/catalog").json()
    assert (cat["max_region_mb"], cat["hicdiffusion_mb"]) == (3, 1), "the catalog gives the limits of the file"
    r = limited.get("/api/region", params={"cell": "GMX", "q": "chr1:0-3500000"})
    assert r.status_code == 400 and "3,000,000" in r.json()["error"], "a region over the limit of the file is refused: " + r.text
    form = {"cell_type": "GMX", "region": "chr1:0-1500000", "factors": "CTCF", "n_models": "2", "background": "hicdiffusion"}
    r = limited.post("/api/jobs", data=form)
    said = r.json().get("error", "")
    assert r.status_code == 400 and r.json()["field"] == "background", "a HiCDiffusion job over the limit of the file is refused: " + r.text
    assert "up to 1,000,000 bp" in said and "up to 1 Mb" in said, "the refusal names the limit of the file: " + said

    config.write_text('{"limits": {"region_mb": 30}}')
    os.utime(config, ns=(5, 5))
    cat = limited.get("/api/catalog").json()
    assert (cat["max_region_mb"], cat["hicdiffusion_mb"]) == (30, 2), "an edit of the limits is read without a restart"
    assert limited.get("/api/region", params={"cell": "GMX", "q": "chr1:0-25000000"}).status_code == 200, "a region the new limit allows is taken"


def test_a_job_the_file_allows_runs_past_the_limit_of_the_environment(site: SimpleNamespace) -> None:
    TestClient, create_app, root = site.TestClient, site.create_app, site.root
    os.environ["MAX_REGION_MB"] = "1"
    (root / "data" / "config.json").write_text('{"limits": {"region_mb": 3}}')
    c = TestClient(create_app())
    r = c.post("/api/jobs", data={"cell_type": "GMX", "region": "chr1:0-2000000", "factors": "CTCF", "n_models": "2", "background": "hic"})
    assert r.status_code == 201, "the limit of the file stands over MAX_REGION_MB: " + r.text
    state = c.get("/api" + r.json()["url"]).json()
    assert state["state"] == "done", "the job runs. An accepted request is not held to a limit again: " + str(state)[:300]
    del os.environ["MAX_REGION_MB"]


@pytest.mark.parametrize(
    ("text", "said"),
    [
        ('{"limits": {"hicdiffusion_mb": 5}}', "one window of 2,097,152 bp"),
        ('{"limits": {"region": 5}}', "names the limit region"),
        ('{"limits": {"region_mb": 0}}', "above zero"),
        ('{"limits": {"region_mb": true}}', "above zero"),
        ('{"limits": 20}', "limits must be an object"),
    ],
)
def test_a_site_does_not_start_on_a_limit_it_cannot_keep(site: SimpleNamespace, text: str, said: str) -> None:
    config = site.root / "data" / "config.json"
    config.write_text(text)
    with pytest.raises(ValueError, match=said):
        site.create_app()
    config.unlink()
