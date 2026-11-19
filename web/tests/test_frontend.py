"""The app serves the frontend build for client routes and keeps its API and job routes.

    python web/tests/test_frontend.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from _check import check, done
from fixture import make_data


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
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
            ENGINE_CMD=f"{Path(__file__).resolve().parents[2] / '.venv' / 'bin' / 'python'} {Path(__file__).resolve().parent / 'fake_engine.py'}",
        )
        from fastapi.testclient import TestClient

        from gnome3d_web.app import create_app

        c = TestClient(create_app())
        for path in ("/", "/method", "/cite", "/some/deep/link"):
            r = c.get(path)
            check(f"{path} returns the app", r.status_code == 200 and "FRONTEND-OK" in r.text, str(r.status_code))
        r = c.get("/assets/app.js")
        check("an asset is served", r.status_code == 200 and "console.log" in r.text, str(r.status_code))
        r = c.get("/model/classic")
        check("/model/classic returns the old form", r.status_code == 200 and "GMX, lymphoblastoid" in r.text and "FRONTEND-OK" not in r.text)
        check("/model returns the app", "FRONTEND-OK" in c.get("/model").text)
        r = c.get("/api/catalog")
        check("the catalog API stays JSON", r.status_code == 200 and r.headers["content-type"].startswith("application/json"))
        form = {"cell_type": "GMX", "region": "chr1:0-3000000", "factors": "CTCF", "n_models": "2"}
        r = c.post("/jobs", data=form, follow_redirects=False)
        check("a job post redirects to its job", r.status_code == 303 and r.headers["location"].startswith("/jobs/"))
        page = c.get(r.headers["location"] + "/classic")
        check("the job page is the server's, not the app", page.status_code == 200 and "FRONTEND-OK" not in page.text and "done" in page.text)
        check("an unknown API path is a 404, not the app", c.get("/api/nothing").status_code == 404)
        check("a path under a job is a 404, not the app", c.get(r.headers["location"] + "/files/..%2F..%2Fcatalog.json").status_code == 404)
        check("a path under a private job is a 404, not the app", c.get("/p/nothing/files/x").status_code == 404)
        check("a job's page is the app", "FRONTEND-OK" in c.get(r.headers["location"]).text)
        check("a job's file is the server's", c.get(r.headers["location"] + "/files/ep.tsv").status_code == 200)

        check("no part is hidden by default", "site-hide" not in c.get("/").text)
        (dist / "index.html").write_text("<html><head></head><body>FRONTEND-OK</body></html>")
        config = root / "data" / "config.json"
        config.write_text('{"features": {"trios": false, "results": false, "method": true}}')
        flagged = TestClient(create_app())
        r = flagged.get("/")
        check("a feature config.json turns off reaches the app in the page", '<meta name="site-hide" content="results,trios">' in r.text, r.text[:120])
        config.write_text('{"features": {"trios": false}}')
        os.utime(config, ns=(1, 1))
        check("an edit of config.json is read without a restart", '<meta name="site-hide" content="trios">' in flagged.get("/").text)
        config.write_text('{"features": {"trois": false}}')
        os.utime(config, ns=(2, 2))
        check("a broken edit keeps the configuration the site had", '<meta name="site-hide" content="trios">' in flagged.get("/").text)
        for text, said in (('{"features": {"trois": false}}', "trois"), ('{"features": {"trios": "no"}}', "true or false"), ('{"feature": {}}', "unknown key"), ("{", "not JSON")):
            config.write_text(text)
            try:
                create_app()
                check(f"a site does not start on a config.json with {said}", False)
            except ValueError as e:
                check(f"a site does not start on a config.json with {said}", said in str(e), str(e))
        config.unlink()
        check("a site without config.json shows everything", "site-hide" not in TestClient(create_app()).get("/").text)
        (dist / "index.html").write_text("<html><body>FRONTEND-OK</body></html>")

        os.environ["FRONTEND_DIST"] = str(root / "missing")
        bare = TestClient(create_app())
        r = bare.get("/")
        check("without a build the app says so", r.status_code == 503 and "The frontend is not built" in r.text, str(r.status_code))
        check("without a build /model/classic still works", bare.get("/model/classic").status_code == 200)

        # The site under a prefix: BASE_PATH on the server and a build made for the same path.
        (dist / "index.html").write_text('<html><body>FRONTEND-OK<script src="/new/assets/app.js"></script></body></html>')
        os.environ.update(FRONTEND_DIST=str(dist), BASE_PATH="/new/")
        under = TestClient(create_app())
        for path in ("/new", "/new/", "/new/method", "/new/jobs", "/", "/method", "/newer"):
            r = under.get(path)
            check(f"under a prefix {path} returns the app", r.status_code == 200 and "FRONTEND-OK" in r.text, str(r.status_code))
        for path in ("/new/assets/app.js", "/assets/app.js"):
            check(f"under a prefix {path} is the asset", "console.log" in under.get(path).text)
        for path in ("/new/api/catalog", "/api/catalog"):
            r = under.get(path)
            check(f"under a prefix {path} is JSON", r.status_code == 200 and r.headers["content-type"].startswith("application/json"), str(r.status_code))
        check("under a prefix an unknown API path is a 404", under.get("/new/api/nothing").status_code == 404)
        r = under.post("/new/api/jobs", data=form)
        check("under a prefix the API gives the job's client route, without the prefix", r.status_code == 201 and r.json()["url"].startswith("/jobs/"), r.text[:80])
        r = under.post("/new/jobs", data=form, follow_redirects=False)
        check("under a prefix a job post redirects under the prefix", r.status_code == 303 and r.headers["location"].startswith("/new/jobs/"), r.headers.get("location", ""))
        page = under.get(r.headers["location"] + "/classic")
        check("under a prefix the old job page links its files under the prefix", page.status_code == 200 and 'href="/new/jobs/' in page.text and 'href="/new/"' in page.text)
        check("under a prefix a job's file is the server's", under.get(r.headers["location"] + "/files/ep.tsv").status_code == 200)
        check("under a prefix the old form posts under the prefix", 'action="/new/jobs"' in under.get("/new/model/classic").text)

        (dist / "index.html").write_text('<html><body>FRONTEND-OK<script src="/assets/app.js"></script></body></html>')
        r = TestClient(create_app()).get("/new/")
        check("a build for another path is refused with both paths named", r.status_code == 503 and "built for the path /" in r.text and "BASE_PATH /new" in r.text, r.text[:120])
        os.environ.pop("BASE_PATH")
        (dist / "index.html").write_text('<html><body>FRONTEND-OK<script src="/new/assets/app.js"></script></body></html>')
        r = TestClient(create_app()).get("/")
        check("a prefixed build at the root is refused", r.status_code == 503 and "built for the path /new" in r.text, r.text[:120])
    return done()


if __name__ == "__main__":
    sys.exit(main())
