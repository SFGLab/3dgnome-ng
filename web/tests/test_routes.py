"""The addresses of the site.

The pages of the site, links in the wild and scripts call these addresses. A change of this list is
a change of the interface of the site and is made here on purpose.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from fixture import make_data

RESULTS = ("annotations", "contacts", "distance", "files.zip", "genes", "log", "loops", "picks", "structure/{i}", "summary")
ROUTES = sorted(
    [
        ("GET", "/api/catalog"),
        ("GET", "/api/compare"),
        ("GET", "/api/compare/expression"),
        ("GET", "/api/compare/gene"),
        ("GET", "/api/expression"),
        ("GET", "/api/genes"),
        ("GET", "/api/preview"),
        ("GET", "/api/region"),
        ("GET", "/api/settings"),
        ("POST", "/api/settings/ini"),
        ("GET", "/api/sv_sets/{name}/samples"),
        ("POST", "/api/jobs"),
        ("GET", "/api/jobs/{key}"),
        ("GET", "/api/p/{key}"),
        ("GET", "/jobs/{key}/files/{name}"),
        ("GET", "/p/{key}/files/{name}"),
        ("POST", "/api/uploads"),
        ("POST", "/api/uploads/link"),
        ("GET", "/api/uploads/{uid}/preview"),
        ("POST", "/api/variants/check"),
        ("POST", "/api/lists"),
        ("GET", "/api/lists/{code}"),
        ("PUT", "/api/lists/{code}"),
    ]
    + [("GET", f"/api/{kind}/{{key}}/{part}") for kind in ("jobs", "p") for part in RESULTS]
)


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    root = Path(str(tmp_path_factory.mktemp("work")))
    make_data(root / "data")
    os.environ.update(DATA_DIR=str(root / "data"), RESULTS_DIR=str(root / "results"), QUEUE="inline", FRONTEND_DIST=str(root / "missing"))
    from gnome3d_web.app import create_app

    return TestClient(create_app())


def test_the_site_answers_on_its_addresses_and_no_others(client: TestClient) -> None:
    described = client.get("/openapi.json").json()["paths"]
    routes = sorted((method.upper(), path) for path, methods in described.items() for method in methods)
    assert routes == ROUTES, "the routes of the app are the list of this file: " + str(sorted(set(routes) ^ set(ROUTES)))


def test_the_description_of_the_interface_is_served(client: TestClient) -> None:
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 200, f"{path} answers"


def test_every_other_address_gets_the_app_and_none_under_api(client: TestClient) -> None:
    r = client.get("/some/page")
    assert r.status_code == 503 and "frontend is not built" in r.text, "an address of no route goes to the frontend, which is not built here: " + r.text[:120]
    r = client.get("/api/nothing")
    assert r.status_code == 404 and r.json() == {"detail": "Not Found"}, "an address under api that no route has is a 404 of the server: " + r.text[:120]
    assert client.get("/api/catalog").headers["content-type"].startswith("application/json"), "the route of every other address takes no address of another route"
