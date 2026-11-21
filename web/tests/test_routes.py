"""The addresses the site answers on.

    pytest web/tests/test_routes.py

The pages of the site, links in the wild and scripts call these addresses. A change of this list
is a change of the site's interface and is made here on purpose.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
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
        ("GET", "/api/jobs/{key}/files"),
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
        # The description of the interface that FastAPI serves.
        ("GET", "/docs"),
        ("GET", "/docs/oauth2-redirect"),
        ("GET", "/openapi.json"),
        ("GET", "/redoc"),
        # Every other address gets the app or the page of a missing address.
        ("GET", "/{path:path}"),
    ]
    + [("GET", f"/api/{kind}/{{key}}/{part}") for kind in ("jobs", "p") for part in RESULTS]
)


@pytest.fixture(scope="module")
def routes(tmp_path_factory: pytest.TempPathFactory) -> list[tuple[str, str]]:
    root = Path(str(tmp_path_factory.mktemp("work")))
    make_data(root / "data")
    os.environ.update(DATA_DIR=str(root / "data"), RESULTS_DIR=str(root / "results"), QUEUE="inline", FRONTEND_DIST=str(root / "missing"))
    from gnome3d_web.app import create_app

    out = []
    for route in create_app().routes:
        out += [(m, route.path) for m in getattr(route, "methods", None) or ["MOUNT"] if m not in ("HEAD", "OPTIONS")]  # type: ignore[attr-defined]
    return out


def test_the_site_answers_on_its_addresses_and_no_others(routes: list[tuple[str, str]]) -> None:
    assert sorted(routes) == ROUTES, "the routes of the app are the list of this file: " + str(sorted(set(routes) ^ set(ROUTES)))


def test_no_address_is_registered_twice(routes: list[tuple[str, str]]) -> None:
    twice = sorted({r for r in routes if routes.count(r) > 1})
    assert twice == [], "each method and path has one route: " + str(twice)


def test_the_route_of_every_other_address_is_the_last(routes: list[tuple[str, str]]) -> None:
    assert routes[-1] == ("GET", "/{path:path}"), "the route of every other address is registered last and takes no address of another route: " + str(routes[-1])
