"""The head of each page, the sitemap and the files a browser asks for by name."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from fixture import make_data

from gnome3d_web.frontend import PAGES

PAGES_JSON = Path(__file__).resolve().parents[1] / "frontend" / "src" / "content" / "pages.json"
META = json.loads(PAGES_JSON.read_text())


def _site(root: Path, monkeypatch: pytest.MonkeyPatch, **env: str) -> TestClient:
    make_data(root / "data")
    dist = root / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html><head><title>the title of the build</title></head><body>APP</body></html>")
    (dist / "pages.json").write_text(PAGES_JSON.read_text())
    (dist / "favicon.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    for key in ("PUBLIC_URL", "BASE_PATH", "SITE_CONFIG"):
        monkeypatch.delenv(key, raising=False)
    for key, value in {"DATA_DIR": str(root / "data"), "RESULTS_DIR": str(root / "results"), "QUEUE": "inline", "FRONTEND_DIST": str(dist), **env}.items():
        monkeypatch.setenv(key, value)
    from gnome3d_web.app import create_app

    return TestClient(create_app())


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A site with a public address at the root of its host."""
    return _site(tmp_path, monkeypatch, PUBLIC_URL="https://example.org/")


def test_the_build_describes_every_page_of_the_app() -> None:
    assert set(META["pages"]) == PAGES, "pages.json has a title for each page of the app and for no other"
    for key, page in META["pages"].items():
        assert page["title"] and page["description"], f"the page {key!r} has a title and a description"
        assert len(page["description"]) <= 200, f"the description of {key!r} fits a search result"


def test_the_title_page_has_its_head(site: TestClient) -> None:
    text = site.get("/").text
    assert f"<title>{META['pages']['']['title']}</title>" in text, "the title page has its own title and not that of the build"
    assert "the title of the build" not in text, "the title of the build is replaced"
    assert '<meta name="description" content="3D-GNOME NG is a free web server' in text, "the description of the title page"
    assert '<link rel="canonical" href="https://example.org/">' in text, "the canonical link is the public address"
    assert '<meta property="og:image" content="https://example.org/og.png">' in text, "the picture of a shared link"
    assert '"@type": "WebApplication"' in text and '"alternateName": ["3D-GNOME NG", "3D-GNOME"]' in text, "the title page describes the tool to a search engine"
    assert "noindex" not in text, "the title page may be indexed"


def test_a_page_has_its_own_title_and_link(site: TestClient) -> None:
    text = site.get("/help").text
    assert "<title>Help | 3D-GNOMEᴺᴳ</title>" in text, "the title of a page ends with the tool"
    assert '<link rel="canonical" href="https://example.org/help">' in text, "the canonical link of a page"
    assert "application/ld+json" not in text, "the description of the tool is on the title page alone"


def test_private_and_missing_pages_are_not_indexed(site: TestClient) -> None:
    jobs = site.get("/jobs").text
    assert '<meta name="robots" content="noindex">' in jobs and "canonical" not in jobs, "the job list of a browser is not indexed"
    missing = site.get("/nothing")
    assert missing.status_code == 404 and '<meta name="robots" content="noindex">' in missing.text, "a missing address is not indexed"
    assert "<title>Page not found | 3D-GNOMEᴺᴳ</title>" in missing.text, "a missing address has the title of the missing page"
    job = site.post("/api/jobs", data={"cell_type": "GMX", "region": "chr1:0-3000000", "factors": "CTCF", "n_models": "2"}).json()["url"]
    page = site.get(job)
    assert '<meta name="robots" content="noindex">' in page.text and "<title>Job | 3D-GNOMEᴺᴳ</title>" in page.text, "the page of a job is not indexed"
    assert page.headers["X-Robots-Tag"] == "noindex", "the page of a job carries the mark in its headers too"
    assert site.get("/api/catalog").headers["X-Robots-Tag"] == "noindex", "an answer of the API is not indexed"
    assert "X-Robots-Tag" not in site.get("/help").headers, "a page of the app carries no such header"


def test_the_sitemap_lists_the_public_pages(site: TestClient, tmp_path: Path) -> None:
    r = site.get("/sitemap.xml")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/xml"), "the sitemap is XML"
    for key in ("", "model", "examples", "method", "results", "help", "cite", "privacy"):
        assert f"<loc>https://example.org/{key}</loc>" in r.text, f"the sitemap has the page {key!r}"
    assert "https://example.org/jobs<" not in r.text, "the job list of a browser is not in the sitemap"
    (tmp_path / "data" / "config.json").write_text('{"features": {"results": false}}')
    assert "https://example.org/results<" not in site.get("/sitemap.xml").text, "a page the site hides is not in the sitemap"
    assert '<meta name="robots" content="noindex">' in site.get("/results").text, "a page the site hides is not indexed"


def test_the_files_a_browser_asks_for_by_name(site: TestClient) -> None:
    r = site.get("/favicon.svg")
    assert r.status_code == 200 and r.headers["content-type"].startswith("image/svg+xml"), "the icon is served with its type"
    assert site.get("/og.png").status_code == 404, "a file the build lacks is a missing address"
    robots = site.get("/robots.txt").text
    assert "Disallow: /api/" in robots and "Disallow: /p/" in robots and "Sitemap: https://example.org/sitemap.xml" in robots, "robots.txt keeps the API and the jobs out"


def test_a_site_under_a_base_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c = _site(tmp_path, monkeypatch, PUBLIC_URL="https://example.org", BASE_PATH="/new")
    text = c.get("/new/help").text
    assert '<link rel="canonical" href="https://example.org/new/help">' in text, "the canonical link has the base path"
    assert "<loc>https://example.org/new/</loc>" in c.get("/new/sitemap.xml").text, "the sitemap has the base path"


def test_a_site_without_a_public_address(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c = _site(tmp_path, monkeypatch)
    text = c.get("/help").text
    assert "<title>Help | 3D-GNOMEᴺᴳ</title>" in text and "canonical" not in text, "a page has its title and no canonical link"
    assert c.get("/sitemap.xml").status_code == 404, "a site without a public address has no sitemap"
