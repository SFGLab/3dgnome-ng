"""The pages of the app. The built frontend answers every address that is no route of the server.

The build holds `pages.json`, the title and the description of each page. The server gives a page
its own title, description and link tags in the head of the app. A search engine reads them
without running the app. A job page and a page of one browser ask not to be indexed.
"""

from __future__ import annotations

import html as markup
import json
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .context import Site

# The pages of the app. Any other address gets the app too, which shows its page of a missing address.
PAGES = {"", "model", "jobs", "examples", "method", "results", "help", "cite", "privacy"}
# The files of the build beside index.html that a browser or a search engine asks for by name.
ROOT_FILES = {
    "favicon.svg": "image/svg+xml",
    "favicon.ico": "image/x-icon",
    "favicon-32.png": "image/png",
    "apple-touch-icon.png": "image/png",
    "icon-192.png": "image/png",
    "icon-512.png": "image/png",
    "og.png": "image/png",
    "site.webmanifest": "application/manifest+json",
}
JOB_PAGE = re.compile(r"(jobs|p)/[^/]+")


def page_meta(dist: Path) -> dict[str, Any] | None:
    """Returns the titles and descriptions of the pages from the build, None for a build without them."""
    path = dist / "pages.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def indexed(meta: dict[str, Any], hidden: tuple[str, ...]) -> list[str]:
    """Returns the pages a search engine may index, in the order of the build."""
    return [key for key, page in meta["pages"].items() if not page.get("private") and key not in hidden]


def head(meta: dict[str, Any], path: str, found: bool, hidden: tuple[str, ...], address: str) -> tuple[str, str]:
    """Returns the title of a page and the tags of its head.

    Parameters
    ----------
    meta
        The content of `pages.json`.
    path
        The path of the page without the base path.
    found
        False for an address the site does not have.
    hidden
        The features that the configuration of the site turns off.
    address
        The public address of the site with its base path and no slash at its end. Empty leaves
        out the canonical link and the address of the picture.
    """
    key = path.strip("/")
    page = meta["pages"].get(key) if found else None
    if page is None:
        page = meta["job"] if found and JOB_PAGE.fullmatch(key) else meta["missing"]
    title = page["title"] if key == "" and found else f"{page['title']} | {meta['site']}"
    private = bool(page.get("private")) or key in hidden
    esc = markup.escape
    tags = ['<meta name="robots" content="noindex">'] if private else []
    description = page.get("description", "")
    if description:
        tags.append(f'<meta name="description" content="{esc(description)}">')
    if not private:
        tags += [
            f'<meta property="og:title" content="{esc(title)}">',
            f'<meta property="og:description" content="{esc(description)}">',
            '<meta property="og:type" content="website">',
            f'<meta property="og:site_name" content="{esc(meta["plain"])}">',
            '<meta name="twitter:card" content="summary_large_image">',
        ]
        if address:
            link = f"{address}/{key}"
            tags += [
                f'<link rel="canonical" href="{esc(link)}">',
                f'<meta property="og:url" content="{esc(link)}">',
                f'<meta property="og:image" content="{esc(address)}/og.png">',
            ]
        if key == "":
            about = {
                "@context": "https://schema.org",
                "@type": "WebApplication",
                "name": meta["site"],
                "alternateName": [meta["plain"], "3D-GNOME"],
                "description": description,
                "applicationCategory": "ScienceApplication",
                "operatingSystem": "Any",
                "isAccessibleForFree": True,
                "license": "https://www.gnu.org/licenses/gpl-3.0.html",
                "offers": {"@type": "Offer", "price": "0", "priceCurrency": "EUR"},
            }
            if address:
                about["url"] = f"{address}/"
            tags.append('<script type="application/ld+json">' + json.dumps(about, ensure_ascii=False).replace("</", "<\\/") + "</script>")
    return title, "".join(tags)


def known(site: Site, path: str) -> bool:
    """Returns true for an address that is a page of the app or the page of a job the site has."""
    path = path.strip("/")
    if path in PAGES:
        return True
    m = re.fullmatch(r"(jobs|p)/([^/]+)", path)
    return m is not None and site.job(m[1], m[2]) is not None


def serve(app: FastAPI, site: Site) -> None:
    """Adds the assets of the build and the route of every other address. Call it after every other
    route is added. A path that no earlier route answers is a client route of the frontend."""
    s = site.settings
    index_html = s.frontend_dist / "index.html"
    if (s.frontend_dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=s.frontend_dist / "assets"), name="assets")
    address = f"{s.public_url}{s.base_path}" if s.public_url else ""

    @app.middleware("http")
    async def keep_out_of_the_index(request: Request, call_next: Any) -> Response:
        """Asks a search engine not to index the answers of the API and the files of a job."""
        response = await call_next(request)
        # This step is outside the one that takes the base path off an address.
        path = request.url.path.removeprefix(s.base_path).lstrip("/")
        if path.startswith("api/") or re.match(r"(jobs|p)/[^/]+", path):
            response.headers["X-Robots-Tag"] = "noindex"
        return response

    @app.get("/robots.txt", include_in_schema=False)
    def robots() -> PlainTextResponse:
        # A search engine reads this file at the root of a host alone. A site under a base path
        # relies on the noindex marks of its pages.
        lines = ["User-agent: *", f"Disallow: {s.base_path}/api/", f"Disallow: {s.base_path}/jobs/", f"Disallow: {s.base_path}/p/"]
        if address:
            lines.append(f"Sitemap: {address}/sitemap.xml")
        return PlainTextResponse("\n".join(lines) + "\n")

    @app.get("/sitemap.xml", include_in_schema=False)
    def sitemap() -> Response:
        meta = page_meta(s.frontend_dist)
        if not address or meta is None:
            raise HTTPException(404)
        urls = "".join(f"<url><loc>{markup.escape(f'{address}/{key}')}</loc></url>" for key in indexed(meta, site.config.get().hidden))
        return Response(f'<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>', media_type="application/xml")

    @app.get("/{path:path}", response_class=HTMLResponse, include_in_schema=False)
    def frontend(path: str) -> Response:
        if path in ROOT_FILES and (s.frontend_dist / path).is_file():
            return FileResponse(s.frontend_dist / path, media_type=ROOT_FILES[path], headers={"Cache-Control": "public, max-age=86400"})
        # The page of a job is a page of the app. The paths below it are routes of the server.
        if path.startswith(("api/", "assets/")) or re.match(r"(jobs|p)/[^/]+/", path):
            raise HTTPException(404)
        if not index_html.is_file():
            return HTMLResponse(f"The frontend is not built. Run npm run build in web/frontend, or set FRONTEND_DIST. Expected {index_html}.", 503)
        html = index_html.read_text()
        built = re.search(r'(?:src|href)="([^"]*)/assets/', html)
        if built and built[1] != s.base_path:
            return HTMLResponse(
                f"The frontend is built for the path {built[1] or '/'} and the server has BASE_PATH {s.base_path or '/'}. "
                f"Build with BASE_PATH={s.base_path or '/'} npm run build in web/frontend, or set BASE_PATH to match.",
                503,
            )
        # The features that config.json turns off reach the app in the page itself. The app does not
        # draw one of them first.
        off = site.config.get().hidden
        if off:
            html = html.replace("<head>", f'<head><meta name="site-hide" content="{",".join(off)}">', 1)
        found = known(site, path)
        meta = page_meta(s.frontend_dist)
        if meta is not None and "</head>" in html:
            title, tags = head(meta, path, found, off, address)
            html = re.sub(r"<title>.*?</title>", "", html, count=1, flags=re.S)
            html = html.replace("</head>", f"<title>{markup.escape(title)}</title>{tags}</head>", 1)
        # The page refers to its scripts by their hash. A browser asks for the page anew each time.
        # A new deployment is seen on the next load.
        return HTMLResponse(html, 200 if found else 404, headers={"Cache-Control": "no-cache"})
