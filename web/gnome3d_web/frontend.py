"""The pages of the app: the built frontend for every address that is no route of the server."""

from __future__ import annotations

import re

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .context import Site

# The pages of the app. Any other address gets the app too, which shows its page of a missing address.
PAGES = {"", "model", "jobs", "examples", "method", "results", "help", "cite", "privacy"}


def known(site: Site, path: str) -> bool:
    """Whether an address is a page of the app or the page of a job the site has."""
    path = path.strip("/")
    if path in PAGES:
        return True
    m = re.fullmatch(r"(jobs|p)/([^/]+)", path)
    return m is not None and site.job(m[1], m[2]) is not None


def serve(app: FastAPI, site: Site) -> None:
    """Adds the assets of the build and the route of every other address. Called after every
    other route is added: a path no route above answers is a client route of the frontend."""
    s = site.settings
    index_html = s.frontend_dist / "index.html"
    if (s.frontend_dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=s.frontend_dist / "assets"), name="assets")

    @app.get("/{path:path}", response_class=HTMLResponse, include_in_schema=False)
    def frontend(path: str) -> HTMLResponse:
        # A job's own page is the app's; the paths below it are the server's.
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
        # The features config.json turns off reach the app in the page itself. It never draws
        # one first.
        off = site.config.get().hidden
        if off:
            html = html.replace("<head>", f'<head><meta name="site-hide" content="{",".join(off)}">', 1)
        # The page names its scripts by their hash. A browser asks for the page anew each time. A
        # new deployment is seen on the next load.
        # An address the site does not have is answered with the app and the status 404.
        return HTMLResponse(html, 200 if known(site, path) else 404, headers={"Cache-Control": "no-cache"})
