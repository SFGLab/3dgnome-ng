"""The web app. It has the routes of every part of the site and the built frontend."""

from __future__ import annotations

from fastapi import FastAPI

from . import frontend
from .catalog import routes as catalog
from .engine import routes as engine
from .answers import StripBase, handle_errors
from .jobs import routes as jobs
from .jobs.store import sweep_private
from .lists import routes as lists
from .results import routes as results
from .context import Site
from .uploads import routes as uploads
from .variants import routes as variants


def interface() -> FastAPI:
    """Returns the app with every route and no site behind it. It describes the interface of the
    site at /openapi.json. The types of the frontend are generated from that description."""
    app = FastAPI(title="3D-GNOME")
    handle_errors(app)
    for part in (jobs.router, uploads.router, variants.router, catalog.router, engine.router, results.job_routes("jobs"), results.job_routes("p"), results.router, lists.router):
        app.include_router(part)
    return app


def create_app() -> FastAPI:
    site = Site.from_env()
    s = site.settings
    app = interface()
    app.state.site = site
    if s.base_path:
        app.add_middleware(StripBase, base=s.base_path)
    s.jobs_dir.mkdir(parents=True, exist_ok=True)
    s.private_dir.mkdir(parents=True, exist_ok=True)
    sweep_private(s.results_dir, s.retention_days)
    site.queue.resume()
    frontend.serve(app, site)
    return app
