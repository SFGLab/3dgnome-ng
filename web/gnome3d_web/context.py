"""The site a request is answered by: its settings, its hosted data, its limits and its jobs.

The module is not named site or http, which are modules of Python.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import Depends, Request

from .catalog.model import Catalog, CellType
from .errors import UnknownCell
from .jobs import store
from .jobs.workers import JobQueue, queue_of
from .lists.store import Lists
from .regions import HICDIFFUSION_SPAN
from .settings import Settings
from .siteconfig import ConfigFile


@dataclass
class Site:
    settings: Settings
    catalog: Catalog
    config: ConfigFile
    lists: Lists
    queue: JobQueue
    # A link read holds a worker thread for up to READ_SECONDS. Past the limit a read is refused
    # at once and does not wait behind the others.
    link_slots: threading.BoundedSemaphore

    @classmethod
    def from_env(cls) -> Site:
        s = Settings.from_env()
        return cls(
            settings=s,
            catalog=Catalog.load(s.data_dir),
            config=ConfigFile(s.site_config or s.data_dir / "config.json"),
            lists=Lists(s.lists_dir),
            queue=queue_of(s),
            link_slots=threading.BoundedSemaphore(max(1, s.link_reads)),
        )

    def max_span(self) -> int:
        """The longest region of a job in bp, from config.json or else from MAX_REGION_MB."""
        mb = self.config.get().region_mb
        return self.settings.max_span if mb is None else int(mb * 1_000_000)

    def hicdiffusion_span(self) -> int:
        """The longest region of a job with a HiCDiffusion background in bp."""
        mb = self.config.get().hicdiffusion_mb
        return HICDIFFUSION_SPAN if mb is None else int(mb * 1_000_000)

    def cell(self, name: str) -> CellType:
        """The cell type of a request.

        @raises UnknownCell for a name the catalog lacks
        """
        if name not in self.catalog.cell_types:
            raise UnknownCell(name)
        return self.catalog.cell_types[name]

    def base_cell(self, name: str) -> CellType:
        """The cell type whose genes and hosted data a request reads: the named one, or the first
        of the catalog for a request on a user's own cell type."""
        return self.catalog.cell_types.get(name) or next(iter(self.catalog.cell_types.values()))

    def job(self, kind: str, key: str) -> Path | None:
        """The directory of a job the site has, by its address: `jobs` and the key of a shared
        job, `p` and the token of a private one."""
        return store.find(self.settings.jobs_dir if kind == "jobs" else self.settings.private_dir, key)


def site_of(request: Request) -> Site:
    return request.app.state.site


Here = Annotated[Site, Depends(site_of)]
