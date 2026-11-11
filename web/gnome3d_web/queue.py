"""Running jobs: rq workers in production, a thread of the web process for development."""

from __future__ import annotations

import shutil
import threading
import time
from dataclasses import asdict
from pathlib import Path

from .catalog import Catalog
from .jobs import JobRequest, run_job
from .settings import Settings


def work(req_dict: dict[str, object], job_dir: str) -> str:
    """The function a worker runs, with everything it needs read from the environment."""
    s = Settings.from_env()
    catalog = Catalog.load(s.data_dir)
    run_job(JobRequest(**req_dict), catalog, Path(job_dir), engine=s.engine_cmd)  # type: ignore[arg-type]
    return job_dir


def enqueue(s: Settings, req: JobRequest, job_dir: Path) -> None:
    if s.queue == "inline":
        # Synchronous, so a request returns after its job; for development and tests.
        work(asdict(req), str(job_dir))
        return
    if s.queue == "thread":
        threading.Thread(target=work, args=(asdict(req), str(job_dir)), daemon=True).start()
        return
    from redis import Redis
    from rq import Queue

    Queue("gnome3d", connection=Redis.from_url(s.redis_url), default_timeout=6 * 3600).enqueue(
        work, asdict(req), str(job_dir), job_timeout=6 * 3600
    )


def sweep_private(results_dir: Path, retention_days: int) -> list[str]:
    """Remove private jobs older than the retention period, as the form promises users, and
    return their names. Shared jobs are never removed here."""
    gone = []
    cutoff = time.time() - retention_days * 86400
    root = results_dir / "private"
    if not root.is_dir():
        return gone
    for d in root.iterdir():
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d)
            gone.append(d.name)
    return gone
