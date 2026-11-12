"""Running jobs: rq workers, or a local pool of slots in the web process, one GPU per slot."""

from __future__ import annotations

import queue
import shutil
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from .catalog import Catalog
from .jobs import JobRequest, run_job
from .settings import Settings


def work(req_dict: dict[str, object], job_dir: str, gpu: str | None = None, engine: list[str] | None = None) -> str:
    """The function a worker runs, with everything it needs read from the environment. `gpu`
    is the only GPU the engine sees, all of the worker's GPUs when None."""
    s = Settings.from_env()
    catalog = Catalog.load(s.data_dir)
    extra = {"CUDA_VISIBLE_DEVICES": gpu} if gpu is not None else None
    run_job(JobRequest(**req_dict), catalog, Path(job_dir), engine=engine or s.engine_cmd, env_extra=extra)  # type: ignore[arg-type]
    return job_dir


class LocalPool:
    """Runs jobs in the web process on a fixed set of slots, one GPU id per slot. A job waits
    for a free slot, and its engine sees only the slot's GPU. Waiting jobs are lost when the
    process stops."""

    def __init__(self, slots: list[str], engine: list[str] | None = None) -> None:
        self._jobs: queue.Queue[tuple[dict[str, object], str]] = queue.Queue()
        self._engine = engine
        for gpu in slots:
            threading.Thread(target=self._serve, args=(gpu,), daemon=True).start()

    def _serve(self, gpu: str) -> None:
        while True:
            req_dict, job_dir = self._jobs.get()
            try:
                work(req_dict, job_dir, gpu, self._engine)
            except Exception:  # noqa: BLE001
                # A job that raises before run_job records it must not end the slot.
                traceback.print_exc()
            finally:
                self._jobs.task_done()

    def submit(self, req_dict: dict[str, object], job_dir: str) -> None:
        """Queues one job."""
        self._jobs.put((req_dict, job_dir))

    def join(self) -> None:
        """Waits until every queued job has finished."""
        self._jobs.join()

    def position(self, job_dir: str) -> int | None:
        """The number of waiting jobs in front of a job, None when it is not waiting."""
        with self._jobs.mutex:
            waiting = [d for _, d in self._jobs.queue]
        return waiting.index(job_dir) if job_dir in waiting else None


_POOL: LocalPool | None = None
_POOL_LOCK = threading.Lock()


def _local_pool(s: Settings) -> LocalPool:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = LocalPool(s.gpu_slots, s.engine_cmd)
        return _POOL


def queue_position(s: Settings, job_dir: Path) -> int | None:
    """The number of jobs waiting in front of a job, None when it is not waiting."""
    if s.queue == "thread":
        return _local_pool(s).position(str(job_dir))
    if s.queue == "inline":
        return None
    from redis import Redis
    from rq import Queue

    q = Queue("gnome3d", connection=Redis.from_url(s.redis_url))
    for i, job in enumerate(q.jobs):
        if len(job.args) > 1 and job.args[1] == str(job_dir):
            return i
    return None


def enqueue(s: Settings, req: JobRequest, job_dir: Path) -> None:
    if s.queue == "inline":
        # Synchronous, so a request returns after its job; for development and tests.
        work(asdict(req), str(job_dir))
        return
    if s.queue == "thread":
        _local_pool(s).submit(asdict(req), str(job_dir))
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
