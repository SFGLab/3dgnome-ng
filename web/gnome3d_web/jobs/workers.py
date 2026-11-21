"""Running jobs: rq workers, or a local pool of slots in the web process, one card per slot.

`Settings` has the workers of the site: the cards, the jobs at once on a card, the cores of a
job and its GPU memory pool. The local pool has one slot for each job at once, `Settings.slots`.
The engine of a job runs on the cores of its slot, `slot_cores`, and two jobs at once do not
share cores. An engine takes as many threads as it has cores. An rq worker is one slot. Its
container gives it its card, and it takes a free slot for each job, `claim_slot`, for its cores.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import threading
import time
import traceback
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

from ..catalog.model import Catalog
from ..settings import Settings
from .request import JobRequest
from .run import run_job
from .store import queued, write_json

log = logging.getLogger(__name__)


def server_cores() -> list[int]:
    """The cores this process may use, in order. Empty on a system that cannot pin a process."""
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []


def slot_cores(cores: list[int], slot: int, slots: int, per_job: int | None = None) -> list[int]:
    """The cores of the job that runs on one slot.

    Each slot has the next `per_job` cores of the server. With more cores asked for than the
    server has for every slot, the sets of the slots overlap. A `per_job` of all the cores gives
    every job the whole server. Without `per_job` the cores are divided between the slots, and
    the first slots get one core more where the division leaves a rest.

    Parameters
    ----------
    cores
        The cores of the server.
    slot
        The index of the slot, from 0.
    slots
        The number of slots of the queue.
    per_job
        The cores of one job. None divides the cores by the slots.
    """
    n = len(cores)
    if n == 0:
        return []
    base, rest = divmod(n, max(slots, 1))
    if per_job or base == 0:
        count = min(n, per_job) if per_job else 1
        first = (slot * count) % n
    else:
        count = base + (1 if slot < rest else 0)
        first = slot * base + min(slot, rest)
    return sorted(cores[(first + k) % n] for k in range(count))


@contextlib.contextmanager
def claim_slot(root: Path, total: int) -> Iterator[int | None]:
    """Takes a free slot of `total` for the time of one job, for workers that are processes of
    their own and share `root`. The slot is a lock on a file there. Yields its index, None
    when every slot is taken or the system has no file locks."""
    try:
        import fcntl  # noqa: PLC0415
    except ImportError:
        yield None
        return
    root.mkdir(parents=True, exist_ok=True)
    for k in range(total):
        handle = (root / f"{k}.lock").open("w")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            continue
        try:
            yield k
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
        return
    yield None


def work(
    req_dict: dict[str, object], job_dir: str, gpu: str | None = None, engine: list[str] | None = None, slot: tuple[int, int] | None = None
) -> str:
    """The function a worker runs, with everything it needs read from the environment. `gpu`
    is the only GPU the engine sees, all of the worker's GPUs when None. `slot` is the index of
    the slot of the local pool and the number of its slots, and gives the engine the cores of
    that slot. An rq worker has no slot of its own and takes a free one for the job, so the
    engines of the workers of one server share no core."""
    s = Settings.from_env()
    catalog = Catalog.load(s.data_dir)
    request = JobRequest(**req_dict)  # type: ignore[arg-type]
    if slot is None and s.queue == "rq":
        with claim_slot(s.results_dir / ".slots", len(s.slots)) as k:
            cores = slot_cores(server_cores(), k, len(s.slots), s.worker_cores) if k is not None else []
            run_job(request, catalog, Path(job_dir), engine=engine or s.engine_cmd, env_extra=s.engine_env(gpu) or None, cores=cores or None)
        return job_dir
    cores = slot_cores(server_cores(), slot[0], slot[1], s.worker_cores) if slot else []
    run_job(request, catalog, Path(job_dir), engine=engine or s.engine_cmd, env_extra=s.engine_env(gpu) or None, cores=cores or None)
    return job_dir


class LocalPool:
    """Runs jobs in the web process on a fixed set of slots, one card id per slot. A job waits
    for a free slot and its engine sees only the slot's GPU. Waiting jobs are lost when the
    process stops."""

    def __init__(self, slots: list[str], engine: list[str] | None = None) -> None:
        self._jobs: queue.Queue[tuple[dict[str, object], str]] = queue.Queue()
        self._engine = engine
        for i, gpu in enumerate(slots):
            threading.Thread(target=self._serve, args=(gpu, (i, len(slots))), daemon=True).start()

    def _serve(self, gpu: str, slot: tuple[int, int]) -> None:
        while True:
            req_dict, job_dir = self._jobs.get()
            try:
                work(req_dict, job_dir, gpu, self._engine, slot)
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
            _POOL = LocalPool(s.slots, s.engine_cmd)
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
    # The request and the first status are on disk from the moment the job waits. A restart
    # finds a waiting job by its status and queues it again.
    job_dir.mkdir(parents=True, exist_ok=True)
    write_json(job_dir / "request.json", asdict(req))
    queued(job_dir)
    if s.queue == "inline":
        # Synchronous. A request returns after its job; for development and tests.
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


def resume_interrupted(results_dir: Path, pool: LocalPool) -> list[str]:
    """Queues again the jobs that a stop of the web process left running or waiting, oldest
    first. Returns their names. A job without its request on disk is marked failed.
    """
    left = []
    for root in (results_dir / "jobs", results_dir / "private"):
        for st in root.glob("*/status.json"):
            try:
                status = json.loads(st.read_text())
            except (OSError, ValueError):
                continue
            if status.get("state") in ("running", "queued"):
                left.append((st.stat().st_mtime, st))
    again = []
    for _, st in sorted(left):
        job = st.parent
        try:
            req = json.loads((job / "request.json").read_text())
        except (OSError, ValueError):
            write_json(st, {"state": "failed", "error": "The site restarted before this job started. Submit it again.", "finished": time.time()})
            continue
        queued(job)
        pool.submit(req, str(job))
        again.append(job.name)
    return again


def resume_after_restart(s: Settings) -> list[str]:
    """`resume_interrupted` for the local pool. Other queues keep their jobs through a restart."""
    return resume_interrupted(s.results_dir, _local_pool(s)) if s.queue == "thread" else []
