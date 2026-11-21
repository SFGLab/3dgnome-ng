"""The site's settings, from the environment.

DATA_DIR      the hosted data, with catalog.json at its root
RESULTS_DIR   jobs, shared ones under jobs/, private ones under private/
QUEUE         "rq" with REDIS_URL, "thread" for a pool of slots in the web process, or "inline"
              to run each job in the request, for tests
GPUS          the cards the jobs run on, their ids separated by commas, default "0"
WORKERS_PER_GPU  the jobs that run at once on one card, default 1. The site runs the number of
              cards times this number of jobs at once
WORKER_CORES  the cores the engine of one job runs on, default the cores of the server divided
              by the jobs at once. The "thread" queue gives each job a set of its own
GPU_POOL_GIB  the GPU memory pool of the engine of one job, in GiB. One worker on a card keeps the
              pool JAX takes. For more the default leaves each engine 0.5 GiB beside its pool and
              divides nine tenths of the rest of the card between the workers
ENGINE_CMD    the command that starts the engine, default ENGINE_PYTHON -m gnome3d.cli
RETENTION_DAYS  how long a private job is kept, default 14
MAX_REGION_MB   the longest region the form accepts, in Mb, default 20. limits.region_mb of the
                site's config.json stands over it
FRONTEND_DIST   the frontend build, default web/frontend/dist
BASE_PATH     the path the site is served under, /new for a site at /new/, default the root.
              The frontend is built with the same value
MAX_STRUCTURES  the most structures a job may ask for, default 100
UPLOAD_MAX_MB   the largest upload, in MB, default 200
LINK_READS    how many contact map links the server reads at once, default 4
LINK_ALLOW_PRIVATE  1 lets a contact map link point at a private or loopback address, for tests;
              off by default
SITE_CONFIG   the site's config.json, default DATA_DIR/config.json. It holds the feature flags
              and the limits of a job, see siteconfig.py. A site without the file shows
              everything and keeps the default limits
HICDIFFUSION_DIR, HICDIFFUSION_PYTHON, HICDIFFUSION_MODELS, REFERENCE_FASTA  the parts of the
              HiCDiffusion step, see hicdiffusion.py; it is offered when every part is in place
"""

from __future__ import annotations

import functools
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


# The share of a card that JAX preallocates for one process.
JAX_POOL_SHARE = 0.75
# The memory an engine needs on its card beside its pool, for the CUDA context and the
# workspaces of the libraries, in GiB. gnome3d.mc.jax.share keeps the same amount free.
ENGINE_RESERVE_GIB = 0.5
# The part of the card left after the reserves that the pools of its workers take.
POOL_SHARE = 0.9


@functools.cache
def card_gib(gpu: str | None) -> float | None:
    """The memory of a card in GiB, of the smallest card of this process for None, from
    nvidia-smi. None on a machine without it."""
    cmd = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", *(["-i", gpu] if gpu is not None else [])]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=True).stdout
        return min(float(line) for line in out.split()) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


# A job runs at once in its request, in a thread of the web process or in an rq worker.
QUEUES = ("inline", "thread", "rq")


class EnvError(ValueError):
    """A variable of the environment the site cannot use, with the fix as its message."""


def _whole(name: str, default: int | None) -> int | None:
    """A variable that is a whole number of 1 or more, `default` when it is unset or empty."""
    text = os.environ.get(name, "").strip()
    if not text:
        return default
    if not text.isdigit() or int(text) < 1:
        raise EnvError(f"{name} is {text!r}. Give a whole number, 1 or more.")
    return int(text)


def base_path(text: str) -> str:
    """A base path as a prefix with a leading slash and no trailing one. The root is empty."""
    inner = text.strip().strip("/")
    return f"/{inner}" if inner else ""


@dataclass
class Settings:
    data_dir: Path
    results_dir: Path
    queue: str
    redis_url: str
    engine_cmd: list[str] | None
    retention_days: int
    gpus: list[str]
    max_span: int
    frontend_dist: Path
    max_structures: int
    upload_max_mb: int
    link_allow_private: bool = False
    link_reads: int = 4
    base_path: str = ""
    site_config: Path | None = None
    workers_per_gpu: int = 1
    # None divides the cores of the server by the jobs at once.
    worker_cores: int | None = None
    # None divides the share of JAX by the workers of a card.
    gpu_pool_gib: float | None = None

    @property
    def jobs_dir(self) -> Path:
        """The jobs on hosted data, each under the key of its request."""
        return self.results_dir / "jobs"

    @property
    def private_dir(self) -> Path:
        """The jobs with an upload, each under a random token."""
        return self.results_dir / "private"

    @property
    def uploads_dir(self) -> Path:
        return self.results_dir / "uploads"

    @property
    def lists_dir(self) -> Path:
        return self.results_dir / "lists"

    @property
    def contrasts_dir(self) -> Path:
        """The finished comparisons of the expression of two jobs."""
        return self.results_dir / "expression"

    @property
    def locks_dir(self) -> Path:
        """The lock files by which the rq workers of one server share out the slots."""
        return self.results_dir / ".slots"

    @property
    def slots(self) -> list[str]:
        """The card of each job that runs at once. The first jobs go to one card each."""
        return [gpu for _ in range(self.workers_per_gpu) for gpu in self.gpus]

    def engine_env(self, gpu: str | None = None) -> dict[str, str]:
        """The variables of the engine of one job: its card, all cards of the process for None,
        and its GPU memory pool. A pool set in the environment of the process itself stays."""
        env = {"CUDA_VISIBLE_DEVICES": gpu} if gpu is not None else {}
        if "GNOME3D_GPU_POOL_GIB" in os.environ or "XLA_PYTHON_CLIENT_MEM_FRACTION" in os.environ:
            return env
        if self.gpu_pool_gib is not None:
            env["GNOME3D_GPU_POOL_GIB"] = f"{self.gpu_pool_gib:g}"
        elif self.workers_per_gpu > 1:
            n, card = self.workers_per_gpu, card_gib(gpu)
            pool = POOL_SHARE * (card - n * ENGINE_RESERVE_GIB) / n if card else 0.0
            if pool >= 1:
                env["GNOME3D_GPU_POOL_GIB"] = f"{pool:.2f}"
            else:
                # A card of unknown size, or one too small for a pool of 1 GiB each: the share of JAX is divided.
                env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{JAX_POOL_SHARE / n:.4f}"
        return env

    @classmethod
    def from_env(cls) -> Settings:
        cmd = os.environ.get("ENGINE_CMD", "")
        if os.environ.get("GPU_SLOTS"):
            raise EnvError(
                f"GPU_SLOTS={os.environ['GPU_SLOTS']} is not read. Set GPUS to the ids of the cards and WORKERS_PER_GPU to the "
                "jobs at once on one card. GPU_SLOTS=0,0,1,1 is GPUS=0,1 with WORKERS_PER_GPU=2."
            )
        pool = os.environ.get("GPU_POOL_GIB", "").strip()
        try:
            gpu_pool_gib = float(pool) if pool else None
        except ValueError:
            raise EnvError(f"GPU_POOL_GIB is {pool!r}. Give a number of GiB above zero.") from None
        if gpu_pool_gib is not None and gpu_pool_gib <= 0:
            raise EnvError(f"GPU_POOL_GIB is {pool!r}. Give a number of GiB above zero.")
        queue = os.environ.get("QUEUE", "rq")
        if queue not in QUEUES:
            raise EnvError(f"QUEUE is {queue!r}. Set it to one of {', '.join(QUEUES)}.")
        return cls(
            data_dir=Path(os.environ.get("DATA_DIR", "/data")),
            results_dir=Path(os.environ.get("RESULTS_DIR", "/results")),
            queue=queue,
            redis_url=os.environ.get("REDIS_URL", "redis://redis:6379/0"),
            engine_cmd=shlex.split(cmd) if cmd else None,
            retention_days=int(os.environ.get("RETENTION_DAYS", "14")),
            gpus=[g.strip() for g in os.environ.get("GPUS", "0").split(",") if g.strip()] or ["0"],
            workers_per_gpu=_whole("WORKERS_PER_GPU", 1) or 1,
            worker_cores=_whole("WORKER_CORES", None),
            gpu_pool_gib=gpu_pool_gib,
            max_span=int(float(os.environ.get("MAX_REGION_MB", "20")) * 1_000_000),
            max_structures=int(os.environ.get("MAX_STRUCTURES", "100")),
            upload_max_mb=int(os.environ.get("UPLOAD_MAX_MB", "200")),
            link_allow_private=os.environ.get("LINK_ALLOW_PRIVATE", "") == "1",
            link_reads=int(os.environ.get("LINK_READS", "4")),
            base_path=base_path(os.environ.get("BASE_PATH", "")),
            site_config=Path(os.environ["SITE_CONFIG"]) if os.environ.get("SITE_CONFIG") else None,
            frontend_dist=Path(os.environ.get("FRONTEND_DIST", Path(__file__).resolve().parents[1] / "frontend" / "dist")),
        )
