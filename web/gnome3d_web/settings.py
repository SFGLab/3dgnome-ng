"""The site's settings, from the environment.

DATA_DIR      the hosted data, with catalog.json at its root
RESULTS_DIR   jobs, shared ones under jobs/, private ones under private/
QUEUE         "rq" with REDIS_URL, "thread" for a pool of slots in the web process, or "inline"
              to run each job in the request, for tests
GPU_SLOTS     the slots of the "thread" queue, one GPU id per slot, default "0"; "0,0,1,1" runs
              two jobs at once on each of two GPUs
ENGINE_CMD    the command that starts the engine, default ENGINE_PYTHON -m gnome3d.cli
RETENTION_DAYS  how long a private job is kept, default 14
MAX_REGION_MB   the longest region the form accepts, in Mb, default 20
FRONTEND_DIST   the frontend build, default web/frontend/dist
BASE_PATH     the path the site is served under, /new for a site at /new/, default the root.
              The frontend is built with the same value
MAX_STRUCTURES  the most structures a job may ask for, default 100
UPLOAD_MAX_MB   the largest upload, in MB, default 200
LINK_READS    how many contact map links the server reads at once, default 4
LINK_ALLOW_PRIVATE  1 lets a contact map link point at a private or loopback address, for tests;
              off by default
HICDIFFUSION_DIR, HICDIFFUSION_PYTHON, HICDIFFUSION_MODELS, REFERENCE_FASTA  the optional
              HiCDiffusion step; it is offered only when all four are set
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

HICDIFFUSION_KEYS = ("HICDIFFUSION_DIR", "HICDIFFUSION_PYTHON", "HICDIFFUSION_MODELS", "REFERENCE_FASTA")


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
    gpu_slots: list[str]
    max_span: int
    frontend_dist: Path
    max_structures: int
    upload_max_mb: int
    link_allow_private: bool = False
    link_reads: int = 4
    base_path: str = ""

    @classmethod
    def from_env(cls) -> Settings:
        cmd = os.environ.get("ENGINE_CMD", "")
        return cls(
            data_dir=Path(os.environ.get("DATA_DIR", "/data")),
            results_dir=Path(os.environ.get("RESULTS_DIR", "/results")),
            queue=os.environ.get("QUEUE", "rq"),
            redis_url=os.environ.get("REDIS_URL", "redis://redis:6379/0"),
            engine_cmd=shlex.split(cmd) if cmd else None,
            retention_days=int(os.environ.get("RETENTION_DAYS", "14")),
            gpu_slots=[g.strip() for g in os.environ.get("GPU_SLOTS", "0").split(",") if g.strip()],
            max_span=int(float(os.environ.get("MAX_REGION_MB", "20")) * 1_000_000),
            max_structures=int(os.environ.get("MAX_STRUCTURES", "100")),
            upload_max_mb=int(os.environ.get("UPLOAD_MAX_MB", "200")),
            link_allow_private=os.environ.get("LINK_ALLOW_PRIVATE", "") == "1",
            link_reads=int(os.environ.get("LINK_READS", "4")),
            base_path=base_path(os.environ.get("BASE_PATH", "")),
            frontend_dist=Path(os.environ.get("FRONTEND_DIST", Path(__file__).resolve().parents[1] / "frontend" / "dist")),
        )

    @staticmethod
    def hicdiffusion_missing() -> list[str]:
        """The HiCDiffusion settings that are not set, empty when the step can run."""
        return [k for k in HICDIFFUSION_KEYS if not os.environ.get(k)]
