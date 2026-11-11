"""The site's settings, from the environment.

DATA_DIR      the hosted data, with catalog.json at its root
RESULTS_DIR   jobs, shared ones under jobs/, private ones under private/
QUEUE         "rq" with REDIS_URL, or "inline" to run each job in a thread of the web process
ENGINE_CMD    the command that starts the engine, default ENGINE_PYTHON -m gnome3d.cli
RETENTION_DAYS  how long a private job is kept, default 14
HICDIFFUSION_DIR, HICDIFFUSION_PYTHON, HICDIFFUSION_MODELS, REFERENCE_FASTA  the optional
              HiCDiffusion step; it is offered only when all four are set
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

HICDIFFUSION_KEYS = ("HICDIFFUSION_DIR", "HICDIFFUSION_PYTHON", "HICDIFFUSION_MODELS", "REFERENCE_FASTA")


@dataclass
class Settings:
    data_dir: Path
    results_dir: Path
    queue: str
    redis_url: str
    engine_cmd: list[str] | None
    retention_days: int

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
        )

    @staticmethod
    def hicdiffusion_missing() -> list[str]:
        """The HiCDiffusion settings that are not set, empty when the step can run."""
        return [k for k in HICDIFFUSION_KEYS if not os.environ.get(k)]
