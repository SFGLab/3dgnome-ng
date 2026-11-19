"""The GPU memory pool of this process when other processes use the same card.

JAX preallocates a pool of `XLA_PYTHON_CLIENT_MEM_FRACTION` of the card, 0.75 by default, when
its backend starts. A process that starts on a card another process holds gets a smaller pool
than it asked for, while `memory_stats()` still reports the full limit. The batched kernels size
their launches from that limit and XLA aborts the process. `configure_gpu_pool` runs before the
backend starts and sets a pool that fits the free memory. On a free card it changes nothing.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass

from gnome3d import log

LOG = log.get("mc.jax.share")

GIB = 2**30
JAX_DEFAULT_FRACTION = 0.75
# Memory outside the pool for the CUDA context and library workspaces.
RESERVE_BYTES = GIB // 2
MIN_POOL_BYTES = GIB

_CONFIGURED = False


class GpuShareError(RuntimeError):
    """The card has too little free memory for the pool this process needs."""


@dataclass(frozen=True)
class PoolPlan:
    """The fraction to set before JAX starts, None to leave the environment unchanged and the
    status line to log, empty for none.
    """

    fraction: float | None
    line: str


def plan_gpu_pool(env: Mapping[str, str], gpus: Sequence[tuple[int, int]]) -> PoolPlan:
    """Decides the pool of this process from the environment and the visible GPUs.

    @param env the process environment
    @param gpus total and free bytes of each visible GPU
    @return the plan, with the fraction of the card to preallocate
    @raises GpuShareError when the requested or the possible pool does not fit the free memory
    """
    if env.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() in ("false", "0"):
        return PoolPlan(None, "")
    if env.get("XLA_PYTHON_CLIENT_ALLOCATOR", "").lower() in ("platform", "vmm", "cuda_async"):
        return PoolPlan(None, "")
    if not gpus:
        return PoolPlan(None, "")
    total, free = min(gpus, key=lambda g: g[1])
    asked = env.get("XLA_PYTHON_CLIENT_MEM_FRACTION")
    gib = env.get("GNOME3D_GPU_POOL_GIB")
    if gib is not None:
        if asked is not None:
            raise GpuShareError(
                f"GNOME3D_GPU_POOL_GIB={gib} and XLA_PYTHON_CLIENT_MEM_FRACTION={asked} are both set. "
                "Set one of them."
            )
        pool = float(gib) * GIB
        if pool + RESERVE_BYTES > free:
            raise GpuShareError(
                f"GNOME3D_GPU_POOL_GIB={gib} asks for a pool of {pool / GIB:.1f} GiB and the GPU has "
                f"{free / GIB:.1f} GiB free. Other processes hold the rest. Lower the pool so the "
                "processes on this card fit, or wait for one to finish."
            )
        return PoolPlan(
            int(pool / total * 10000) / 10000,
            f"GPU memory pool {pool / GIB:.1f} GiB of {total / GIB:.1f} GiB",
        )
    if asked is not None:
        pool = float(asked) * total
        if pool + RESERVE_BYTES > free:
            raise GpuShareError(
                f"XLA_PYTHON_CLIENT_MEM_FRACTION={asked} asks for a pool of {pool / GIB:.1f} GiB and the GPU has "
                f"{free / GIB:.1f} GiB free. Other processes hold the rest. Lower the fraction so the "
                "processes on this card add up to less than 1, or wait for one to finish."
            )
        return PoolPlan(None, f"GPU memory pool {pool / GIB:.1f} GiB of {total / GIB:.1f} GiB")
    if JAX_DEFAULT_FRACTION * total + RESERVE_BYTES <= free:
        return PoolPlan(None, "")
    pool = free - RESERVE_BYTES
    if pool < MIN_POOL_BYTES:
        raise GpuShareError(
            f"The GPU has {free / GIB:.1f} GiB free. Other processes hold the rest. Wait for one to finish, or "
            "set XLA_PYTHON_CLIENT_MEM_FRACTION in each process so the processes add up to less than 1."
        )
    fraction = int(pool / total * 1000) / 1000
    return PoolPlan(
        fraction,
        f"GPU has {free / GIB:.1f} of {total / GIB:.1f} GiB free, the pool of this process is "
        f"{fraction * total / GIB:.1f} GiB. Set XLA_PYTHON_CLIENT_MEM_FRACTION in each process to share "
        "the card evenly.",
    )


def visible_gpu_memory(env: Mapping[str, str]) -> list[tuple[int, int]]:
    """Reads total and free bytes of the GPUs in `CUDA_VISIBLE_DEVICES`, or of every GPU when it
    is unset, with nvidia-smi. Returns an empty list without nvidia-smi."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = [[f.strip() for f in line.split(",")] for line in out.splitlines() if line.strip()]
    wanted = [v.strip() for v in env.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    if wanted:
        picked = [r for r in rows if r[0] in wanted or r[1] in wanted]
        rows = picked or rows
    return [(int(r[2]) * 2**20, int(r[3]) * 2**20) for r in rows]


def configure_gpu_pool(env: MutableMapping[str, str] | None = None) -> None:
    """Sets `XLA_PYTHON_CLIENT_MEM_FRACTION` from the free memory of the card when the default
    pool does not fit. Runs once per process, before the JAX backend starts.

    @param env the environment to read and change, the process environment by default
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True
    env = os.environ if env is None else env
    if env.get("JAX_PLATFORMS", "").lower() == "cpu":
        return
    plan = plan_gpu_pool(env, visible_gpu_memory(env))
    if plan.fraction is not None:
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{plan.fraction:.4f}"
    if plan.line:
        log.status(LOG, "%s", plan.line)
