"""Unit checks for the GPU memory pool a process takes when it shares the card.

    python harness/test_gpu_share.py

JAX takes 75 percent of the card at start. A second process on the same card gets a smaller
pool than it asked for, still reports the full limit, sizes its launches from that limit and
aborts. `plan_gpu_pool` decides the pool before JAX starts. On a free card it changes nothing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.mc.jax.share import GpuShareError, plan_gpu_pool  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []
GIB = 2**30
CARD = 16 * GIB


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def raises(env: dict[str, str], gpus: list[tuple[int, int]]) -> str:
    try:
        plan_gpu_pool(env, gpus)
    except GpuShareError as e:
        return str(e)
    return ""


def main() -> int:
    p = plan_gpu_pool({}, [(CARD, int(15.4 * GIB))])
    check("a free card keeps the JAX default", p.fraction is None and p.line == "", str(p))

    p = plan_gpu_pool({}, [(CARD, int(3.9 * GIB))])
    want = (3.9 - 0.5) / 16
    check(
        "a card another process holds gives a pool from the free memory",
        p.fraction is not None
        and abs(p.fraction - want) < 0.002
        and "XLA_PYTHON_CLIENT_MEM_FRACTION" in p.line,
        str(p),
    )

    p = plan_gpu_pool({"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.45"}, [(CARD, int(15.4 * GIB))])
    check(
        "a fraction that fits is kept and reported",
        p.fraction is None and "7.2 GiB" in p.line,
        str(p),
    )

    msg = raises({"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.45"}, [(CARD, int(3.9 * GIB))])
    check("a fraction over the free memory is refused", "0.45" in msg and "3.9 GiB" in msg, msg)

    msg = raises({}, [(CARD, int(0.6 * GIB))])
    check("a nearly full card is refused", "0.6 GiB" in msg, msg)

    p = plan_gpu_pool({"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}, [(CARD, int(0.6 * GIB))])
    check("no preallocation is left alone", p.fraction is None, str(p))

    p = plan_gpu_pool({"XLA_PYTHON_CLIENT_ALLOCATOR": "platform"}, [(CARD, int(0.6 * GIB))])
    check("the platform allocator is left alone", p.fraction is None, str(p))

    p = plan_gpu_pool({}, [])
    check("no visible GPU changes nothing", p.fraction is None and p.line == "", str(p))

    p = plan_gpu_pool({}, [(CARD, int(15.4 * GIB)), (CARD, int(5.0 * GIB))])
    check(
        "several GPUs take the pool from the fullest one",
        p.fraction is not None and abs(p.fraction - (5.0 - 0.5) / 16) < 0.002,
        str(p),
    )

    p = plan_gpu_pool({"GNOME3D_GPU_POOL_GIB": "1.5"}, [(CARD, int(15.4 * GIB))])
    check(
        "a pool in GiB becomes the fraction of the card",
        p.fraction is not None and abs(p.fraction - 1.5 / 16) < 0.001 and "1.5 GiB" in p.line,
        str(p),
    )
    p = plan_gpu_pool({"GNOME3D_GPU_POOL_GIB": "1.5"}, [(8 * GIB, int(7.6 * GIB))])
    check(
        "the same pool on an 8 GiB card is twice the fraction",
        p.fraction is not None and abs(p.fraction - 1.5 / 8) < 0.001,
        str(p),
    )

    msg = raises({"GNOME3D_GPU_POOL_GIB": "1.5"}, [(CARD, int(1.2 * GIB))])
    check(
        "a pool in GiB over the free memory is refused", "1.5 GiB" in msg and "1.2 GiB" in msg, msg
    )

    msg = raises(
        {"GNOME3D_GPU_POOL_GIB": "1.5", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.2"},
        [(CARD, int(15.4 * GIB))],
    )
    check(
        "a pool in GiB and a fraction together are refused",
        "GNOME3D_GPU_POOL_GIB" in msg and "XLA_PYTHON_CLIENT_MEM_FRACTION" in msg,
        msg,
    )

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
