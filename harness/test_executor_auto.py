"""Unit checks for the executor an `auto` stage resolves to.

    python harness/test_executor_auto.py

The smooth stage runs threaded on numba from four workers and on the JAX device below that.
A stage named outright keeps its name.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d import reconstruct  # noqa: E402
from gnome3d.pipeline.executor import ExecutorStrategy  # noqa: E402
from gnome3d.pipeline.stage import StageKind  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}   {detail}")


def strategy(kind: StageKind, value: str, workers: int, jax: bool) -> ExecutorStrategy:
    s = Settings()
    s.mc_executor_threaded_workers = workers
    s.mc_executor_smooth = s.mc_executor_arcs = s.mc_executor_estimate_dist = value
    reconstruct.jax_is_available = lambda: jax  # type: ignore[assignment]
    return reconstruct.stage_strategy(s, kind)


def main() -> int:
    T, B, S = ExecutorStrategy.THREADED, ExecutorStrategy.BATCH, ExecutorStrategy.SERIAL
    sm = StageKind.SMOOTH
    check("auto smooth is threaded at 4 workers with JAX", strategy(sm, "auto", 4, True) == T)
    check("auto smooth is threaded at 36 workers with JAX", strategy(sm, "auto", 36, True) == T)
    check("auto smooth is the device at 3 workers with JAX", strategy(sm, "auto", 3, True) == B)
    check("auto smooth is the device at 1 worker with JAX", strategy(sm, "auto", 1, True) == B)
    check("auto smooth is threaded at 2 workers without JAX", strategy(sm, "auto", 2, False) == T)
    check("auto smooth is serial at 1 worker without JAX", strategy(sm, "auto", 1, False) == S)
    check("a named executor is kept", strategy(sm, "batch", 36, True) == B)
    check("a named executor is kept at one worker", strategy(sm, "threaded", 1, True) == T)
    est = StageKind.ESTIMATE_DIST
    check("auto estimate stays on the device with JAX", strategy(est, "auto", 36, True) == B)
    check("auto arcs is threaded with workers", strategy(StageKind.ARCS, "auto", 36, True) == T)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
