"""Unit checks for the executor that an `auto` stage resolves to."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")


from gnome3d import reconstruct
from gnome3d.pipeline.executor import ExecutorStrategy
from gnome3d.pipeline.stage import StageKind
from gnome3d.settings import Settings


def strategy(kind: StageKind, value: str, workers: int, jax: bool) -> ExecutorStrategy:
    s = Settings()
    s.mc_executor_threaded_workers = workers
    s.mc_executor_smooth = s.mc_executor_arcs = s.mc_executor_estimate_dist = value
    reconstruct.jax_is_available = lambda: jax  # type: ignore[assignment]
    return reconstruct.stage_strategy(s, kind)


@pytest.fixture(scope="module", autouse=True)
def shared() -> SimpleNamespace:
    T, B, S = ExecutorStrategy.THREADED, ExecutorStrategy.BATCH, ExecutorStrategy.SERIAL
    sm = StageKind.SMOOTH
    return SimpleNamespace(sm=sm, T=T, B=B, S=S)


def test_auto_smooth_is_threaded_at_4_workers_with_jax(shared: SimpleNamespace) -> None:
    sm, T, B, S = shared.sm, shared.T, shared.B, shared.S
    assert strategy(sm, "auto", 4, True) == T, "auto smooth is threaded at 4 workers with JAX"
    assert strategy(sm, "auto", 36, True) == T, "auto smooth is threaded at 36 workers with JAX"
    assert strategy(sm, "auto", 3, True) == B, "auto smooth is the device at 3 workers with JAX"
    assert strategy(sm, "auto", 1, True) == B, "auto smooth is the device at 1 worker with JAX"
    assert strategy(sm, "auto", 2, False) == T, "auto smooth is threaded at 2 workers without JAX"
    assert strategy(sm, "auto", 1, False) == S, "auto smooth is serial at 1 worker without JAX"
    assert strategy(sm, "batch", 36, True) == B, "a named executor is kept"
    assert strategy(sm, "threaded", 1, True) == T, "a named executor is kept at one worker"


def test_auto_estimate_stays_on_the_device_with_jax(shared: SimpleNamespace) -> None:
    B, T = shared.B, shared.T
    est = StageKind.ESTIMATE_DIST
    assert strategy(est, "auto", 36, True) == B, "auto estimate stays on the device with JAX"
    assert strategy(StageKind.ARCS, "auto", 36, True) == T, "auto arcs is threaded with workers"
