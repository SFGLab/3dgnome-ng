"""Unit checks for the seeds of the replicates of the estimate stage."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from gnome3d.pipeline.ib.estimate_dist import _expand_replicates
from gnome3d.settings import Settings


def problem(n: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    pos = np.cumsum(rng.normal(size=(n, 3)), axis=0).astype(np.float32)
    fixed = np.zeros(n, dtype=bool)
    fixed[0] = fixed[-1] = True
    s = Settings()
    s.subanchor_estimate_replicates = 2
    s.subanchor_estimate_steps = 2
    return {
        "pos": pos,
        "fixed": fixed,
        "dtn": np.linalg.norm(np.diff(pos, axis=0), axis=1).astype(np.float32),
        "step_size": 0.5,
        "settings": s,
        "seed": seed,
    }


def seeds_of(problems: list[dict[str, object]]) -> list[int]:
    expanded, _spans = _expand_replicates(problems, 4)
    return [int(e["seed"]) for e in expanded]


def test_every_replicate_carries_a_seed() -> None:
    got = seeds_of([problem(64, 7)])
    assert len(got) == 4 and all(isinstance(x, int) for x in got), "each replicate carries a seed"
    assert len(set(got)) == 4, f"sibling replicates get different seeds: {got}"


def test_seeds_follow_the_parent_not_the_batch() -> None:
    alone = seeds_of([problem(64, 7)])
    first = seeds_of([problem(64, 7), problem(64, 9)])[:4]
    second = seeds_of([problem(64, 9), problem(64, 7)])[4:]
    assert alone == first, f"a block's replicate seeds do not depend on its partners: {alone}"
    assert alone == second, f"nor on its position in the batch: {second}"


def test_different_blocks_get_different_seeds() -> None:
    a, b = seeds_of([problem(64, 7)]), seeds_of([problem(64, 9)])
    assert not set(a) & set(b), f"two blocks draw disjoint replicate seeds: {a} vs {b}"
