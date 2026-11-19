"""Unit checks for the walk start and the Hilbert start of the arcs stage."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from gnome3d.pipeline.ib.arcs import hilbert_start, walk_start
from gnome3d.settings import Settings
from gnome3d.util import seed_rng


@pytest.fixture(scope="module", autouse=True)
def shared() -> SimpleNamespace:
    s = Settings()
    law = s.polymer_law()
    mids = np.array([0, 50_000, 150_000, 1_000_000, 1_020_000], dtype=np.int64)
    pos0 = np.tile(np.array([1.0, 2.0, 3.0], dtype=np.float32), (5, 1))
    seed_rng(7)
    out = walk_start(pos0, mids, law)
    steps = np.linalg.norm(np.diff(out.astype(np.float64), axis=0), axis=1)
    want = np.array([law.background(int(b - a)) for a, b in zip(mids[:-1], mids[1:], strict=True)])
    return SimpleNamespace(steps=steps, want=want, out=out, pos0=pos0, mids=mids, law=law)


def test_consecutive_anchors_sit_at_the_law_s_distance_for_their_gap_vs(
    shared: SimpleNamespace,
) -> None:
    steps, want, out, pos0, mids, law = (
        shared.steps,
        shared.want,
        shared.out,
        shared.pos0,
        shared.mids,
        shared.law,
    )
    assert np.allclose(steps, want, rtol=1e-5), (
        f"consecutive anchors sit at the law's distance for their gap: {steps.round(3)} vs {want.round(3)}"
    )
    assert np.allclose(out.mean(axis=0), [1, 2, 3], atol=1e-4), (
        "the walk keeps the block's centroid: " + str(out.mean(axis=0).round(4))
    )
    assert out.dtype == np.float32 and out.shape == pos0.shape, (
        "it returns float32 of the input shape"
    )
    seed_rng(7)
    again = walk_start(pos0, mids, law)
    assert np.array_equal(out, again), "the same seed gives the same walk"
    seed_rng(8)
    other = walk_start(pos0, mids, law)
    assert not np.array_equal(out, other), "another seed gives another walk"
    assert Settings().arcs_start == "centroid", "the start is the centroid unless asked"


def test_consecutive_anchors_sit_at_the_law_s_bond_on_average_vs(shared: SimpleNamespace) -> None:
    law = shared.law
    rng = np.random.default_rng(1)
    gaps = rng.integers(5_000, 60_000, 400)
    mids = np.concatenate(([0], np.cumsum(gaps))).astype(np.int64)
    pos0 = np.tile(np.array([2.0, -1.0, 0.5], dtype=np.float32), (mids.size, 1))
    h = hilbert_start(pos0, mids, law)
    steps = np.linalg.norm(np.diff(h.astype(np.float64), axis=0), axis=1)
    want = np.mean([law.background(int(g)) for g in gaps])
    assert abs(steps.mean() / want - 1.0) < 0.05, (
        f"consecutive anchors sit at the law's bond on average: {steps.mean():.3f} vs {want:.3f}"
    )
    assert np.allclose(h.mean(axis=0), [2, -1, 0.5], atol=1e-3), "it keeps the block's centroid"
    assert np.array_equal(h, hilbert_start(pos0, mids, law)), "it is deterministic"
    i = rng.integers(0, mids.size, 20000)
    j = rng.integers(0, mids.size, 20000)
    keep = i != j
    sep = np.abs(mids[i[keep]] - mids[j[keep]]).astype(np.float64)
    dist = np.linalg.norm(h[i[keep]].astype(np.float64) - h[j[keep]], axis=1)
    # A space filling curve preserves locality and is not monotone. The mean distance rises with
    # separation up to the size of the cube.
    edges = np.array([0, 20e3, 100e3, 500e3, 2e6, 1e9])
    means = [
        dist[(sep >= a) & (sep < b)].mean() for a, b in zip(edges[:-1], edges[1:], strict=True)
    ]
    assert all(a < b for a, b in zip(means[:-1], means[1:], strict=True)), (
        "mean distance rises with separation: " + str(" < ".join(f"{m:.1f}" for m in means))
    )
    mids2 = np.concatenate(([0], np.cumsum(rng.integers(5_000, 60_000, 3200)))).astype(np.int64)
    h2 = hilbert_start(np.zeros((mids2.size, 3), dtype=np.float32), mids2, law)
    rg = lambda x: float(np.sqrt(np.mean(np.sum((x - x.mean(axis=0)) ** 2, axis=1))))  # noqa: E731
    ratio = rg(h2.astype(np.float64)) / rg(h.astype(np.float64))
    assert abs(ratio / 2.0 - 1.0) < 0.25, (
        f"size grows as the cube root of the count: 8x anchors gives {ratio:.2f}x the radius"
    )
