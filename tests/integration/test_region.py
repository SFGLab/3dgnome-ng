"""A real region of GM12878 through the whole pipeline on the production settings.

    .venv/bin/pytest tests/integration/test_region.py

Reads the ChIA-PET inputs under `data/GM12878`, which are not in the repository. The test is
skipped without them. The region is chr1:18288319-20307135, 102 anchors and 114 loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from gnome3d.data import ContactData
from gnome3d.io import load_arcs, parse_region
from gnome3d.simulate import simulate
from gnome3d.types import BeadOut, F64Array
from tests.integration import support

CELL = "GM12878"
REGION = "chr1:18288319-20307135"
DATA = Path(__file__).resolve().parents[2] / "data"
INPUTS = (
    f"{CELL}_anchors_3+_oriented.bed",
    f"{CELL}_clusters_3+.bedpe",
    f"ccds_all_hg38_merged100k_{CELL}.breakpoints.bed",
    "hg38_centromeres.bed",
)

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not all((DATA / CELL / name).is_file() for name in INPUTS),
        reason=f"the inputs of {CELL} are not under data/",
    ),
]


@dataclass
class Loops:
    """The 3D distance of every loop and of every anchor pair that no loop joins."""

    score: F64Array
    distance: F64Array
    separation: F64Array
    control_distance: F64Array
    control_separation: F64Array


@dataclass
class Run:
    ensemble: list[list[BeadOut]]
    again: list[list[BeadOut]]
    loops: list[Loops]


def loop_distances(beads: list[BeadOut], arcs: list[tuple[int, int, float]]) -> Loops:
    """Measures each loop between the anchors that hold its two ends."""
    coords, mids = support.arrays(beads)
    anchors = [i for i, b in enumerate(beads) if b.kind == "anchor"]
    starts = np.array([beads[i].start for i in anchors])

    def anchor_of(pos: int) -> int:
        return anchors[max(int(np.searchsorted(starts, pos, side="right")) - 1, 0)]

    joined: dict[tuple[int, int], float] = {}
    for a, b, score in arcs:
        i, j = sorted((anchor_of(a), anchor_of(b)))
        if i != j:
            joined[(i, j)] = max(score, joined.get((i, j), 0.0))
    pairs = np.array(list(joined))
    free = np.array(
        [(i, j) for n, i in enumerate(anchors) for j in anchors[n + 1 :] if (i, j) not in joined]
    )

    def dist(p: np.ndarray) -> F64Array:
        return np.linalg.norm(coords[p[:, 0]] - coords[p[:, 1]], axis=1)

    def sep(p: np.ndarray) -> F64Array:
        return np.abs(mids[p[:, 0]] - mids[p[:, 1]]).astype(np.float64)

    return Loops(np.array(list(joined.values())), dist(pairs), sep(pairs), dist(free), sep(free))


@pytest.fixture(scope="module")
def run() -> Run:
    """Reconstructs two structures twice from the same seeds."""
    s = support.cpu_settings(CELL, DATA, quality="balanced")
    bed = parse_region(REGION)
    assert bed is not None
    data = ContactData.from_files(s, [bed.chr], bed)
    ensemble = [m[bed.chr] for m in simulate(s, data, [bed.chr], 2, region=bed)]
    again = [m[bed.chr] for m in simulate(s, data, [bed.chr], 2, region=bed)]
    loaded, _long = load_arcs(s.data_path(s.data_pet_clusters), {bed.chr}, bed)
    arcs = [(a.start, a.end, float(a.score)) for a in loaded[bed.chr]]
    return Run(ensemble, again, [loop_distances(beads, arcs) for beads in ensemble])


def test_beads_are_a_sorted_finite_chain(run: Run) -> None:
    for beads in run.ensemble:
        starts = [b.start for b in beads]
        assert starts == sorted(starts)
        assert len(set(starts)) == len(starts)
        assert sum(b.kind == "anchor" for b in beads) == 102
        assert sum(b.kind == "subanchor" for b in beads) > 1000
        assert np.isfinite(support.arrays(beads)[0]).all()


def test_a_run_repeats_and_members_differ(run: Run) -> None:
    for first, second in zip(run.ensemble, run.again, strict=True):
        assert np.array_equal(support.arrays(first)[0], support.arrays(second)[0])
    a, b = (support.arrays(m)[0] for m in run.ensemble)
    assert not np.allclose(a, b)


def test_a_stronger_loop_is_shorter(run: Run) -> None:
    """Self-consistency of the 2016 paper: the PET count against the 3D distance, about -0.6."""
    for loops in run.loops:
        assert support.spearman(loops.score, loops.distance) < -0.3


def test_loops_are_closer_than_other_anchor_pairs_as_far_apart(run: Run) -> None:
    """Loops are at a median of 3.7 beads and other pairs of the same separations at 5.3."""
    for loops in run.loops:
        lo, hi = np.percentile(loops.separation, [10, 90])
        held = loops.distance[(loops.separation >= lo) & (loops.separation <= hi)]
        free = loops.control_distance[
            (loops.control_separation >= lo) & (loops.control_separation <= hi)
        ]
        assert np.median(held) < 0.85 * np.median(free)


def test_few_beads_overlap(run: Run) -> None:
    """Pairs under 0.7 of a bead that are not chain neighbours, 1 to 4 per thousand beads."""
    for beads in run.ensemble:
        coords, _ = support.arrays(beads)
        close = sum(1 for i, j in cKDTree(coords).query_pairs(0.7) if j - i > 1)
        assert 1000 * close / len(beads) < 20
