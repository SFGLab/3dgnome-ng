"""Reconstruction of a known structure from its own contacts, through the whole pipeline.

    .venv/bin/pytest tests/integration/test_synthetic.py

The inputs are written by the test. No file under `data/` is read and no GPU is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from gnome3d.data import ContactData
from gnome3d.io import parse_region, write_cif
from gnome3d.settings import Settings
from gnome3d.simulate import simulate
from gnome3d.types import BeadOut, BedRegion, F64Array, I64Array
from tests.integration import support

NODES = 30


@dataclass
class Run:
    truth: F64Array
    edges: I64Array
    ensemble: list[list[BeadOut]]
    settings: Settings
    data: ContactData
    bed: BedRegion


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    """Reconstructs two structures of one known walk of 30 beads."""
    root = tmp_path_factory.mktemp("synthetic")
    truth = support.synthetic_structure(NODES, seed=1)
    region, edges = support.write_inputs(truth, root)
    s = support.cpu_settings(support.CELL, root)
    bed = parse_region(region)
    assert bed is not None
    data = ContactData.from_files(s, [bed.chr], bed)
    ensemble = [m[bed.chr] for m in simulate(s, data, [bed.chr], 2, region=bed)]
    return Run(truth, edges, ensemble, s, data, bed)


def test_every_bin_is_modelled(run: Run) -> None:
    for beads in run.ensemble:
        assert not np.isnan(support.bin_centroids(beads, run.edges)).any()


def test_beads_are_a_sorted_finite_chain(run: Run) -> None:
    beads = run.ensemble[0]
    starts = [b.start for b in beads]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)
    assert all(b.end >= b.start for b in beads)
    assert {b.kind for b in beads} <= {"anchor", "subanchor"}
    assert sum(b.kind == "anchor" for b in beads) == 2 * NODES
    coords, _ = support.arrays(beads)
    assert np.isfinite(coords).all()


def test_reconstruction_is_closer_to_the_truth_than_a_random_structure(run: Run) -> None:
    """The bounds hold on every seed tried. The production settings place far bins by the
    distance law and recover the walk only in part, a rank correlation of 0.4 to 0.6."""
    rng = np.random.default_rng(999)
    random = support.scale_to(run.truth, rng.normal(size=run.truth.shape))
    for beads in run.ensemble:
        x = support.scale_to(run.truth, support.bin_centroids(beads, run.edges))
        assert support.rmsd(run.truth, x) < support.rmsd(run.truth, random)
        assert support.spearman(pdist(run.truth), pdist(x)) > 0.2


def test_a_run_repeats_and_members_differ(run: Run) -> None:
    again = simulate(run.settings, run.data, [run.bed.chr], 2, region=run.bed)
    first, second = (support.arrays(m[run.bed.chr])[0] for m in again)
    a, b = (support.arrays(m)[0] for m in run.ensemble)
    assert np.array_equal(first, a)
    assert np.array_equal(second, b)
    assert a.shape != b.shape or not np.allclose(a, b)


def test_cif_holds_every_bead(run: Run, tmp_path: Path) -> None:
    path = tmp_path / "model.cif"
    write_cif(str(path), run.ensemble[0], entry_id="synthetic")
    rows = [ln for ln in path.read_text().splitlines() if ln.startswith(("ATOM", "HETATM"))]
    assert len(rows) == len(run.ensemble[0])
