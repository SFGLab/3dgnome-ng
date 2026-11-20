"""The whole pipeline with `[main] uniform_chain` on the synthetic inputs.

    .venv/bin/pytest tests/integration/test_uniform_pipeline.py

Each bin of the synthetic region holds one loop of two anchors of 2 kb, 2 kb apart. Each loop is
a block of 6 kb.
"""

from __future__ import annotations

import copy
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from gnome3d.data import ContactData
from gnome3d.io import parse_region
from gnome3d.simulate import simulate
from gnome3d.types import BeadOut
from tests.integration import support

NODES = 12


@pytest.fixture(scope="module")
def beads(tmp_path_factory: pytest.TempPathFactory) -> list[BeadOut]:
    """The beads of one structure, with the smooth stage on numba."""
    root = tmp_path_factory.mktemp("uniform")
    truth = support.synthetic_structure(NODES, seed=1)
    region, _edges = support.write_inputs(truth, root)
    s = copy.deepcopy(support.cpu_settings(support.CELL, root))
    s.uniform_chain = True
    s.mc_executor_smooth = "serial"
    bed = parse_region(region)
    assert bed is not None
    data = ContactData.from_files(s, [bed.chr], bed)
    return simulate(s, data, [bed.chr], 1, region=bed)[0][bed.chr]


def test_every_block_is_six_beads_of_one_kilobase(beads: list[BeadOut]) -> None:
    assert len(beads) == 6 * NODES
    assert all(b.end - b.start == 1000 for b in beads)


def test_no_base_is_in_two_beads(beads: list[BeadOut]) -> None:
    starts = np.array([b.start for b in beads])
    ends = np.array([b.end for b in beads])
    assert (np.diff(starts) > 0).all(), "the output is in chain order"
    assert (starts[1:] >= ends[:-1]).all()


def test_beads_under_an_anchor_have_the_kind_anchor(beads: list[BeadOut]) -> None:
    kinds = [b.kind for b in beads[:6]]
    assert kinds == ["anchor", "anchor", "subanchor", "subanchor", "anchor", "anchor"]


def test_coordinates_are_finite_and_bonds_are_short(beads: list[BeadOut]) -> None:
    xyz, _mid = support.arrays(beads)
    assert np.isfinite(xyz).all()
    block = xyz.reshape(NODES, 6, 3)
    bonds = np.linalg.norm(np.diff(block, axis=1), axis=2)
    assert np.median(bonds) < 2.0, f"median bond {np.median(bonds):.2f} beads"
