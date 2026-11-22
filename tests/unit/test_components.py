"""The loci of a uniform chain in its node solve and the component track of `[data] components`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gnome3d.pipeline.ib.components import (
    hold_loci,
    locus_nodes,
    node_affinity,
    read_components,
)
from gnome3d.pipeline.ib.nodes import contact_nodes, locus_beads
from gnome3d.polymer import PolymerLaw
from gnome3d.settings import Settings

LAW = PolymerLaw(nu=0.5, s0_bp=1000, q_half=1.0)

TRACK = """# sign chr1 0 +1 -1
chr1\t0\t100000\t0\t1.0 0.5
chr1\t100000\t200000\t0\t1.0 -0.5
chr1\t200000\t300000\t0\t-1.0 0.0
# sign chr1 1 +1 +1
chr1\t900000\t1000000\t1\t2.0 0.0
"""


@pytest.fixture
def track(tmp_path: Path) -> str:
    path = tmp_path / "components.bed"
    path.write_text(TRACK)
    return str(path)


def settings() -> Settings:
    s = Settings()
    s.heatmap_distance_stretching = 2.5
    return s


def test_track_is_read_by_arm(track: str) -> None:
    arms = read_components(track)["chr1"]
    assert [a.x.shape for a in arms] == [(3, 2), (1, 2)], (
        "three bins of one arm and one of the other"
    )
    assert arms[0].sign.tolist() == [1.0, -1.0] and arms[1].start.tolist() == [900_000]


def test_affinity_is_the_signed_sum_of_the_products(track: str) -> None:
    arms = read_components(track)["chr1"]
    mids = np.array([50_000, 150_000, 250_000, 450_000, 950_000], dtype=np.int64)
    affinity, arm_of, bin_of = node_affinity(arms, mids)
    assert bin_of.tolist() == [0, 1, 2, -1, 0], "the bin of each node in its arm"
    assert arm_of.tolist() == [0, 0, 0, -1, 1], "a node in a bin without a line has no arm"
    assert affinity[0, 1] == pytest.approx(1.0 * 1.0 - 0.5 * -0.5), (
        "1.25 with the second sign turned"
    )
    assert affinity[0, 2] == pytest.approx(-1.0)
    assert affinity[0, 4] == 0.0 and affinity[0, 3] == 0.0, (
        "two arms, and a node without components"
    )


def test_a_free_pair_is_held_at_the_background_times_the_affinity(track: str) -> None:
    arms = read_components(track)["chr1"]
    mids = np.array([50_000, 150_000, 250_000, 450_000, 950_000], dtype=np.int64)
    mat = np.full((5, 5), -0.5)
    np.fill_diagonal(mat, 0.0)
    mat[0, 1] = mat[1, 0] = 3.0  # a loop
    out = hold_loci(mat, mids, arms, LAW, settings())
    assert out[0, 1] == 3.0, "a pair with a loop keeps its target"
    bg = (200_000 / 1000) ** 0.5
    assert out[0, 2] == pytest.approx(-bg * np.exp(1.0 / 3.0)), (
        "an affinity of -1 holds the pair further"
    )
    assert out[1, 2] == pytest.approx(-((100_000 / 1000) ** 0.5) * np.exp(1.0 / 3.0))
    assert out[0, 3] == pytest.approx(-((400_000 / 1000) ** 0.5)), (
        "a locus that the track does not hold is at the law"
    )
    assert out[0, 4] == pytest.approx(-((900_000 / 1000) ** 0.5)), "two arms, at the law"
    assert np.array_equal(out, out.T) and (np.diag(out) == 0.0).all()


def test_a_pair_with_a_spring_keeps_it(track: str) -> None:
    arms = read_components(track)["chr1"]
    mids = np.array([50_000, 250_000], dtype=np.int64)
    mat = np.array([[0.0, -7.0], [-7.0, 0.0]])
    assert np.array_equal(hold_loci(mat, mids, arms, LAW, settings()), mat), (
        "a pair held by the background or by its cell is not changed"
    )


def test_reading_a_missing_track_stops(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="is not a file"):
        read_components(str(tmp_path / "none.bed"))


def test_one_node_stands_for_a_locus_and_the_affinity_is_in_the_energy_once(track: str) -> None:
    arms = read_components(track)["chr1"]
    # two nodes in the first locus, one in the second, two in the third
    mids = np.array([10_000, 55_000, 150_000, 240_000, 295_000], dtype=np.int64)
    assert locus_nodes(mids).tolist() == [1, 2, 3], "the node nearest to the middle of each locus"
    mat = np.full((5, 5), -0.5)
    np.fill_diagonal(mat, 0.0)
    out = hold_loci(mat, mids, arms, LAW, settings())
    held = np.argwhere(np.triu(out != mat, 1)).tolist()
    assert held == [[1, 2], [1, 3], [2, 3]], "one pair of nodes for each pair of loci"
    assert out[0, 4] == -0.5 and out[0, 1] == -0.5, "the other nodes keep what they had"


def test_without_a_track_a_pair_of_loci_holds_the_law() -> None:
    mids = np.array([50_000, 250_000, 950_000], dtype=np.int64)
    mat = np.full((3, 3), -0.5)
    np.fill_diagonal(mat, 0.0)
    out = hold_loci(mat, mids, [], LAW, settings())
    assert out[0, 1] == pytest.approx(-((200_000 / 1000) ** 0.5)), "the background at 200 kb"
    assert out[0, 2] == pytest.approx(-((900_000 / 1000) ** 0.5)), "the background at 900 kb"


def test_a_track_of_another_bin_stops(tmp_path: Path) -> None:
    path = tmp_path / "wide.bed"
    path.write_text("# sign chr1 0 +1\nchr1\t0\t250000\t0\t1.0\nchr1\t250000\t500000\t0\t1.0\n")
    mids = np.array([50_000, 300_000], dtype=np.int64)
    mat = np.array([[0.0, -0.5], [-0.5, 0.0]])
    with pytest.raises(ValueError, match="a locus is 100000 bp"):
        hold_loci(mat, mids, read_components(str(path))["chr1"], LAW, settings())


def test_locus_beads_are_the_beads_nearest_to_the_middle_of_each_locus() -> None:
    # one chain of 1 kb beads over 30 to 230 kb: the loci 0, 1 and 2
    edges = [np.arange(30_000, 231_000, 1000, dtype=np.int64)]
    beads, loci = locus_beads(edges, 100_000)
    assert loci.tolist() == [0, 1, 2]
    mid = (edges[0][beads] + edges[0][beads + 1]) // 2
    assert np.abs(mid[:2] - np.array([50_000, 150_000])).max() <= 500, (
        "the bead at the middle of a locus that the chain covers"
    )
    assert mid[2] == 229_500, "the bead nearest to the middle where the chain covers a part"


def test_a_locus_bead_is_a_node_and_joins_the_bead_of_a_loop_end() -> None:
    edges = [np.arange(0, 301_000, 1000, dtype=np.int64)]
    anchor_block = np.zeros(2, dtype=np.int32)
    ends = np.array([[0, 1]], dtype=np.int32)
    extra, _loci = locus_beads(edges, 100_000)
    beads = np.array([[int(extra[0]), 200]], dtype=np.int64)
    nodes = contact_nodes(edges, anchor_block, ends, beads, np.ones(1), extra)
    assert nodes.bead.tolist() == [int(extra[0]), int(extra[1]), 200, int(extra[2])], (
        "the loop end at the middle of the first locus is one node with its locus bead"
    )
    assert nodes.loop_node.tolist() == [[0, 2]], "the loop joins its two nodes"
    assert nodes.pair_i.tolist() == [0] and nodes.pair_j.tolist() == [2]
