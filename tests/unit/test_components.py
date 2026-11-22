"""The components per locus of `[data] components` in the node solve of a uniform chain."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gnome3d.pipeline.ib.components import hold_by_components, node_affinity, read_components
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
    affinity, arm_of = node_affinity(arms, mids)
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
    out = hold_by_components(mat, mids, arms, LAW, settings())
    assert out[0, 1] == 3.0, "a pair with a loop keeps its target"
    bg = (200_000 / 1000) ** 0.5
    assert out[0, 2] == pytest.approx(-bg * np.exp(1.0 / 3.0)), (
        "an affinity of -1 holds the pair further"
    )
    assert out[1, 2] == pytest.approx(-((100_000 / 1000) ** 0.5) * np.exp(1.0 / 3.0))
    assert out[0, 3] == -0.5 and out[0, 4] == -0.5, "no components or another arm, left as it is"
    assert np.array_equal(out, out.T) and (np.diag(out) == 0.0).all()


def test_a_pair_with_a_spring_keeps_it(track: str) -> None:
    arms = read_components(track)["chr1"]
    mids = np.array([50_000, 250_000], dtype=np.int64)
    mat = np.array([[0.0, -7.0], [-7.0, 0.0]])
    assert np.array_equal(hold_by_components(mat, mids, arms, LAW, settings()), mat), (
        "a pair held by the background or by its cell is not changed"
    )


def test_reading_a_missing_track_stops(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="is not a file"):
        read_components(str(tmp_path / "none.bed"))
