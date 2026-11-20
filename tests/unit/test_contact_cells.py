"""Unit checks for the Hi-C cells of the pairs of nodes of a uniform chain."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from gnome3d.pipeline.ib.cells import cell_uri, expected_by_offset, hold_by_cells, read_cells
from gnome3d.polymer import PolymerLaw
from gnome3d.settings import Settings

LAW = PolymerLaw(nu=0.5, s0_bp=1000, q_half=1.0)


def settings(z: float = 2.0) -> Settings:
    s = Settings()
    s.contact_background_z = z
    s.heatmap_distance_stretching = 2.5
    return s


def banded(n: int, per_offset: dict[int, float]) -> sparse.csr_matrix:
    """A symmetric matrix with one count on every cell of the given offsets."""
    m = np.zeros((n, n))
    for d, c in per_offset.items():
        i = np.arange(n - d)
        m[i, i + d] = c
        m[i + d, i] = c
    return sparse.csr_matrix(m)


def test_cell_uri_addresses_the_resolution_of_an_mcool() -> None:
    assert cell_uri("a/b.mcool::/resolutions/1000", 5000) == "a/b.mcool::/resolutions/5000"
    assert cell_uri("a/b.mcool", 5000) == "a/b.mcool::/resolutions/5000"
    assert cell_uri("a/b.cool", 5000) == "a/b.cool", "a file of one resolution is not addressed"


def test_expectation_is_the_mean_count_at_each_offset() -> None:
    counts = banded(12, {1: 8.0, 2: 3.0})
    exp = expected_by_offset(counts, np.ones(12, dtype=np.bool_))
    assert exp[0] == 0.0 and exp[1] == pytest.approx(8.0) and exp[2] == pytest.approx(3.0)
    assert exp[5] == 0.0, "no count at that offset"


def test_expectation_leaves_out_the_cells_of_an_empty_bin() -> None:
    dense = banded(10, {1: 6.0}).toarray()
    dense[4, :] = 0.0
    dense[:, 4] = 0.0
    valid = dense.sum(axis=1) > 0
    exp = expected_by_offset(sparse.csr_matrix(dense), valid)
    assert exp[1] == pytest.approx(6.0), "the two cells of the empty bin are not counted as zero"


def test_expectation_pools_offsets_in_eight_bins_a_decade() -> None:
    """Offsets 10 to 13 are in one bin of log offset."""
    counts = banded(40, {10: 4.0})
    exp = expected_by_offset(counts, np.ones(40, dtype=np.bool_))
    assert exp[10] == exp[13] == pytest.approx(4.0 * 30 / (30 + 29 + 28 + 27))
    assert exp[14] == 0.0


def cells_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Three nodes 16 kb and 64 kb from the first. Expectation 25 at every offset."""
    mids = np.array([500, 16_500, 64_500], dtype=np.int64)
    offset = np.array([[0, 16, 64], [16, 0, 48], [64, 48, 0]], dtype=np.int64)
    expected = np.full(100, 25.0)
    mapped = np.ones(3, dtype=np.bool_)
    mat = np.array([[0.0, -4.0, -0.5], [-4.0, 0.0, 3.0], [-0.5, 3.0, 0.0]])
    return mids, offset, expected, mapped, mat


def test_a_cell_over_its_expectation_holds_the_pair_closer() -> None:
    mids, offset, expected, mapped, mat = cells_case()
    count = np.array([[0.0, 100.0, 25.0], [100.0, 0.0, 400.0], [25.0, 400.0, 0.0]])
    out = hold_by_cells(mat, mids, count, offset, expected, mapped, LAW, settings())
    want = 4.0 * (100.5 / 25.5) ** (-1.0 / 3.0)
    assert out[0, 1] == pytest.approx(-want) and out[1, 0] == out[0, 1], "15 deviations over"
    assert out[0, 2] == -0.5, "a cell at its expectation holds nothing"
    assert out[1, 2] == 3.0, "a pair with a loop keeps its loop target"


def test_a_cell_under_its_expectation_holds_the_pair_apart() -> None:
    mids, offset, expected, mapped, mat = cells_case()
    count = np.array([[0.0, 25.0, 4.0], [25.0, 0.0, 25.0], [4.0, 25.0, 0.0]])
    out = hold_by_cells(mat, mids, count, offset, expected, mapped, LAW, settings())
    bg = 8.0
    assert out[0, 2] == pytest.approx(-bg * (4.5 / 25.5) ** (-1.0 / 3.0)), "4.2 deviations under"
    assert out[0, 1] == -4.0, "the near pair is at its expectation and keeps the background"


def test_the_target_is_clipped_to_one_bead_and_to_the_stretch() -> None:
    mids, offset, expected, mapped, mat = cells_case()
    count = np.array([[0.0, 1e6, 0.0], [1e6, 0.0, 25.0], [0.0, 25.0, 0.0]])
    out = hold_by_cells(mat, mids, count, offset, expected, mapped, LAW, settings())
    assert out[0, 1] == -1.0, "no closer than one bead"
    assert out[0, 2] == pytest.approx(-2.5 * 8.0), "no farther than the stretch"


def test_weak_cells_and_unmapped_nodes_hold_nothing() -> None:
    mids, offset, expected, mapped, mat = cells_case()
    count = np.array([[0.0, 100.0, 100.0], [100.0, 0.0, 25.0], [100.0, 25.0, 0.0]])
    low = hold_by_cells(mat, mids, count, offset, np.full(100, 3.0), mapped, LAW, settings())
    assert np.array_equal(low, mat), "an expectation under z squared is not tested"
    unmapped = np.array([True, False, True])
    out = hold_by_cells(mat, mids, count, offset, expected, unmapped, LAW, settings())
    assert out[0, 1] == -4.0 and out[0, 2] != -0.5, "a node in a bin without a count is left"
    same = offset.copy()
    same[0, 1] = same[1, 0] = 0
    out = hold_by_cells(mat, mids, count, same, expected, mapped, LAW, settings())
    assert out[0, 1] == -4.0, "two nodes in one cell are left"


def test_read_cells_from_an_mcool(tmp_path: Path) -> None:
    cooler = pytest.importorskip("cooler")
    import pandas as pd

    res, n = 5000, 40
    bins = pd.DataFrame(
        {"chrom": "chr1", "start": np.arange(n) * res, "end": (np.arange(n) + 1) * res}
    )
    i, j = np.triu_indices(n, 1)
    count = np.where(j - i == 1, 9, np.where(j - i == 2, 4, 0))
    count[(i == 3) & (j == 20)] = 30
    keep = count > 0
    pix = pd.DataFrame({"bin1_id": i[keep], "bin2_id": j[keep], "count": count[keep]})
    path = tmp_path / "tiny.mcool"
    cooler.create_cooler(f"{path}::/resolutions/{res}", bins, pix, ordered=True)

    mids = np.array([17_500, 22_500, 102_500], dtype=np.int64)  # bins 3, 4 and 20
    got, offset, expected, mapped = read_cells(f"{path}::/resolutions/1000", "chr1", mids, res)
    assert got.tolist() == [[0.0, 9.0, 30.0], [9.0, 0.0, 0.0], [30.0, 0.0, 0.0]]
    assert offset.tolist() == [[0, 1, 17], [1, 0, 16], [17, 16, 0]]
    assert expected[1] == pytest.approx(9.0) and expected[2] == pytest.approx(4.0)
    assert mapped.all()
    with pytest.raises(ValueError, match="has no map at"):
        read_cells(str(path), "chr1", mids, 1000)


def test_a_pair_is_read_at_the_finest_resolution_that_can_test_it() -> None:
    """At the fine resolution only the near pair has an expectation above z squared. Its cell is at
    its expectation. The coarse resolution could hold it and may not.
    """
    mids, offset, _expected, mapped, mat = cells_case()
    tested = np.zeros((3, 3), dtype=np.bool_)
    fine_expected = np.where(np.arange(100) <= 16, 25.0, 1.0)
    fine_count = np.array([[0.0, 25.0, 9.0], [25.0, 0.0, 9.0], [9.0, 9.0, 0.0]])
    out = hold_by_cells(
        mat, mids, fine_count, offset, fine_expected, mapped, LAW, settings(), tested
    )
    assert np.array_equal(out, mat), "nothing is held at the fine resolution"
    assert tested[0, 1] and not tested[0, 2], "the near pair is tested, the far pair is not"

    coarse_offset = np.array([[0, 2, 6], [2, 0, 5], [6, 5, 0]], dtype=np.int64)
    coarse_count = np.array([[0.0, 400.0, 400.0], [400.0, 0.0, 400.0], [400.0, 400.0, 0.0]])
    out = hold_by_cells(
        out, mids, coarse_count, coarse_offset, np.full(100, 25.0), mapped, LAW, settings(), tested
    )
    assert out[0, 1] == -4.0, "the pair tested at the fine resolution keeps its background"
    assert out[0, 2] != -0.5 and tested[0, 2], "the far pair is held by its coarse cell"
