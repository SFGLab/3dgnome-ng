"""Unit checks for the anchor contact table and the two sided contact background."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from gnome3d import anchor_contacts as ac
from gnome3d.pipeline.coarse.build import add_contact_background
from gnome3d.polymer import PolymerLaw
from gnome3d.settings import Settings


def test_overlap() -> None:
    """A pixel is shared in proportion to the overlap."""
    bins = np.arange(0, 10_000, 1000, dtype=np.int64)
    w = ac.overlap_weights(bins, 1000, np.array([500, 5000]), np.array([2500, 6000])).toarray()
    assert np.allclose(w[:3, 0], [0.5, 1.0, 0.5]) and w[3:, 0].sum() == 0, (
        "an anchor over two and a half bins"
    )
    assert w[5, 1] == 1.0 and w[:, 1].sum() == 1.0, "an anchor on one whole bin"
    st, en = np.array([500, 5000]), np.array([1500, 6000])
    w = ac.overlap_weights(bins, 1000, st, en)
    b1 = np.array([0, 1, 5, 0])
    b2 = np.array([5, 5, 5, 1])
    cnt = np.array([4.0, 2.0, 9.0, 7.0])
    c = ac.aggregate_pixels(w, b1, b2, cnt).toarray()
    assert c[0, 1] == 0.5 * 4 + 0.5 * 2, f"pair count is the overlap weighted sum: {c[0, 1]}"
    assert c.sum() == c[0, 1], "a pixel inside one anchor and a diagonal pixel count for no pair"


def test_table() -> None:
    """A run reads the table by anchor interval."""
    t = ac.AnchorContacts(
        starts=np.array([0, 100, 100, 900], dtype=np.int64),
        ends=np.array([50, 150, 180, 950], dtype=np.int64),
        i=np.array([0, 1, 0], dtype=np.int32),
        j=np.array([2, 3, 3], dtype=np.int32),
        count=np.array([5.0, 7.0, 2.0], dtype=np.float32),
    )
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.npz"
        ac.save(p, {"chr1": t}, 1000, "test")
        back = ac.load(str(p), {"chr1", "chr2"})
    assert set(back) == {"chr1"} and np.array_equal(back["chr1"].count, t.count), "round trip"
    st = np.array([0, 100, 500, 900], dtype=np.int64)
    en = np.array([50, 180, 600, 950], dtype=np.int64)
    idx = ac.match(back["chr1"], st, en)
    assert list(idx) == [0, 2, -1, 3], (
        f"anchors sharing a start are told apart by the end: {list(idx)}"
    )
    m = ac.dense_counts(back["chr1"], st, en)
    want = np.zeros((4, 4))
    want[0, 1] = want[1, 0] = 5.0
    want[0, 3] = want[3, 0] = 2.0
    assert np.array_equal(m, want), "dense counts, symmetric, nothing for an anchor the table lacks"


def settings(z: float) -> Settings:
    s = Settings()
    s.polymer = PolymerLaw(nu=0.3, s0_bp=1000, q_half=1.0)
    s.use_contact_background = True
    s.background_weight = 0.1
    s.contact_background_z = z
    return s


def test_two_sided() -> None:
    """Enriched pairs held in, depleted held out, the rest untouched."""
    n = 60
    mids = [int(k * 50_000) for k in range(n)]
    lens = np.full(n, 1000.0)
    counts = np.full((n, n), 100.0)
    np.fill_diagonal(counts, 0.0)
    counts[10, 14] = counts[14, 10] = 400.0
    counts[20, 24] = counts[24, 20] = 0.0
    counts[30, 34] = counts[34, 30] = 400.0
    heat = counts / (np.outer(lens, lens) / 1e6)
    mat = np.full((n, n), -0.5)
    np.fill_diagonal(mat, 0.0)
    mat[30, 34] = mat[34, 30] = 2.0
    s = settings(3.0)
    out = add_contact_background(mat, mids, heat, s, lens)
    bg = s.polymer_law().background(200_000)
    assert -bg < out[10, 14] < -1.0, (
        f"an enriched arcless pair is held in: {out[10, 14]:.3f} against {-bg:.3f}"
    )
    assert out[20, 24] < -bg, (
        f"a depleted arcless pair is held out: {out[20, 24]:.3f} against {-bg:.3f}"
    )
    assert out[10, 14] == out[14, 10] and out[20, 24] == out[24, 20], "symmetric"
    assert out[30, 34] == 2.0, "an arc pair is never touched"
    assert out[5, 9] == -0.5, "a pair at its expectation keeps the repulsion"
    assert float(np.abs(np.diag(out)).max()) == 0.0, "the diagonal stays zero"
    hi = s.heatmap_distance_stretching * bg
    assert np.isfinite(out[20, 24]) and -out[20, 24] <= hi + 1e-9, (
        "a pair with no contact sits at a finite distance"
    )
    thin = add_contact_background(mat, mids, heat * 0.05, s, lens)
    assert thin[10, 14] == -0.5 and thin[20, 24] == -0.5, (
        f"a pair whose expectation is under z squared is never tested, even when enriched: expected 5 counts, {thin[10, 14]:.3f}"
    )
    old = add_contact_background(mat, mids, heat, settings(0.0), lens)
    assert old[20, 24] == -0.5 and old[10, 14] < -1.0, (
        "at z zero only enriched pairs are held, the old rule"
    )
    with pytest.raises(ValueError):
        add_contact_background(mat, mids, heat, s)


def test_settings() -> None:
    """The two keys are read."""
    s = Settings.from_dict(
        {"data": {"anchor_contacts": "x.npz"}, "springs": {"contact_background_z": "3"}}
    )
    assert s.data_anchor_contacts == "x.npz" and s.contact_background_z == 3.0, "read"
    d = Settings()
    assert d.data_anchor_contacts == "" and d.contact_background_z == 0.0, "defaults off"
