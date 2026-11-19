"""Unit checks for loops of more than one factor: two cluster files, a strength fit per factor,
the factor in the target matrix and an orientation graph of the CTCF loops alone.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from gnome3d.data import fit_arc_strengths, mark_arcs
from gnome3d.io import load_arcs
from gnome3d.pipeline.coarse.build import arc_expected_matrix
from gnome3d.polymer import ArcStrengthFit, PolymerLaw
from gnome3d.settings import Settings
from gnome3d.types import Anchor


def write_bedpe(path: Path, rows: list[tuple[int, int, int]]) -> None:
    path.write_text(
        "".join(f"chr1\t{a}\t{a + 1000}\tchr1\t{b}\t{b + 1000}\t{s}\n" for a, b, s in rows)
    )


def test_cluster_files() -> None:
    s = Settings()
    s.data_dir = "/d"
    s.data_pet_clusters = "ctcf.bedpe"
    assert s.cluster_files() == [("/d/ctcf.bedpe", 0, "CTCF")], "one file is CTCF, factor 0"
    s.data_pet_clusters = "ctcf.bedpe, pol2.bedpe"
    s.data_factors = "CTCF,RNAPOL2"
    assert s.cluster_files() == [("/d/ctcf.bedpe", 0, "CTCF"), ("/d/pol2.bedpe", 1, "RNAPOL2")], (
        "two files carry their names and indices"
    )
    s.data_factors = "CTCF"
    with pytest.raises(ValueError):
        s.cluster_files()


def test_factor_through_marking_and_fit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        rng = np.random.default_rng(0)
        ctcf = [
            (int(a), int(a + sp), int(c))
            for a, sp, c in zip(
                rng.integers(1e6, 5e7, 300),
                rng.integers(2e4, 8e5, 300),
                rng.integers(3, 40, 300),
                strict=True,
            )
        ]
        pol2 = [
            (int(a), int(a + sp), int(c))
            for a, sp, c in zip(
                rng.integers(1e6, 5e7, 300),
                rng.integers(1e4, 3e5, 300),
                rng.integers(3, 12, 300),
                strict=True,
            )
        ]
        write_bedpe(d / "ctcf.bedpe", ctcf)
        write_bedpe(d / "pol2.bedpe", pol2)
        raw0, _ = load_arcs(str(d / "ctcf.bedpe"), {"chr1"}, None, 10**9, factor=0)
        raw1, _ = load_arcs(str(d / "pol2.bedpe"), {"chr1"}, None, 10**9, factor=1)
        assert all(a.factor == 0 for a in raw0["chr1"]) and all(
            a.factor == 1 for a in raw1["chr1"]
        ), "the loader tags each file's arcs"
        # Anchors at every loop end, with an orientation for the CTCF set only.
        ends = sorted({p for a in raw0["chr1"] + raw1["chr1"] for p in (a.start, a.end)})
        anchors = {
            "chr1": [
                Anchor(
                    "chr1",
                    p - 500,
                    p + 500,
                    "L" if any(x.start == p or x.end == p for x in raw0["chr1"]) else "N",
                )
                for p in ends
            ]
        }
        merged = {"chr1": sorted(raw0["chr1"] + raw1["chr1"], key=lambda a: a.start)}
        arcs = mark_arcs(anchors, merged)
        factors = {a.factor for a in arcs["chr1"]}
        assert factors == {0, 1}, "marking keeps the factor: " + str(factors)
        fits = fit_arc_strengths(arcs)
        assert set(fits) == {0, 1}, "one strength fit per factor"
        assert abs(fits[0].median - fits[1].median) > 1.0, (
            f"the fits differ, since the libraries do: medians {fits[0].median:.1f} {fits[1].median:.1f}"
        )


def test_target_uses_the_factor_fit() -> None:
    s = Settings()
    s.background_weight = 0.0
    strong = ArcStrengthFit(log_a=0.0, slope=0.0, median=2.0, ok=False, reason="test")
    weak = ArcStrengthFit(log_a=0.0, slope=0.0, median=20.0, ok=False, reason="test")
    s.polymer = PolymerLaw(nu=0.3, s0_bp=1000, arcs=strong, arcs_by_factor={0: strong, 1: weak})
    mids = [0, 500_000, 1_000_000]
    m0 = arc_expected_matrix(s, mids, [(0, 1, 10, 0)])
    m1 = arc_expected_matrix(s, mids, [(0, 1, 10, 1)])
    assert m0[0, 1] < m1[0, 1], (
        f"the same count reads as a stronger loop under the factor with the lower typical count: {m0[0, 1]:.3f} vs {m1[0, 1]:.3f}"
    )
    m2 = arc_expected_matrix(s, mids, [(0, 1, 10)])
    assert m2[0, 1] == m0[0, 1], "a three field arc is factor 0"


def test_factor_strength() -> None:
    s = Settings()
    s.background_weight = 0.0
    fit = ArcStrengthFit(log_a=0.0, slope=0.0, median=2.0, ok=False, reason="test")
    s.polymer = PolymerLaw(
        nu=0.3, s0_bp=1000, arcs=fit, arcs_by_factor={0: fit, 1: fit}, strength_by_factor={1: 0.3}
    )
    mids = [0, 500_000]
    full = arc_expected_matrix(s, mids, [(0, 1, 10, 0)])[0, 1]
    weak = arc_expected_matrix(s, mids, [(0, 1, 10, 1)])[0, 1]
    assert weak > full, (
        f"a factor at strength 0.3 holds its loop farther than CTCF holds the same count: {full:.3f} vs {weak:.3f}"
    )
    s.polymer = PolymerLaw(
        nu=0.3, s0_bp=1000, arcs=fit, arcs_by_factor={0: fit, 1: fit}, strength_by_factor={1: 0.0}
    )
    none = arc_expected_matrix(s, mids, [(0, 1, 10, 1)])[0, 1]
    assert abs(none - s.polymer.background(500_000)) < 1e-9, (
        f"at strength 0 the loop sits at the background: {none:.3f}"
    )
    s2 = Settings()
    s2.factor_strength = "1, 0.3"
    assert s2.factor_strengths() == {0: 1.0, 1: 0.3}, "the setting parses per file"
