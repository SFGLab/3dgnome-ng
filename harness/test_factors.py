"""Loops of more than one factor: two cluster files, a strength fit per factor, the factor
carried into the target matrix, and the orientation graph reading CTCF's loops alone.

    python harness/test_factors.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnome3d.data import fit_arc_strengths, mark_arcs  # noqa: E402
from gnome3d.io import load_arcs  # noqa: E402
from gnome3d.pipeline.coarse.build import arc_expected_matrix  # noqa: E402
from gnome3d.polymer import ArcStrengthFit, PolymerLaw  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402
from gnome3d.types import Anchor  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def write_bedpe(path: Path, rows: list[tuple[int, int, int]]) -> None:
    path.write_text(
        "".join(f"chr1\t{a}\t{a + 1000}\tchr1\t{b}\t{b + 1000}\t{s}\n" for a, b, s in rows)
    )


def test_cluster_files() -> None:
    s = Settings()
    s.data_dir = "/d"
    s.data_pet_clusters = "ctcf.bedpe"
    check("one file is CTCF, factor 0", s.cluster_files() == [("/d/ctcf.bedpe", 0, "CTCF")])
    s.data_pet_clusters = "ctcf.bedpe, pol2.bedpe"
    s.data_factors = "CTCF,RNAPOL2"
    check(
        "two files carry their names and indices",
        s.cluster_files() == [("/d/ctcf.bedpe", 0, "CTCF"), ("/d/pol2.bedpe", 1, "RNAPOL2")],
    )
    s.data_factors = "CTCF"
    try:
        s.cluster_files()
        check("a name count that does not match is refused", False)
    except ValueError:
        check("a name count that does not match is refused", True)


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
        check(
            "the loader tags each file's arcs",
            all(a.factor == 0 for a in raw0["chr1"]) and all(a.factor == 1 for a in raw1["chr1"]),
        )
        # anchors at every loop end, orientation only for the CTCF set
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
        check("marking keeps the factor", factors == {0, 1}, str(factors))
        fits = fit_arc_strengths(arcs)
        check("one strength fit per factor", set(fits) == {0, 1})
        check(
            "the fits differ, since the libraries do",
            abs(fits[0].median - fits[1].median) > 1.0,
            f"medians {fits[0].median:.1f} {fits[1].median:.1f}",
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
    check(
        "the same count reads as a stronger loop under the factor with the lower typical count",
        m0[0, 1] < m1[0, 1],
        f"{m0[0, 1]:.3f} vs {m1[0, 1]:.3f}",
    )
    m2 = arc_expected_matrix(s, mids, [(0, 1, 10)])
    check("a three field arc is factor 0", m2[0, 1] == m0[0, 1])


def main() -> int:
    test_cluster_files()
    test_factor_through_marking_and_fit()
    test_target_uses_the_factor_fit()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
