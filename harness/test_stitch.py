"""Unit checks for the boundary stitch pass.

    python harness/test_stitch.py

The pass moves whole blocks rigidly so that the last anchor of one block and the first anchor
of the next sit at the distance an interior pair of the same genomic separation realises. Four
properties.

  * the within block curve is read off the structure itself
  * a boundary pair lands on that curve to a tight tolerance
  * intra block geometry is untouched, which is what rigid means
  * centroid excluded volume keeps non adjacent blocks apart

Plus the pass through cases a chromosome with one block or no anchors must take unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.pipeline.stitch import stitch_blocks, within_block_curve  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402
from gnome3d.types import BeadOut  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}{('  ' + detail) if detail else ''}")


D = 10.0
TET = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float64) * (
    D / np.sqrt(8.0)
)


def block(
    start_bp: int, offset: np.ndarray, kinds: tuple[str, ...] = ("anchor",) * 4
) -> list[BeadOut]:
    """Four beads on a tetrahedron of edge D, 1 kb apart genomically, shifted by offset."""
    out: list[BeadOut] = []
    for k, (p, kind) in enumerate(zip(TET + offset, kinds, strict=True)):
        s = start_bp + k * 1000
        out.append(BeadOut(s, s + 500, float(p[0]), float(p[1]), float(p[2]), kind))  # type: ignore[arg-type]
    return out


def settings(**kw: object) -> Settings:
    s = Settings()
    s.use_boundary_stitch = True
    s.exclusion_radius_ib = 0.5  # never binds in the tetrahedron tests unless set otherwise
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def edge_distance(a: list[BeadOut], b: list[BeadOut]) -> float:
    la = [x for x in a if x.kind == "anchor"][-1]
    fb = [x for x in b if x.kind == "anchor"][0]
    return float(np.linalg.norm(np.array([la.x, la.y, la.z]) - np.array([fb.x, fb.y, fb.z])))


def pairwise(bl: list[BeadOut]) -> np.ndarray:
    p = np.array([[x.x, x.y, x.z] for x in bl])
    return np.linalg.norm(p[:, None] - p[None], axis=-1)


def centroid(bl: list[BeadOut]) -> np.ndarray:
    return np.array([[x.x, x.y, x.z] for x in bl]).mean(axis=0)


def test_curve() -> None:
    print("\n[curve] the within block curve is read off the structure")
    blocks = [block(0, np.zeros(3)), block(100_000, np.array([500.0, 0, 0]))]
    curve = within_block_curve(blocks)
    check(
        "flat structure gives a flat curve",
        abs(curve(2_000) - D) < 1e-6 and abs(curve(3_000) - D) < 1e-6,
    )
    check("gaps beyond the sampled range clamp", abs(curve(50_000_000) - D) < 1e-6)


def test_boundary_lands_on_curve() -> None:
    print("\n[boundary] adjacent edges are pulled onto the interior curve")
    blocks = [block(0, np.zeros(3)), block(100_000, np.array([500.0, 0, 0]))]
    before = edge_distance(blocks[0], blocks[1])
    out = stitch_blocks(blocks, settings())
    after = edge_distance(out[0], out[1])
    check(
        "edge distance moves from far to the curve",
        before > 400 and abs(after - D) < 1e-3,
        f"{before:.1f} -> {after:.4f}, target {D}",
    )


def test_rigid() -> None:
    print("\n[rigid] intra block geometry is untouched")
    rng = np.random.default_rng(3)
    blocks = [block(i * 100_000, rng.normal(0, 300, 3)) for i in range(5)]
    out = stitch_blocks(blocks, settings())
    worst = max(
        float(np.abs(pairwise(a) - pairwise(b)).max()) for a, b in zip(blocks, out, strict=True)
    )
    check("pairwise distances inside every block unchanged", worst < 1e-6, f"max drift {worst:.1e}")
    check(
        "bead count, ranges and kinds preserved",
        all(
            (a.start, a.end, a.kind) == (b.start, b.end, b.kind)
            for x, y in zip(blocks, out, strict=True)
            for a, b in zip(x, y, strict=True)
        ),
    )


def test_excluded_volume() -> None:
    print("\n[ev] centroid excluded volume keeps non adjacent blocks apart")
    # a one anchor middle block: both springs want the outer blocks' edges at D from one point,
    # so without excluded volume the outer blocks can fold onto each other
    mid = [BeadOut(100_000, 100_500, 0.0, 0.0, 0.0, "anchor")]
    blocks = [block(0, np.array([-600.0, 0, 0])), mid, block(200_000, np.array([600.0, 0, 0]))]
    free = stitch_blocks(blocks, settings(boundary_stitch_ev_weight=0.0))
    held = stitch_blocks(blocks, settings(exclusion_radius_ib=8 * D, boundary_stitch_ev_weight=1.0))
    d_free = float(np.linalg.norm(centroid(free[0]) - centroid(free[2])))
    d_held = float(np.linalg.norm(centroid(held[0]) - centroid(held[2])))
    check(
        "outer blocks sit further apart with excluded volume on",
        d_held > d_free + D,
        f"{d_free:.2f} -> {d_held:.2f}",
    )
    check(
        "excluded volume did not detach the boundaries",
        edge_distance(held[0], held[1]) < 4 * D and edge_distance(held[1], held[2]) < 4 * D,
        f"{edge_distance(held[0], held[1]):.1f}, {edge_distance(held[1], held[2]):.1f}",
    )


def test_per_pair_radius() -> None:
    print("\n[radius] the excluded volume radius comes from the blocks' own size")
    # outer tetrahedra have Rg D*sqrt(3/8) = 6.12 each, so two of them touch at 12.2 apart. The
    # old radius came from gld of the centroid gap, 0.5 * gld(100 kb) = 8.4, below touching.
    mid = [BeadOut(100_000, 100_500, 0.0, 0.0, 0.0, "anchor")]
    blocks = [block(0, np.array([-600.0, 0, 0])), mid, block(200_000, np.array([600.0, 0, 0]))]
    held = stitch_blocks(blocks, settings(exclusion_radius_ib=0.0, boundary_stitch_ev_weight=1.0))
    d = float(np.linalg.norm(centroid(held[0]) - centroid(held[2])))
    rg = float(np.sqrt(np.mean(np.sum((TET - TET.mean(axis=0)) ** 2, axis=1))))
    check(
        "outer blocks held at or beyond touching",
        d > 0.9 * 2 * rg and d > 8.4,
        f"{d:.2f}, touching {2 * rg:.2f}, old radius 8.4",
    )
    fixed = stitch_blocks(
        blocks, settings(exclusion_radius_ib=4 * D, boundary_stitch_ev_weight=1.0)
    )
    d_fixed = float(np.linalg.norm(centroid(fixed[0]) - centroid(fixed[2])))
    check(
        "explicit exclusion_radius_ib still overrides", d_fixed > d + D, f"{d_fixed:.2f} vs {d:.2f}"
    )


def test_pass_through() -> None:
    print("\n[pass through] nothing to stitch")
    one = [block(0, np.array([7.0, 8.0, 9.0]))]
    check("single block returned unchanged", stitch_blocks(one, settings()) == one)
    sub = [
        block(0, np.zeros(3), ("subanchor",) * 4),
        block(100_000, np.array([500.0, 0, 0]), ("subanchor",) * 4),
    ]
    check("blocks without anchors returned unchanged", stitch_blocks(sub, settings()) == sub)


def main() -> int:
    print("boundary stitch checks")
    test_curve()
    test_boundary_lands_on_curve()
    test_rigid()
    test_excluded_volume()
    test_per_pair_radius()
    test_pass_through()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
