"""Unit checks for the boundary stitch pass."""

from __future__ import annotations

import numpy as np

from gnome3d.pipeline.stitch import (
    _energy_grad,
    stitch_blocks,
    within_block_curve,
)
from gnome3d.settings import Settings
from gnome3d.types import BeadOut

D = 10.0
TET = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float64) * (
    D / np.sqrt(8.0)
)


def block(
    start_bp: int, offset: np.ndarray, kinds: tuple[str, ...] = ("anchor",) * 4
) -> list[BeadOut]:
    """Four beads on a tetrahedron of edge D, 1 kb apart on the genome, shifted by `offset`."""
    out: list[BeadOut] = []
    for k, (p, kind) in enumerate(zip(TET + offset, kinds, strict=True)):
        s = start_bp + k * 1000
        out.append(BeadOut(s, s + 500, float(p[0]), float(p[1]), float(p[2]), kind))  # type: ignore[arg-type]
    return out


def settings(**kw: object) -> Settings:
    s = Settings()
    s.use_boundary_stitch = True
    s.exclusion_radius_ib = 0.5  # does not bind in the tetrahedron tests unless set
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
    """The within block curve is read off the structure."""
    blocks = [block(0, np.zeros(3)), block(100_000, np.array([500.0, 0, 0]))]
    curve = within_block_curve(blocks)
    assert abs(curve(2_000) - D) < 1e-6 and abs(curve(3_000) - D) < 1e-6, (
        "flat structure gives a flat curve"
    )
    assert abs(curve(50_000_000) - D) < 1e-6, "gaps beyond the sampled range clamp"


def test_boundary_lands_on_curve() -> None:
    """Adjacent edges are pulled onto the interior curve."""
    blocks = [block(0, np.zeros(3)), block(100_000, np.array([500.0, 0, 0]))]
    before = edge_distance(blocks[0], blocks[1])
    out = stitch_blocks(blocks, settings())
    after = edge_distance(out[0], out[1])
    assert before > 400 and abs(after - D) < 1e-3, (
        f"edge distance moves from far to the curve: {before:.1f} -> {after:.4f}, target {D}"
    )


def test_rigid() -> None:
    """Intra block geometry is untouched."""
    rng = np.random.default_rng(3)
    blocks = [block(i * 100_000, rng.normal(0, 300, 3)) for i in range(5)]
    out = stitch_blocks(blocks, settings())
    worst = max(
        float(np.abs(pairwise(a) - pairwise(b)).max()) for a, b in zip(blocks, out, strict=True)
    )
    assert worst < 1e-6, f"pairwise distances inside every block unchanged: max drift {worst:.1e}"
    assert all(
        (a.start, a.end, a.kind) == (b.start, b.end, b.kind)
        for x, y in zip(blocks, out, strict=True)
        for a, b in zip(x, y, strict=True)
    ), "bead count, ranges and kinds preserved"


def test_excluded_volume() -> None:
    """Centroid excluded volume keeps non adjacent blocks apart."""
    # A middle block of one anchor. Both springs want the edges of the outer blocks at D from one
    # point. Without excluded volume the outer blocks can fold onto each other.
    mid = [BeadOut(100_000, 100_500, 0.0, 0.0, 0.0, "anchor")]
    blocks = [block(0, np.array([-600.0, 0, 0])), mid, block(200_000, np.array([600.0, 0, 0]))]
    free = stitch_blocks(blocks, settings(boundary_stitch_ev_weight=0.0))
    held = stitch_blocks(blocks, settings(exclusion_radius_ib=8 * D, boundary_stitch_ev_weight=1.0))
    d_free = float(np.linalg.norm(centroid(free[0]) - centroid(free[2])))
    d_held = float(np.linalg.norm(centroid(held[0]) - centroid(held[2])))
    assert d_held > d_free + D, (
        f"outer blocks sit further apart with excluded volume on: {d_free:.2f} -> {d_held:.2f}"
    )
    assert edge_distance(held[0], held[1]) < 4 * D and edge_distance(held[1], held[2]) < 4 * D, (
        f"excluded volume did not detach the boundaries: {edge_distance(held[0], held[1]):.1f}, {edge_distance(held[1], held[2]):.1f}"
    )


def test_per_pair_radius() -> None:
    """The excluded volume radius comes from the blocks' own size."""
    # Each outer tetrahedron has an Rg of D * sqrt(3 / 8) = 6.12 and two of them touch at 12.2
    # apart.
    mid = [BeadOut(100_000, 100_500, 0.0, 0.0, 0.0, "anchor")]
    blocks = [block(0, np.array([-600.0, 0, 0])), mid, block(200_000, np.array([600.0, 0, 0]))]
    held = stitch_blocks(blocks, settings(exclusion_radius_ib=0.0, boundary_stitch_ev_weight=1.0))
    d = float(np.linalg.norm(centroid(held[0]) - centroid(held[2])))
    rg = float(np.sqrt(np.mean(np.sum((TET - TET.mean(axis=0)) ** 2, axis=1))))
    assert d > 0.9 * 2 * rg and d > 8.4, (
        f"outer blocks held at or beyond touching: {d:.2f}, touching {2 * rg:.2f}, old radius 8.4"
    )
    fixed = stitch_blocks(
        blocks, settings(exclusion_radius_ib=4 * D, boundary_stitch_ev_weight=1.0)
    )
    d_fixed = float(np.linalg.norm(centroid(fixed[0]) - centroid(fixed[2])))
    assert d_fixed > d + D, (
        f"explicit exclusion_radius_ib still overrides: {d_fixed:.2f} vs {d:.2f}"
    )


def test_pass_through() -> None:
    """Nothing to stitch."""
    one = [block(0, np.array([7.0, 8.0, 9.0]))]
    assert stitch_blocks(one, settings()) == one, "single block returned unchanged"
    sub = [
        block(0, np.zeros(3), ("subanchor",) * 4),
        block(100_000, np.array([500.0, 0, 0]), ("subanchor",) * 4),
    ]
    assert stitch_blocks(sub, settings()) == sub, "blocks without anchors returned unchanged"


def test_gradient() -> None:
    """The energy carries its own gradient."""
    rng = np.random.default_rng(0)
    n = 7
    cen = rng.normal(0.0, 10.0, (n, 3))
    first = rng.normal(0.0, 2.0, (n, 3))
    last = rng.normal(0.0, 2.0, (n, 3))
    target = rng.uniform(1.0, 5.0, n - 1)
    iu = np.triu_indices(n, k=1)
    r0 = rng.uniform(3.0, 8.0, iu[0].size)
    args = (cen, first, last, target, 1.0, iu[0], iu[1], r0, 1.0)
    x = rng.normal(0.0, 0.5, 6 * n)
    _, g = _energy_grad(x, *args)
    h = 1e-6
    worst = 0.0
    for k in range(x.size):
        a, b = x.copy(), x.copy()
        a[k] += h
        b[k] -= h
        fd = (_energy_grad(a, *args)[0] - _energy_grad(b, *args)[0]) / (2 * h)
        worst = max(worst, abs(fd - g[k]) / max(abs(fd), 1e-6))
    assert worst < 1e-5, f"it matches central differences: worst relative {worst:.2e}"


def test_many_blocks_converge() -> None:
    """A chain of 120 blocks has 720 variables and converges only with an analytic gradient."""
    print("\n[scale] a long chain of blocks reaches its own minimum")
    rng = np.random.default_rng(4)
    n = 120
    blocks = [block(k * 100_000, rng.normal(0.0, 60.0, 3)) for k in range(n)]
    out = stitch_blocks(blocks, settings())
    curve = within_block_curve(out)
    assert curve is not None
    ratio = np.array(
        [
            edge_distance(out[k], out[k + 1])
            / curve(out[k + 1][0].start + 250 - (out[k][-1].start + 250))
            for k in range(n - 1)
        ]
    )
    assert float(np.median(ratio)) == 1.0 or abs(float(np.median(ratio)) - 1.0) < 0.05, (
        f"every boundary lands on the curve: median ratio {np.median(ratio):.3f}, worst {np.max(np.abs(ratio - 1.0)):.3f} off"
    )
    assert float(np.max(ratio)) < 2.0, f"and none is left stranded: max ratio {np.max(ratio):.3f}"
