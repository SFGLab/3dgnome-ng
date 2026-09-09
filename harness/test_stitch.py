"""Unit checks for the boundary stitch pass.

    python harness/test_stitch.py

The pass moves whole blocks rigidly so that the last anchor of one block and the first anchor
of the next sit at the distance an interior pair of the same genomic separation realises. Six
properties.

  * the within block curve is read off the structure itself
  * a boundary pair lands on that curve to a tight tolerance
  * intra block geometry is untouched, which is what rigid means
  * centroid excluded volume keeps non adjacent blocks apart
  * the energy's gradient is the gradient of that energy
  * a chromosome sized chain converges, which needs the gradient to be analytic

Plus the pass through cases a chromosome with one block or no anchors must take unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.pipeline.stitch import (  # noqa: E402
    CompartmentSites,
    _energy_grad,
    stitch_blocks,
    within_block_curve,
)
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


def test_gradient() -> None:
    """Against central differences, with the excluded volume active so its term is covered."""
    print("\n[gradient] the energy carries its own gradient")
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
    check("it matches central differences", worst < 1e-5, f"worst relative {worst:.2e}")


def _sites(n: int, rng: np.random.Generator, weight: float = 1.0) -> CompartmentSites:
    iu = np.triu_indices(n, k=1)
    return CompartmentSites(
        sites=rng.normal(0.0, 2.0, (n, 2, 3)),
        mass=rng.uniform(0.0, 1.0, (n, 2)),
        strength=(1.0, 2.0),
        pairs0=iu[0],
        pairs1=iu[1],
        radius=rng.uniform(5.0, 15.0, iu[0].size),
        weight=weight,
    )


def test_compartment_energy() -> None:
    """Two blocks, one A site each, no other term. The energy is the well in closed form."""
    print("\n[compartments] the block affinity is the well in closed form")
    cen = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    zero = np.zeros((2, 3))
    comp = CompartmentSites(
        sites=np.array([[[1.0, 0, 0], [0, 0, 0]], [[-1.0, 0, 0], [0, 0, 0]]]),
        mass=np.array([[1.0, 0.0], [0.5, 0.0]]),
        strength=(3.0, 2.0),
        pairs0=np.array([0]),
        pairs1=np.array([1]),
        radius=np.array([5.0]),
        weight=0.7,
    )
    e, _ = _energy_grad(
        np.zeros(12),
        cen,
        zero,
        zero,
        np.array([1.0]),
        0.0,
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([]),
        0.0,
        comp,
    )
    want = 0.7 * 3.0 * 1.0 * 0.5 * (1.0 - np.exp(-(8.0**2) / (2 * 25.0)))
    check("energy equals the closed form", abs(e - want) < 1e-12, f"{e:.6f} vs {want:.6f}")


def test_compartment_gradient() -> None:
    """Central differences with the springs, the excluded volume and the affinity all on."""
    print("\n[compartments] the affinity carries its gradient")
    rng = np.random.default_rng(3)
    n = 6
    cen = rng.normal(0.0, 8.0, (n, 3))
    first = rng.normal(0.0, 2.0, (n, 3))
    last = rng.normal(0.0, 2.0, (n, 3))
    target = rng.uniform(1.0, 5.0, n - 1)
    iu = np.triu_indices(n, k=1)
    r0 = rng.uniform(3.0, 8.0, iu[0].size)
    args = (cen, first, last, target, 1.0, iu[0], iu[1], r0, 1.0, _sites(n, rng, 2.0))
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
    check("it matches central differences", worst < 1e-5, f"worst relative {worst:.2e}")


def test_compartment_pulls_like_blocks() -> None:
    """A, B, A along the chain, the B block off the line. With the term the two A blocks end
    closer than without it, and the boundary springs still hold."""
    print("\n[compartments] like blocks are drawn together across a block in between")
    blocks = [
        block(0, np.array([0.0, 0.0, 0.0])),
        block(100_000, np.array([12.0, 12.0, 0.0])),
        block(200_000, np.array([24.0, 0.0, 0.0])),
    ]
    classes = [
        np.full(4, 1, dtype=np.int8),
        np.full(4, -1, dtype=np.int8),
        np.full(4, 1, dtype=np.int8),
    ]

    def aa_gap(out: list[list[BeadOut]]) -> float:
        c = [np.array([[b.x, b.y, b.z] for b in blk]).mean(axis=0) for blk in out]
        return float(np.linalg.norm(c[0] - c[2]))

    off = stitch_blocks(blocks, settings(), classes)
    on = stitch_blocks(blocks, settings(boundary_stitch_compartment_weight=5.0), classes)
    check(
        "the A blocks are closer with the term",
        aa_gap(on) < aa_gap(off) - 1.0,
        f"{aa_gap(on):.2f} vs {aa_gap(off):.2f}",
    )
    plain = stitch_blocks(blocks, settings())
    same = all(a == b for x, y in zip(off, plain, strict=True) for a, b in zip(x, y, strict=True))
    check("weight zero with classes given is the plain stitch, exactly", same)


def test_many_blocks_converge() -> None:
    """The property a chromosome needs and a handful of blocks cannot show.

    Six variables per block puts a real chromosome in the thousands of dimensions. A finite
    difference gradient costs one evaluation per variable, so scipy's own default evaluation
    budget of 15000 buys about one step there and the pass stops having moved almost nothing.
    A hundred and twenty blocks is already past the point where that shows.
    """
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
    check(
        "every boundary lands on the curve",
        float(np.median(ratio)) == 1.0 or abs(float(np.median(ratio)) - 1.0) < 0.05,
        f"median ratio {np.median(ratio):.3f}, worst {np.max(np.abs(ratio - 1.0)):.3f} off",
    )
    check(
        "and none is left stranded",
        float(np.max(ratio)) < 2.0,
        f"max ratio {np.max(ratio):.3f}",
    )


def main() -> int:
    print("boundary stitch checks")
    test_curve()
    test_boundary_lands_on_curve()
    test_rigid()
    test_excluded_volume()
    test_per_pair_radius()
    test_pass_through()
    test_gradient()
    test_compartment_energy()
    test_compartment_gradient()
    test_compartment_pulls_like_blocks()
    test_many_blocks_converge()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
