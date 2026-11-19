"""Unit checks for the hole beads between blocks."""

from __future__ import annotations

import numpy as np

from gnome3d.pipeline.holes import fill_holes, hole_slots
from gnome3d.settings import Settings
from gnome3d.types import BeadOut


def settings(on: bool, ev: bool = False) -> Settings:
    return Settings.from_dict(
        {
            "excluded_volume": {
                "use_excluded_volume": "yes" if ev else "no",
                "apply_to_smooth": "yes",
            },
            "simulation_arcs_smooth": {"hard_wall": "yes" if ev else "no"},
            "main": {
                "use_dynamic_loop_density": "yes",
                "target_bp_per_subanchor": "1000",
                "max_subanchors_per_arc": "100",
            },
            "distance": {"polymer_exponent": "0.34"},
            "holes": {"use_hole_beads": "yes" if on else "no"},
        }
    )


def block(lo: int, n: int, x0: float) -> list[BeadOut]:
    kinds = ["anchor"] + ["subanchor"] * (n - 2) + ["anchor"]
    return [
        BeadOut(lo + i * 1000, lo + (i + 1) * 1000, x0 + i, 0.0, 0.0, k)  # type: ignore[arg-type]
        for i, k in enumerate(kinds)
    ]


def test_slots() -> None:
    """A hole is tiled at the gap rule's count."""
    sl = hole_slots(10_000, 30_000, 1000)
    assert len(sl) == 19, f"count follows the rule: {len(sl)}"
    assert (
        sl[0][0] >= 10_000
        and sl[-1][1] <= 30_000
        and all(a[1] <= b[0] for a, b in zip(sl, sl[1:], strict=False))
    ), "slots inside the hole and ascending"
    assert len(hole_slots(0, 5_000_000, 1000)) == 4999, "no cap"
    assert hole_slots(10_000, 10_000, 1000) == [] and hole_slots(10_000, 11_000, 1000) == [], (
        "no hole no slots"
    )


def test_fill() -> None:
    """Blocks untouched, holes filled, seeded."""
    blocks = [block(0, 10, 0.0), block(60_000, 10, 30.0), block(20_000, 5, 12.0)]
    off = fill_holes(blocks, settings(False), "chr1", 0)
    assert off is blocks, "off returns the blocks"
    a = fill_holes(blocks, settings(True), "chr1", 0)
    b = fill_holes(blocks, settings(True), "chr1", 0)
    c = fill_holes(blocks, settings(True), "chr1", 7)
    assert a[:3] == blocks, "blocks first and unchanged"
    assert len(a) == 5, f"one list per hole: {len(a)}"
    holes = sorted(a[3:], key=lambda h: h[0].start)
    assert holes[0][0].start >= 10_000 and holes[0][-1].end <= 20_000, (
        "first hole between 10 kb and 20 kb"
    )
    assert holes[1][0].start >= 25_000 and holes[1][-1].end <= 60_000, (
        "second hole between 25 kb and 60 kb"
    )
    flat = sorted((x for blk in a for x in blk), key=lambda x: x.start)
    assert len({x.start for x in flat}) == len(flat), "bead starts unique"
    assert a == b, "same seed same beads"
    assert a[3:] != c[3:], "another seed other beads"
    xs = np.array([[x.x, x.y, x.z] for x in holes[1]])
    assert 14 < float(xs[:, 0].mean()) < 30, (
        f"bridge runs from one block to the next: {xs[:, 0].mean():.1f}"
    )


def test_long() -> None:
    """A hole over 500 beads is drawn in two levels."""
    blocks = [block(0, 10, 0.0), block(1_210_000, 10, 30.0)]
    a = fill_holes(blocks, settings(True), "chr1", 3)
    hole = a[2]
    xs = np.array([[x.x, x.y, x.z] for x in hole])
    bonds = np.linalg.norm(np.diff(xs, axis=0), axis=1)
    assert len(hole) == 1199, f"one bead per kilobase: {len(hole)}"
    assert bool(np.isfinite(xs).all()), "finite"
    assert float(bonds[[499, 500, 999, 1000]].max()) < 4 * float(np.median(bonds)), (
        f"bonds at the segment joins like the rest: median {np.median(bonds):.2f} joins {bonds[[499, 500, 999, 1000]].round(2)}"
    )


def test_blocks_repel() -> None:
    """A hole does not pass through a block."""
    a = block(0, 10, 0.0)
    b = block(60_000, 10, 13.0)
    g = np.stack(np.meshgrid(*[np.arange(-2.0, 2.01, 1.0)] * 3), -1).reshape(-1, 3)
    kinds = ["anchor"] + ["subanchor"] * (len(g) - 2) + ["anchor"]
    c = [
        BeadOut(200_000 + i * 1000, 201_000 + i * 1000, 11.0 + x, y, z, k)  # type: ignore[arg-type]
        for i, ((x, y, z), k) in enumerate(zip(g, kinds, strict=True))
    ]
    cp = np.array([[x.x, x.y, x.z] for x in c])
    for ev, want in ((False, "some"), (True, "none")):
        out = fill_holes([a, b, c], settings(True, ev=ev), "chr1", 0)
        hole = sorted(out[3:], key=lambda h: h[0].start)[0]
        hp = np.array([[x.x, x.y, x.z] for x in hole])
        d = np.linalg.norm(hp[:, None] - cp[None, :], axis=2)
        close = int((d < 0.5).sum())
        ok = close == 0 if ev else close > 0
        assert ok, (
            f"excluded volume {'on' if ev else 'off'}, {want} inside the block"
            + ": "
            + f"{close} pairs under 0.5"
        )
        assert out[2] == c, f"excluded volume {'on' if ev else 'off'}, block beads unmoved"


def test_keys() -> None:
    """Flank beads and the hole wall."""
    blocks = [block(0, 10, 0.0), block(60_000, 10, 30.0), block(20_000, 5, 12.0)]
    s = settings(True, ev=True)
    s.holes_flank_beads = 3
    a = fill_holes(blocks, s, "chr1", 0)
    b = fill_holes(blocks, s, "chr1", 0)
    assert a == b, "flank draw seeded"
    flat = sorted((x for blk in a for x in blk), key=lambda x: x.start)
    assert len({x.start for x in flat}) == len(flat), "flank draw keeps starts unique"
    assert a[3:] != fill_holes(blocks, settings(True, ev=True), "chr1", 0)[3:], (
        "flank draw changes the holes"
    )
    s2 = settings(True, ev=True)
    s2.holes_hard_wall = False
    c = fill_holes(blocks, s2, "chr1", 0)
    assert c[:3] == blocks, "blocks untouched without the wall"
