"""Unit checks for the hole beads between blocks.

    python harness/test_holes.py

A hole's slots tile the stretch between two blocks at the gap rule's count, the blocks are left
as they were, the draw follows the seed, and with the key off nothing changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.pipeline.holes import fill_holes, hole_slots  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402
from gnome3d.types import BeadOut  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


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
    print("\n[slots] a hole is tiled at the gap rule's count")
    sl = hole_slots(10_000, 30_000, 1000)
    check("count follows the rule", len(sl) == 19, f"{len(sl)}")
    check(
        "slots inside the hole and ascending",
        sl[0][0] >= 10_000
        and sl[-1][1] <= 30_000
        and all(a[1] <= b[0] for a, b in zip(sl, sl[1:], strict=False)),
    )
    check("no cap", len(hole_slots(0, 5_000_000, 1000)) == 4999)
    check(
        "no hole no slots",
        hole_slots(10_000, 10_000, 1000) == [] and hole_slots(10_000, 11_000, 1000) == [],
    )


def test_fill() -> None:
    print("\n[fill] blocks untouched, holes filled, seeded")
    blocks = [block(0, 10, 0.0), block(60_000, 10, 30.0), block(20_000, 5, 12.0)]
    off = fill_holes(blocks, settings(False), "chr1", 0)
    check("off returns the blocks", off is blocks)
    a = fill_holes(blocks, settings(True), "chr1", 0)
    b = fill_holes(blocks, settings(True), "chr1", 0)
    c = fill_holes(blocks, settings(True), "chr1", 7)
    check("blocks first and unchanged", a[:3] == blocks)
    check("one list per hole", len(a) == 5, f"{len(a)}")
    holes = sorted(a[3:], key=lambda h: h[0].start)
    check(
        "first hole between 10 kb and 20 kb",
        holes[0][0].start >= 10_000 and holes[0][-1].end <= 20_000,
    )
    check(
        "second hole between 25 kb and 60 kb",
        holes[1][0].start >= 25_000 and holes[1][-1].end <= 60_000,
    )
    flat = sorted((x for blk in a for x in blk), key=lambda x: x.start)
    check("bead starts unique", len({x.start for x in flat}) == len(flat))
    check("same seed same beads", a == b)
    check("another seed other beads", a[3:] != c[3:])
    xs = np.array([[x.x, x.y, x.z] for x in holes[1]])
    check(
        "bridge runs from one block to the next",
        14 < float(xs[:, 0].mean()) < 30,
        f"{xs[:, 0].mean():.1f}",
    )


def test_long() -> None:
    print("\n[long] a hole over 500 beads is drawn in two levels")
    blocks = [block(0, 10, 0.0), block(1_210_000, 10, 30.0)]
    a = fill_holes(blocks, settings(True), "chr1", 3)
    hole = a[2]
    xs = np.array([[x.x, x.y, x.z] for x in hole])
    bonds = np.linalg.norm(np.diff(xs, axis=0), axis=1)
    check("one bead per kilobase", len(hole) == 1199, f"{len(hole)}")
    check("finite", bool(np.isfinite(xs).all()))
    check(
        "bonds at the segment joins like the rest",
        float(bonds[[499, 500, 999, 1000]].max()) < 4 * float(np.median(bonds)),
        f"median {np.median(bonds):.2f} joins {bonds[[499, 500, 999, 1000]].round(2)}",
    )


def test_blocks_repel() -> None:
    print("\n[repel] a hole does not pass through a block")
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
        check(
            f"excluded volume {'on' if ev else 'off'}, {want} inside the block",
            ok,
            f"{close} pairs under 0.5",
        )
        check(f"excluded volume {'on' if ev else 'off'}, block beads unmoved", out[2] == c)


def main() -> int:
    test_slots()
    test_fill()
    test_long()
    test_blocks_repel()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
