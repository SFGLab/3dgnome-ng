"""Unit checks for the pile-up at RNAPII loops, `playground/rnapii/pileup.py`.

    python harness/test_pileup.py

A pile-up averages the window around each loop's pixel after dividing every pixel by the mean
of its diagonal, so a flat map scores one and a planted loop scores its own enrichment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "playground" / "rnapii"))

from pileup import apa_score, pile  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []
BIN = 10_000


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}   {detail}")


def loop(a: int, b: int) -> tuple[str, int, int, str, int, int, int]:
    return ("chr1", a * BIN, a * BIN + 100, "chr1", b * BIN, b * BIN + 100, 5)


def test_planted_enrichment() -> None:
    n = 200
    m = np.ones((n, n))
    m[50, 80] = m[80, 50] = 5.0
    p, used = pile(m, np.arange(n) * BIN, BIN, [loop(50, 80)], flank=10, min_sep=5)
    score = apa_score(p)
    # the planted pixel lifts its own diagonal's mean a little, so the centre reads just under 5
    check("a planted loop scores its enrichment", used == 1 and 4.8 < score < 5.0, f"{score:.3f}")


def test_flat_map_scores_one() -> None:
    n = 200
    m = np.ones((n, n))
    p, used = pile(m, np.arange(n) * BIN, BIN, [loop(50, 80), loop(100, 140)], flank=10, min_sep=5)
    check("a flat map scores one", used == 2 and abs(apa_score(p) - 1.0) < 1e-9)


def test_short_and_edge_loops_skipped() -> None:
    n = 200
    m = np.ones((n, n))
    _, used = pile(m, np.arange(n) * BIN, BIN, [loop(50, 52)], flank=10, min_sep=5)
    check("a loop under the minimum separation is not used", used == 0)
    _, used = pile(m, np.arange(n) * BIN, BIN, [loop(5, 100)], flank=10, min_sep=5)
    check("a loop whose window leaves the map is not used", used == 0)


def main() -> int:
    test_planted_enrichment()
    test_flat_map_scores_one()
    test_short_and_edge_loops_skipped()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
