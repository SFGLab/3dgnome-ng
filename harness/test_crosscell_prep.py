"""Unit checks for the cross line RNAPII inputs, `playground/rnapii/crosscell_prep.py`.

    python harness/test_crosscell_prep.py

Loops from two ENCODE replicates are pooled, filtered on the line's peaks and span, thinned to a
common depth and, for the merged anchor arm, their ends moved onto nearby CTCF anchors. Each
check builds a few loops by hand whose expected result is plain from the rule.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "playground" / "rnapii"))

from crosscell_prep import (  # noqa: E402
    PeakIndex,
    absorb_ends,
    build_record,
    check_keep_fractions,
    depth,
    filter_loops,
    pool_replicates,
    read_bedpe,
    thin,
)

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}   {detail}")


def test_read_normalises(tmp: Path) -> None:
    p = tmp / "x.bedpe"
    p.write_text("chr1\t500\t600\tchr1\t100\t200\t3.0\nchr1\t1\t2\tchr2\t5\t6\t9\n")
    got = read_bedpe(p)
    check(
        "a reversed row with a float PET count is normalised and a trans row dropped",
        got == [("chr1", 100, 200, "chr1", 500, 600, 3)],
        str(got),
    )


def test_pool_sums_overlapping() -> None:
    a = [("chr1", 100, 200, "chr1", 5000, 5100, 4)]
    b = [
        ("chr1", 150, 260, "chr1", 5050, 5200, 2),
        ("chr1", 9000, 9100, "chr1", 20000, 20100, 3),
    ]
    pooled = pool_replicates([a, b])
    check(
        "loops with both ends overlapping are one loop with summed PETs and union ends",
        ("chr1", 100, 260, "chr1", 5000, 5200, 6) in pooled
        and ("chr1", 9000, 9100, "chr1", 20000, 20100, 3) in pooled
        and len(pooled) == 2,
        str(pooled),
    )


def test_filter_rules() -> None:
    peaks = PeakIndex({"chr1": [(90, 210)]})
    one_end = ("chr1", 100, 200, "chr1", 5000, 5100, 5)
    far = ("chr1", 100, 200, "chr1", 2_000_000, 2_000_100, 5)
    check(
        "one end on a peak passes the RNAPII rule",
        filter_loops([one_end], peaks, ends_on_peaks=1, max_span=1_000_000) == [one_end],
    )
    check(
        "one end on a peak fails the CTCF rule",
        filter_loops([one_end], peaks, ends_on_peaks=2, max_span=1_000_000) == [],
    )
    check(
        "a loop over the span cap is dropped",
        filter_loops([far], peaks, ends_on_peaks=1, max_span=1_000_000) == [],
    )


def test_thin_keeps_expectation() -> None:
    loops = [
        ("chr1", i * 1000, i * 1000 + 100, "chr1", i * 1000 + 50_000, i * 1000 + 50_100, 40)
        for i in range(2000)
    ]
    out = thin(loops, 0.5, np.random.default_rng(1), min_pets=3)
    ratio = depth(out) / depth(loops)
    check("thinning at one half keeps half the PETs", abs(ratio - 0.5) < 0.02, f"{ratio:.4f}")
    check("no thinned loop is under three PETs", all(lp[6] >= 3 for lp in out))


def test_absorb() -> None:
    ctcf = [("chr1", 1000, 2000, "R")]
    loops = [("chr1", 2500, 2600, "chr1", 90_000, 90_100, 5)]
    out, n = absorb_ends(loops, ctcf, absorb_bp=1000)
    check(
        "an end 500 bp from a CTCF anchor takes the anchor's interval",
        out == [("chr1", 1000, 2000, "chr1", 90_000, 90_100, 5)] and n == 1,
        str(out),
    )
    out, n = absorb_ends(loops, ctcf, absorb_bp=100)
    check("an end beyond the absorb distance is left as it is", out == loops and n == 0)


def test_keep_fraction_guard() -> None:
    try:
        check_keep_fractions({"A": 100, "B": 1000, "C": 1000}, floor=0.2)
        stopped = False
    except SystemExit as e:
        stopped = "B" in str(e)
    check("a line kept at a tenth of its depth stops the prep and is named", stopped)
    got = check_keep_fractions({"A": 800, "B": 1000}, floor=0.2)
    check("fractions above the floor come back", got == {"A": 1.0, "B": 0.8}, str(got))


def test_record_names_bj() -> None:
    rec = build_record("HFFC6", rnapii_source="BJ ENCSR956IXE", seed=1)
    check(
        "the record carries the line and the BJ source",
        rec["line"] == "HFFC6" and "BJ" in str(rec["rnapii_source"]) and rec["seed"] == 1,
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        test_read_normalises(Path(d))
    test_pool_sums_overlapping()
    test_filter_rules()
    test_thin_keeps_expectation()
    test_absorb()
    test_keep_fraction_guard()
    test_record_names_bj()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
