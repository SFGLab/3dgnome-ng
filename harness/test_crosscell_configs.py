"""Unit checks for the cross line arm configs, `slurm/ensemble/crosscell_configs.py`.

    python harness/test_crosscell_configs.py

Arms C, R and M of one line differ only in which loops and anchors they read and in the two
factor keys, so every other section must match the production config of that line.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "slurm" / "ensemble"))

from crosscell_configs import build  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}   {detail}")


def test_arms() -> None:
    c, r, m = (build("H1ESC", a) for a in ("C", "R", "M"))
    check(
        "C reads the thinned CTCF loops", c["data"]["clusters"] == "H1ESC_xc_ctcf_clusters_3+.bedpe"
    )
    check(
        "C names no factors", "factors" not in c["data"] and "factor_strength" not in c["springs"]
    )
    check(
        "R reads both factors at full strength",
        r["data"]["factors"] == "CTCF,RNAPOL2"
        and r["springs"]["factor_strength"] == "1,1"
        and r["data"]["clusters"]
        == "H1ESC_xc_ctcf_clusters_3+.bedpe,H1ESC_xc_rnapii_clusters_3+.bedpe",
    )
    check(
        "M reads the absorbed loops and anchors",
        m["data"]["anchors"] == "H1ESC_xc_anchors_ctcf_rnapii_absorbed.bed"
        and m["data"]["clusters"].endswith("H1ESC_xc_rnapii_absorbed_clusters_3+.bedpe"),
    )
    same = all(
        dict(c[sec].items()) == {k: v for k, v in r[sec].items() if k != "factor_strength"}
        for sec in c.sections()
        if sec != "data"
    )
    check("every section but data is the same in C and R, bar factor_strength", same)


def main() -> int:
    test_arms()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
