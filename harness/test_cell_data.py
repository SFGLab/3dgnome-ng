"""Unit checks for the cell line data section.

    python harness/test_cell_data.py

The Hi-C singletons file is the input when it exists and the ChIA-PET singletons when it does
not, since a deep map is not always available and nothing may require one.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation.core.config import cell_data_section  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def main() -> int:
    print("cell data section checks")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "GM12878").mkdir()
        d = cell_data_section("GM12878", str(root))
        check(
            "without a Hi-C file the ChIA-PET singletons are the input",
            d["singletons"] == "GM12878_singletons_lessthan3.bedpe",
            str(d["singletons"]),
        )
        (root / "GM12878" / "GM12878_hic_25kb_singletons.bedpe").write_text("")
        d = cell_data_section("GM12878", str(root))
        check(
            "with a Hi-C file it is the input",
            d["singletons"] == "GM12878_hic_25kb_singletons.bedpe",
            str(d["singletons"]),
        )
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
