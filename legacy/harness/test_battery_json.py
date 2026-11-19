"""The battery and the saddle write one JSON row per arm with `--json`.

    python legacy/harness/test_battery_json.py

The figure reads these rows instead of parsing printed tables. Two arms of three random walk
structures are scored against a small synthetic cooler, and each file must hold one row per arm
with every key and the structure count.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gnome3d.io import write_cif  # noqa: E402
from gnome3d.types import BeadOut  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []
SPAN = 5_000_000
BATTERY_KEYS = {
    "arm", "n", "pearson", "spearman", "scc", "multimm", "exponent",
    "e_short", "e_long", "rg", "wb_aa", "wb_sa", "xb",
}  # fmt: skip
SADDLE_KEYS = {"arm", "n", "saddle", "eig_r", "kappa"}


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}   {detail}")


def make_cooler(path: Path, binsize: int, balanced: bool = True) -> None:
    """A cooler on chr1:0-5 Mb whose contacts decay with separation, balanced unless asked not."""
    import cooler

    starts = np.arange(0, SPAN, binsize)
    bins = pd.DataFrame({"chrom": "chr1", "start": starts, "end": starts + binsize})
    i, j = np.triu_indices(len(starts))
    sep = (j - i).astype(np.float64)
    count = np.maximum(1, (1000 / (1 + sep) ** 1.0).astype(np.int64))
    pixels = pd.DataFrame({"bin1_id": i, "bin2_id": j, "count": count})
    cooler.create_cooler(str(path), bins, pixels, ordered=True)
    if balanced:
        cooler.balance_cooler(cooler.Cooler(str(path)), store=True, ignore_diags=1)


def make_arm(d: Path, seed: int) -> None:
    """Three random walk chains of 500 beads over 0-5 Mb, an anchor every tenth bead."""
    d.mkdir()
    rng = np.random.default_rng(seed)
    step = SPAN // 500
    for k in range(3):
        pos = np.cumsum(rng.normal(size=(500, 3)), axis=0)
        beads = [
            BeadOut(
                i * step, i * step + step - 1, *pos[i], "anchor" if i % 10 == 0 else "subanchor"
            )
            for i in range(500)
        ]
        write_cif(str(d / f"m{k}.cif"), beads)


def main() -> int:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        make_cooler(tmp / "map.cool", 25_000)
        make_arm(tmp / "armA", 1)
        make_arm(tmp / "armB", 2)
        arms = [str(tmp / "armA"), str(tmp / "armB")]
        out = tmp / "battery.json"
        subprocess.run(
            [sys.executable, str(ROOT / "playground" / "validation_battery.py"), "--balance", "no",
             "--json", str(out), str(tmp / "map.cool"), f"chr1:0-{SPAN}", "25000", *arms],
            check=False, capture_output=True, cwd=ROOT,
        )  # fmt: skip
        rows = json.loads(out.read_text()) if out.is_file() else []
        check(
            "the battery writes one row per arm with every key",
            len(rows) == 2 and all(set(r) == BATTERY_KEYS for r in rows),
            str([sorted(r) for r in rows][:1]),
        )
        check(
            "each battery row counts three structures",
            bool(rows) and all(r.get("n") == 3 for r in rows),
        )

        track = tmp / "track.bedGraph"
        track.write_text(
            "".join(
                f"chr1\t{s}\t{s + 500_000}\t{1 if (s // 500_000) % 2 else -1}\n"
                for s in range(0, SPAN, 500_000)
            )
        )
        out = tmp / "saddle.json"
        subprocess.run(
            [sys.executable, str(ROOT / "playground" / "saddle_offline.py"), "--json", str(out),
             str(tmp / "map.cool"), f"chr1:0-{SPAN}", "25000", str(track), *arms],
            check=False, capture_output=True, cwd=ROOT,
        )  # fmt: skip
        rows = json.loads(out.read_text()) if out.is_file() else []
        check(
            "the saddle writes one row per arm with every key",
            len(rows) == 2 and all(set(r) == SADDLE_KEYS for r in rows) and rows[0]["n"] == 3,
            str(rows[:1]),
        )
        make_cooler(tmp / "raw.cool", 25_000, balanced=False)
        (tmp / "empty").mkdir()
        out = tmp / "saddle_raw.json"
        subprocess.run(
            [sys.executable, str(ROOT / "playground" / "saddle_offline.py"), "--balance", "no", "--json", str(out),
             str(tmp / "raw.cool"), f"chr1:0-{SPAN}", "25000", str(track), arms[0]],
            check=False, capture_output=True, cwd=ROOT,
        )  # fmt: skip
        check("the saddle reads a map with no weights under --balance no", out.is_file())
        out = tmp / "battery_empty.json"
        subprocess.run(
            [sys.executable, str(ROOT / "playground" / "validation_battery.py"), "--balance", "no",
             "--json", str(out), str(tmp / "map.cool"), f"chr1:0-{SPAN}", "25000", arms[0], str(tmp / "empty")],
            check=False, capture_output=True, cwd=ROOT,
        )  # fmt: skip
        rows = json.loads(out.read_text()) if out.is_file() else []
        check(
            "an arm with no structures is a row with n zero, not a missing row",
            len(rows) == 2 and rows[1]["arm"] == "empty" and rows[1]["n"] == 0,
            str([(r.get("arm"), r.get("n")) for r in rows]),
        )
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
