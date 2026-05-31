"""Golden regression gate for the task-DAG pipeline.

`reconstruct` is deterministic (coarse engine seeded, per-IB stage seeds), so a
fixed region has an exact, reproducible signature: the RNG-independent bead
*layout* (every ``(start, end, kind)``) and the deterministic gyration radius.
This records that signature once as a golden and asserts future runs match it —
the forward safety net for the remaining pure-relocation steps (mc/ split) and a
drift flag for the dedup phase.

(Replaces the earlier vs-Solver comparison: Solver was dissolved into
`gnome3d.coarse` + the pipeline, so the live baseline is gone; the golden is the
recorded pre-dissolution behavior — 1673 beads, gyration ~5.9.)

    python playground/refactor/validate_pipeline.py          # check (writes golden if missing)
    python playground/refactor/validate_pipeline.py --update  # regenerate the golden
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from _common import load_region  # noqa: E402

from gnome3d.reconstruct import reconstruct  # noqa: E402

GOLDEN = Path("playground/refactor/golden_chr1_slice.json")


def _gyration(beads) -> float:
    xyz = np.array([(b.x, b.y, b.z) for b in beads], dtype=np.float64)
    return float(np.linalg.norm(xyz - xyz.mean(axis=0), axis=1).mean())


def _signature(beads) -> dict:
    layout = sorted((b.start, b.end, b.kind) for b in beads)
    return {
        "n_beads": len(beads),
        "layout_sha": hashlib.sha256(repr(layout).encode()).hexdigest()[:16],
        "gyration": round(_gyration(beads), 4),
    }


def main() -> int:
    s, bed, data, _ = load_region()
    beads = reconstruct(s, data, [bed.chr], bed)[bed.chr]
    rerun = reconstruct(s, data, [bed.chr], bed)[bed.chr]
    cur = _signature(beads)
    deterministic = len(beads) == len(rerun) and all(a == b for a, b in zip(beads, rerun, strict=True))

    if "--update" in sys.argv or not GOLDEN.exists():
        GOLDEN.write_text(json.dumps(cur, indent=2) + "\n")
        print(f"  wrote golden: {cur}")
        print("PASS (golden written)")
        return 0

    gold = json.loads(GOLDEN.read_text())
    ok = True
    layout_ok = cur["layout_sha"] == gold["layout_sha"] and cur["n_beads"] == gold["n_beads"]
    ok &= layout_ok
    print(f"  {'ok ' if layout_ok else 'FAIL'} bead layout matches golden ({cur['n_beads']} beads, sha {cur['layout_sha']})")
    gyr_rel = abs(cur["gyration"] - gold["gyration"]) / max(gold["gyration"], 1e-9)
    gyr_ok = gyr_rel < 0.01  # deterministic -> should be exact; tolerance flags dedup drift
    ok &= gyr_ok
    print(f"  {'ok ' if gyr_ok else 'FAIL'} gyration matches golden ({cur['gyration']} vs {gold['gyration']}, rel {gyr_rel:.2%})")
    ok &= deterministic
    print(f"  {'ok ' if deterministic else 'FAIL'} deterministic (rerun identical)")
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
