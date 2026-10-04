"""Turn the reference's .hcm models into the battery's cif form, one arm directory.

    python playground/reference_arm.py ANCHORS.bed HCM_DIR OUT_DIR

The reference writes one .hcm per structure with every leaf bead's genomic span and position
and no bead kind. A leaf whose midpoint falls inside an anchor of the cell's anchors bed is
marked an anchor, every other leaf a subanchor, so the battery splits its overlap columns the
way it does for ours. The .hcm parser is the integration harness's.
"""

from __future__ import annotations

import bisect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnome3d.io import write_cif  # noqa: E402
from gnome3d.types import BeadOut  # noqa: E402
from harness.integration import parse_hcm  # noqa: E402


def anchor_spans(bed: Path, chrom: str) -> tuple[list[int], list[int]]:
    starts: list[int] = []
    ends: list[int] = []
    for line in bed.read_text().splitlines():
        f = line.split()
        if len(f) >= 3 and f[0] == chrom:
            starts.append(int(f[1]))
            ends.append(int(f[2]))
    order = sorted(range(len(starts)), key=lambda i: starts[i])
    return [starts[i] for i in order], [ends[i] for i in order]


def is_anchor(mid: int, starts: list[int], ends: list[int]) -> bool:
    k = bisect.bisect_right(starts, mid) - 1
    return k >= 0 and starts[k] <= mid < ends[k]


def convert(hcm: Path, starts: list[int], ends: list[int], out: Path) -> tuple[int, int]:
    leaves = parse_hcm(hcm)
    beads = [
        BeadOut(s, e, x, y, z, "anchor" if is_anchor((s + e) // 2, starts, ends) else "subanchor")
        for s, e, x, y, z, _ in leaves
    ]
    write_cif(str(out), beads)
    return len(beads), sum(b.kind == "anchor" for b in beads)


def main() -> None:
    bed, hcm_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    hcms = sorted(hcm_dir.rglob("*.hcm"))
    if not hcms:
        raise SystemExit(f"no .hcm under {hcm_dir}")
    chrom = None
    for i, h in enumerate(hcms, start=1):
        first = parse_hcm(h)[0]
        if chrom is None:
            # the harness parser carries no chromosome; the bed is filtered on the one in the name
            chrom = sys.argv[4] if len(sys.argv) > 4 else "chr1"
            starts, ends = anchor_spans(bed, chrom)
        n, a = convert(h, starts, ends, out_dir / f"reference_s{i}.cif")
        print(f"{h.name}: {n} beads, {a} anchors, first bead at {first[0]:,}")


if __name__ == "__main__":
    main()
