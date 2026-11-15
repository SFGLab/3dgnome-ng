"""Hi-C agreement of each arm's pooled ensemble map rather than of each structure alone.

    python playground/ensemble_score.py MCOOL REGION ARM_DIR [ARM_DIR ...]

The battery scores every structure against Hi-C and averages, which rewards structures that each
look like the population average. Hi-C is that average, so this pools the contacts of every
structure of an arm into one map first, at the battery's contact radius, and prints Pearson on
log counts, Spearman and SCC against raw and balanced Hi-C at 25 kb.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "playground"))

from validation_battery import CONTACT_BEADS, load  # noqa: E402

from validation.metrics.hic import ensemble_hic_correlation  # noqa: E402

RES = 25_000


def main() -> None:
    mcool, region = sys.argv[1], sys.argv[2]
    print(f"{region}, pooled ensemble against Hi-C at {RES} bp")
    print(f"  {'arm':>44s} {'n':>3s}   raw P  raw Sp  raw SCC   bal P  bal Sp  bal SCC")
    for d in map(Path, sys.argv[3:]):
        cifs = sorted(d.glob("*.cif"))
        if not cifs:
            continue
        coords, mids, radii = [], [], []
        for c in cifs:
            p, m, _, _ = load(c)
            coords.append(p)
            mids.append(m)
            radii.append(CONTACT_BEADS * float(np.median(np.linalg.norm(np.diff(p, axis=0), axis=1))))
        r = float(np.median(radii))
        out = []
        for bal in (False, True):
            res = ensemble_hic_correlation(coords, mids, mcool, region, RES, r, balance=bal)
            out += [res["pearson"], res["spearman"], res["scc"]]
        print(f"  {d.name:>44s} {len(cifs):3d}  " + "  ".join(f"{v:6.3f}" for v in out))


if __name__ == "__main__":
    main()
