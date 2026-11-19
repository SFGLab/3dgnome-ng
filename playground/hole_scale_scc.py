"""Whether the pooled SCC penalises a hole's level or its pattern.

    python playground/hole_scale_scc.py MCOOL REGION HOLE_ARM ARM

Holes are taken from the first structure of HOLE_ARM, which should carry no hole beads. ARM's
pooled map is scored over the mapped bins with every count touching a hole bin multiplied by 0,
0.25, 0.5, 1 and 2. A score that falls steadily from 0 says the measure rewards the hole's level
being extreme, whatever its pattern. Then, per separation of 25 to 1000 kb, the Spearman
correlation of the model and Hi-C over pairs touching a hole bin alone, raw and balanced, which
says whether the hole's contacts follow Hi-C's from one pair to the next.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "playground"))

from ensemble_score import RES, scores  # noqa: E402
from hole_readout import holes, read_spans  # noqa: E402
from law_spring_readout import read_models  # noqa: E402
from validation_battery import CONTACT_BEADS, load  # noqa: E402

from legacy.validation.metrics.hic import observed_hic, simulated_contacts  # noqa: E402

BANDS = ((1, 2), (2, 4), (4, 8), (8, 16), (16, 40))


def main() -> None:
    mcool, region, hole_arm, arm = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
    raw, bins = observed_hic(mcool, region, RES, balance=False)
    bal, _ = observed_hic(mcool, region, RES, balance=True)
    mapped = bal.sum(axis=0) > 0
    _, mids, kinds = read_models(hole_arm)
    st, en = read_spans(hole_arm)
    centre = bins + RES // 2
    hole = np.zeros(len(bins), dtype=bool)
    for a, b in holes(mids, st, en, kinds):
        hole |= (centre > a) & (centre < b)
    sim = np.zeros((len(bins), len(bins)))
    for c in sorted(arm.glob("*.cif")):
        p, m, _, _ = load(c)
        r = CONTACT_BEADS * float(np.median(np.linalg.norm(np.diff(p, axis=0), axis=1)))
        sim += simulated_contacts(p, m, bins, RES, r)
    touch = hole[:, None] | hole[None, :]
    print(f"{region} {arm.name}: {int((hole & mapped).sum())} mapped hole bins")
    print("  hole counts scaled by   SCC raw / balanced over mapped bins")
    for f in (0.0, 0.25, 0.5, 1.0, 2.0):
        s = np.where(touch, sim * f, sim)
        print(f"  {f:>22.2f}   {scores(s, raw, mapped)[2]:.3f} / {scores(s, bal, mapped)[2]:.3f}")
    print("  Spearman over pairs touching a hole bin, raw / balanced, by separation")
    n = len(bins)
    i, j = np.triu_indices(n, 1)
    ok = mapped[i] & mapped[j] & touch[i, j]
    for lo, up in BANDS:
        sel = ok & (j - i >= lo) & (j - i < up)
        if sel.sum() < 20:
            continue
        a = sim[i[sel], j[sel]]
        rr = spearmanr(a, raw[i[sel], j[sel]]).statistic
        rb = spearmanr(a, bal[i[sel], j[sel]]).statistic
        print(f"  {lo * 25:>5d} to {up * 25:>5d} kb  n {int(sel.sum()):5d}   {rr:+.3f} / {rb:+.3f}")


if __name__ == "__main__":
    main()
