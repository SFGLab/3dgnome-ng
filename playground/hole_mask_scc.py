"""What each hole costs the pooled SCC.

    python playground/hole_mask_scc.py MCOOL REGION HOLE_ARM ARM [ARM ...]

Holes are taken from the first structure of HOLE_ARM, which should carry no hole beads. For each
arm's pooled map the SCC against raw and balanced Hi-C is printed over the mapped bins, then with
each hole's bins left out of both maps in turn, then with every hole's bins left out. Per hole the
share of its bins Hi-C maps and their raw coverage over the region's median are printed, the two
things that say how well the map sees it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "playground"))

from ensemble_score import RES, scores  # noqa: E402
from hole_readout import holes, read_spans  # noqa: E402
from law_spring_readout import read_models  # noqa: E402
from validation_battery import CONTACT_BEADS, load  # noqa: E402

from legacy.validation.metrics.hic import observed_hic, simulated_contacts  # noqa: E402


def main() -> None:
    mcool, region, hole_arm = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    raw, bins = observed_hic(mcool, region, RES, balance=False)
    bal, _ = observed_hic(mcool, region, RES, balance=True)
    mapped = bal.sum(axis=0) > 0
    cover = raw.sum(axis=0)
    med = float(np.median(cover[mapped])) if mapped.any() else 1.0
    _, mids, kinds = read_models(hole_arm)
    st, en = read_spans(hole_arm)
    h = holes(mids, st, en, kinds)
    centre = bins + RES // 2
    per = [(centre > a) & (centre < b) for a, b in h]
    big = [(k, m) for k, m in enumerate(per) if m.sum() >= 2]
    print(f"{region}: {len(h)} holes, {len(big)} spanning two bins or more")
    for k, m in big:
        a, b = h[k]
        print(
            f"  hole {k}: {a / 1e6:.2f} to {b / 1e6:.2f} Mb, {int(m.sum())} bins, "
            f"{float(mapped[m].mean()):.2f} mapped, raw coverage {float(np.median(cover[m])) / med:.2f} of the median"
        )
    any_hole = np.zeros(len(bins), dtype=bool)
    for m in per:
        any_hole |= m
    cols = ["mapped"] + [f"no hole {k}" for k, _ in big] + ["no holes"]
    keeps = [mapped] + [mapped & ~m for _, m in big] + [mapped & ~any_hole]
    print("  SCC raw / balanced, " + ", ".join(cols))
    for d in map(Path, sys.argv[4:]):
        cifs = sorted(d.glob("*.cif"))
        if not cifs:
            continue
        sim = np.zeros((len(bins), len(bins)))
        for c in cifs:
            p, m, _, _ = load(c)
            r = CONTACT_BEADS * float(np.median(np.linalg.norm(np.diff(p, axis=0), axis=1)))
            sim += simulated_contacts(p, m, bins, RES, r)
        cells = [f"{scores(sim, raw, k)[2]:.3f}/{scores(sim, bal, k)[2]:.3f}" for k in keeps]
        print(f"  {d.name:>44s}  " + "  ".join(c.rjust(13) for c in cells))


if __name__ == "__main__":
    main()
