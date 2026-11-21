"""The stratum adjusted correlation of the pooled map of an arm, split by the separation of the diagonal.

    python playground/band_scc.py MCOOL REGION OUT_DIR ROW ARM [ARM ...]

OUT_DIR holds the structures of a trial, one folder `<ROW>_<ARM>` per cell, region and arm, as
`out/confirm20`. The pooled map is the one of `ensemble_score.py` per bead pair, at 25 kb, against
the balanced counts of MCOOL. `_scc` weights the correlation of a diagonal by its length times the
two standard deviations. This script prints, per arm, the correlation of four bands of diagonals,
under 250 kb, to 1 Mb, to 2 Mb and beyond, each with the share of the weight of the whole score
that its diagonals hold:

    BAND <row> <arm> <whole> <band>:<share> <band>:<share> <band>:<share> <band>:<share>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "playground"))

from validation_battery import CONTACT_BEADS, load  # noqa: E402

from legacy.validation.metrics.hic import observed_hic, simulated_contacts  # noqa: E402

RES = 25_000
EDGES = [1, 10, 40, 80]


def main() -> None:
    mcool, region, out, row, arms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
    bal, bins = observed_hic(mcool, region, RES, balance=True)
    nb = len(bins)
    bands = list(zip(EDGES, [*EDGES[1:], nb], strict=True))
    for arm in arms:
        cifs = sorted((Path(out) / f"{row}_{arm}").glob("*.cif"))
        if not cifs:
            continue
        sim = np.zeros((nb, nb))
        count = np.zeros(nb)
        for c in cifs:
            p, starts, _, _ = load(c)
            r = CONTACT_BEADS * float(np.median(np.linalg.norm(np.diff(p, axis=0), axis=1)))
            sim += simulated_contacts(p, starts, bins, RES, r)
            idx = np.searchsorted(bins, starts, side="right") - 1
            ok = (idx >= 0) & (idx < nb) & (starts < int(bins[-1]) + RES)
            count += np.bincount(idx[ok], minlength=nb)
        pair = np.outer(count, count)
        per_pair = np.divide(sim, pair, out=np.zeros_like(sim), where=pair > 0)
        parts: list[tuple[float, float]] = []
        for lo, hi in bands:
            num = den = 0.0
            for d in range(lo, min(hi, nb - 1)):
                a, b = np.diagonal(per_pair, d), np.diagonal(bal, d)
                sa, sb = float(a.std()), float(b.std())
                if a.size < 2 or sa < 1e-12 or sb < 1e-12:
                    continue
                r_d = float(np.corrcoef(a, b)[0, 1])
                if np.isfinite(r_d):
                    num += a.size * sa * sb * r_d
                    den += a.size * sa * sb
            parts.append((num, den))
        whole = sum(d for _, d in parts)
        cells = " ".join(f"{n / max(d, 1e-300):.4f}:{d / max(whole, 1e-300):.4f}" for n, d in parts)
        print("BAND", row, arm, f"{sum(n for n, _ in parts) / max(whole, 1e-300):.4f}", cells)


if __name__ == "__main__":
    main()
