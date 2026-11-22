"""Paired comparison of the arms of a trial over its rows.

    python playground/arm_stats.py SCORES_DIR REF_ARM [ARM ...]

SCORES_DIR holds the pooled scores of a trial, one file per cell and region, as
`data/figures/uniform-2026-11-19/scores-ev20q/`. A row is a cell and a region. For each arm and
each metric the script prints the mean over the rows, the mean of the difference to REF_ARM on the
same row with its standard error, the number of rows above REF_ARM and the p value of the Wilcoxon
signed rank test of the differences. Without ARM every arm of the folder is compared.

The rows of one cell share its data and are not independent draws. The test reads them as
independent, and its p value is a lower bound.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uniform_arms_figure import read_scores  # noqa: E402

METRICS = [
    ("raw_scc_pair", "SCC raw, per bead pair"),
    ("bal_scc_pair", "SCC balanced, per bead pair"),
    ("raw_sp", "Spearman raw"),
    ("bal_p", "Pearson balanced"),
]


def main() -> None:
    folder, ref = Path(sys.argv[1]), sys.argv[2]
    data = read_scores(folder)
    arms = sys.argv[3:] or sorted({a for row in data.values() for a in row} - {ref})
    for key, title in METRICS:
        print(f"{title}, against {ref}")
        print(f"  {'arm':24s} rows    mean   difference        above   p")
        for arm in [ref, *arms]:
            rows = [r for r in data if key in data[r].get(arm, {}) and key in data[r].get(ref, {})]
            if not rows:
                continue
            v = np.array([data[r][arm][key] for r in rows])
            d = v - np.array([data[r][ref][key] for r in rows])
            if arm == ref:
                print(f"  {arm:24s} {len(rows):4d}  {v.mean():6.3f}")
                continue
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            p = float(wilcoxon(d).pvalue) if len(d) > 5 and np.any(d != 0) else float("nan")
            print(
                f"  {arm:24s} {len(rows):4d}  {v.mean():6.3f}  {d.mean():+.3f} ± {se:.3f}"
                f"  {int((d > 0).sum()):2d} of {len(d):2d}  {p:.2g}"
            )


if __name__ == "__main__":
    main()
