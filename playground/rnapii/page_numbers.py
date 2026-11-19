"""The numbers of the site's Results page from the readouts of the RNAPII comparison.

    python playground/rnapii/page_numbers.py <readouts> <fig_inputs>

`readouts` holds `<LINE>_battery.json` and `e3d_<ARM>/`, `fig_inputs` holds `ep_<LINE>.tsv` from
`ep_pairs.py`. Prints the three charts of `web/frontend/src/pages/Results.tsx`. Agreement is the
battery's stratum adjusted correlation per line, arms E and C. Distance is the median over the
pairs an RNAPII loop alone holds, arms C and R, with how much closer R is. The link is the
median bootstrap draw of the correlation between the change in distance and the change in
expression, sign flipped, arms E, C and R.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fig_data import agreement, link_draws  # noqa: E402


def main() -> None:
    readouts, fig_inputs = Path(sys.argv[1]), Path(sys.argv[2])
    ag = agreement(readouts)
    hic = ag[ag.measure == "hic"].pivot(index="line", columns="arm", values="value")
    print("agreement with Hi-C, SCC")
    print(hic.round(3).to_string())
    print("\nenhancer to promoter distance, RNAPII only pairs, median")
    for path in sorted([*fig_inputs.glob("ep_*.tsv"), *fig_inputs.glob("ep_*.tsv.gz")]):
        t = pd.read_csv(path, sep="\t")
        t = t[t["set"] == "rnapii_only"]
        med = t.groupby("arm").distance.median()
        n = int(t[t.arm == "C"].shape[0])
        print(f"  {path.name[3:].split('.')[0]:8s} C {med['C']:.2f}  R {med['R']:.2f}  {100 * (1 - med['R'] / med['C']):.0f}% closer  {n} pairs")
    arms = tuple(a for a in ("E", "C", "R", "M") if (readouts / f"e3d_{a}" / "summary.tsv").exists())
    draws = link_draws(readouts, arms)
    med = draws.groupby(["pair", "arm"]).value.median().unstack()
    lo = draws.groupby(["pair", "arm"]).value.quantile(0.025).unstack()
    hi = draws.groupby(["pair", "arm"]).value.quantile(0.975).unstack()
    print("\nlink to expression, median bootstrap draw, sign flipped")
    print(med.round(3).to_string())
    print("95 percent interval")
    print((lo.round(2).astype(str) + " to " + hi.round(2).astype(str)).to_string())


if __name__ == "__main__":
    main()
