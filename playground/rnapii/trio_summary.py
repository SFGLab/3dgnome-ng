"""The trio structure readouts as one paired table, C against R per person.

    python playground/rnapii/trio_summary.py <readouts/trio dir>

Reads `<SAMPLE>_{battery,saddle,pileup}.json` of `readouts_trio.sh` and prints per person the
CTCF only and CTCF and RNAPII values of SCC, Pearson, MultiMM, the saddle and the pile-up at
RNAPII loops over the map's, then the median paired difference and in how many of the nine R
is higher. Standard library only, so it runs on a login node.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

PEOPLE = ("HG00512", "HG00513", "HG00514", "HG00731", "HG00732", "HG00733", "GM19238", "GM19239", "GM19240")


def arms(path: Path) -> dict[str, dict[str, float]]:
    return {r["arm"].split("_")[1]: r for r in json.loads(path.read_text()) if r["arm"] != "map"}


def main() -> None:
    d = Path(sys.argv[1])
    rows = {}
    for s in PEOPLE:
        b, sa = arms(d / f"{s}_battery.json"), arms(d / f"{s}_saddle.json")
        pu = json.loads((d / f"{s}_pileup.json").read_text())
        ref = next(r["apa"] for r in pu if r["arm"] == "map")
        pa = {r["arm"].split("_")[1]: r["apa"] / ref for r in pu if r["arm"] != "map"}
        rows[s] = {
            "scc": (b["C"]["scc"], b["R"]["scc"]),
            "pearson": (b["C"]["pearson"], b["R"]["pearson"]),
            "multimm": (b["C"]["multimm"], b["R"]["multimm"]),
            "saddle": (sa["C"]["saddle"], sa["R"]["saddle"]),
            "apa_vs_map": (pa["C"], pa["R"]),
            "n": (b["C"]["n"], b["R"]["n"]),
        }
    keys = ("scc", "pearson", "multimm", "saddle", "apa_vs_map")
    print("person   " + "  ".join(f"{k:>15s}" for k in keys) + "   n")
    for s, v in rows.items():
        print(f"{s}  " + "  ".join("{:>7.3f} {:>7.3f}".format(*v[k]) for k in keys) + f"   {v['n']}")
    for k in keys:
        diff = [v[k][1] - v[k][0] for v in rows.values()]
        print(f"{k:>11s}: R minus C median {statistics.median(diff):+.3f}, R higher in {sum(x > 0 for x in diff)} of 9")


if __name__ == "__main__":
    main()
