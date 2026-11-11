"""The readouts of the RNAPII comparison as the manifest `figure_rnapii.py` reads.

    python playground/rnapii/collect_readouts.py <readouts dir> <prep records dir> <out dir>

Reads `<LINE>_{battery,saddle,pileup}.json` for the three lines, `trio/<SAMPLE>_*.json` for
the nine people, and `e3d_<ARM>/summary.tsv` of the cross line CLI. Arm directories are named
`<DATASET>_<ARM>_60mb`. The cross line statistics belong to a pair of lines, so they become
datasets of kind pair, `GM12878 vs H1ESC`, and a pair's structure readouts are the mean of its
two lines', which lets the trade-off panel place it. Writes the expression rows as CSVs and the
manifest into the out dir, then draws the figures there.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

LINES = ("GM12878", "H1ESC", "HFFC6")
ARM_DIR = re.compile(r"^(?P<ds>[A-Za-z0-9]+)_(?P<arm>[ECRM])_60mb$")
CROSS = ("r_total", "r_set", "r_conf", "jump_r_conf")


def _arms(path: Path) -> dict[str, str]:
    """Each row's arm directory name mapped to its arm letter."""
    out = {}
    for r in json.loads(path.read_text()):
        m = ARM_DIR.match(str(r["arm"]))
        if m:
            out[str(r["arm"])] = m["arm"]
    return out


def build(readouts: Path, records: Path, out: Path) -> dict[str, object]:
    out.mkdir(parents=True, exist_ok=True)
    man: dict[str, object] = {"battery": [], "saddle": [], "pileup": [], "microc": [], "expression": [], "records": {}, "piles": {}}
    for kind, base, names in (("line", readouts, LINES), ("person", readouts / "trio", None)):
        found = sorted({f.name.split("_")[0] for f in base.glob("*_*.json")}) if base.is_dir() else []
        for ds in found:
            if names is not None and ds not in names:
                continue
            for what in ("battery", "saddle", "pileup", "microc"):
                p = base / f"{ds}_{what}.json"
                if p.is_file() and p.stat().st_size > 2:
                    man[what].append({"dataset": ds, "kind": kind, "path": str(p), "arms": _arms(p)})  # type: ignore[union-attr]
            npz = base / f"{ds}_pileup.npz"
            if kind == "line" and ds == "GM12878" and npz.is_file():
                for key in ("map", *[k for k in _arms(base / f"{ds}_pileup.json")]):
                    arm = "map" if key == "map" else ARM_DIR.match(key)["arm"]  # type: ignore[index]
                    man["piles"][f"{ds}/{arm}"] = {"path": str(npz), "key": key}  # type: ignore[index]
    rows = []
    for d in sorted(readouts.glob("e3d_*")):
        s = d / "summary.tsv"
        if not s.is_file():
            continue
        arm = d.name.split("_", 1)[1]
        t = pd.read_csv(s, sep="\t")
        for r in t.itertuples(index=False):
            pair = str(r.pair).replace("->", " vs ")
            n = int(getattr(r, "n_with_deseq", 10))
            rows += [{"dataset": pair, "arm": arm, "metric": m, "value": float(getattr(r, m)), "n": 10, "genes": n} for m in CROSS if hasattr(r, m)]
    ex = pd.DataFrame(rows)
    if len(ex):
        line_scc = {}
        for e in man["battery"]:  # type: ignore[union-attr]
            if e["kind"] == "line":
                for r in json.loads(Path(e["path"]).read_text()):
                    if str(r["arm"]) in e["arms"]:
                        line_scc[(e["dataset"], e["arms"][str(r["arm"])])] = (float(r["scc"]), int(r["n"]))
        for pair, g in ex.groupby("dataset"):
            a, b = pair.split(" vs ")
            path = out / f"expression_{a}_{b}.csv"
            extra = []
            for arm in sorted(set(g["arm"])):
                if (a, arm) in line_scc and (b, arm) in line_scc:
                    v = (line_scc[(a, arm)][0] + line_scc[(b, arm)][0]) / 2
                    extra.append({"arm": arm, "metric": "scc", "value": v, "n": min(line_scc[(a, arm)][1], line_scc[(b, arm)][1])})
            pd.concat([g[["arm", "metric", "value", "n"]], pd.DataFrame(extra)]).to_csv(path, index=False)
            man["expression"].append({"dataset": pair, "kind": "pair", "path": str(path)})  # type: ignore[union-attr]
    for line in LINES:
        rec = records / f"{line}_xc_prep.json"
        if rec.is_file():
            man["records"][line] = str(rec)  # type: ignore[index]
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    return man


def main() -> None:
    readouts, records, out = (Path(x) for x in sys.argv[1:4])
    man = build(readouts, records, out)
    sys.argv = [sys.argv[0], str(out / "manifest.json"), str(out)]
    from figure_rnapii import main as draw

    draw()
    print(f"manifest {out / 'manifest.json'}: {sum(len(v) for v in man.values() if isinstance(v, list))} sources")


if __name__ == "__main__":
    main()
