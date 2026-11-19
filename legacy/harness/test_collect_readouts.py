"""Unit checks for turning a readouts directory into the figure's manifest,
`playground/rnapii/collect_readouts.py`.

    python legacy/harness/test_collect_readouts.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "playground" / "rnapii"))

from collect_readouts import build  # noqa: E402
from figure_rnapii import collect  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}   {detail}")


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        r = Path(d) / "readouts"
        (r / "trio").mkdir(parents=True)
        for line, scc in (("GM12878", 0.36), ("H1ESC", 0.30)):
            (r / f"{line}_battery.json").write_text(
                json.dumps(
                    [
                        {"arm": f"{line}_{a}_60mb", "n": 10, "scc": scc + k / 100, "pearson": 0.3}
                        for k, a in enumerate("ECRM")
                    ]
                )
            )
        (r / "HG00512_battery.json").write_text("[]")
        (r / "H1ESC_microc.json").write_text(
            json.dumps(
                [
                    {"arm": "map", "n": 0, "n_loops": 900, "apa_microc": 2.0, "r_fine": 1.0},
                    {
                        "arm": "H1ESC_R_60mb",
                        "n": 10,
                        "n_loops": 900,
                        "apa_microc": 1.5,
                        "r_fine": 0.4,
                    },
                ]
            )
        )
        (r / "trio" / "HG00512_battery.json").write_text(
            json.dumps(
                [
                    {"arm": "HG00512_C_60mb", "n": 10, "scc": 0.3},
                    {"arm": "HG00512_R_60mb", "n": 9, "scc": 0.28},
                ]
            )
        )
        for a in "ECRM":
            (r / f"e3d_{a}").mkdir()
            (r / f"e3d_{a}" / "summary.tsv").write_text(
                "pair\tn_with_deseq\tr_total\tr_conf\tjump_r_conf\nGM12878->H1ESC\t700\t-0.2\t-0.1\t-0.19\n"
            )
        (r / "e3d_E_test").mkdir()
        (r / "e3d_E_test" / "summary.tsv").write_text(
            "pair\tn_with_deseq\tr_total\nGM12878->H1ESC\t700\t-0.9\n"
        )
        records = Path(d) / "records"
        records.mkdir()
        for line in ("GM12878", "H1ESC", "HFFC6"):
            (records / f"{line}_xc_prep.json").write_text(
                json.dumps({"line": line, "rnapii_source": "ENCODE"})
            )
        man = build(r, records, Path(d) / "out")
        df = collect(man)
        got = {(x.dataset, x.arm, x.metric): x.value for x in df.itertuples()}
        check(
            "cell line battery rows land under their line and arm",
            got.get(("GM12878", "R", "scc")) == 0.38,
            str(len(got)),
        )
        check(
            "trio rows are persons", set(df.loc[df["dataset"] == "HG00512", "kind"]) == {"person"}
        )
        check(
            "cross line statistics land under their pair",
            got.get(("GM12878 vs H1ESC", "C", "r_conf")) == -0.1,
        )
        check(
            "a pair's SCC is the mean of its two lines'",
            abs(got.get(("GM12878 vs H1ESC", "R", "scc"), 0) - (0.38 + 0.32) / 2) < 1e-9,
            str(got.get(("GM12878 vs H1ESC", "R", "scc"))),
        )
        check(
            "Micro-C rows give the pile-up over the map's and the fine agreement",
            got.get(("H1ESC", "R", "apa_microc_vs_map")) == 0.75
            and got.get(("H1ESC", "R", "r_fine")) == 0.4,
            str(sorted(k for k in got if k[0] == "H1ESC")),
        )
        check(
            "a summary outside the four arms is ignored",
            set(df["arm"]) <= {"E", "C", "R", "M"},
            str(sorted(set(df["arm"]))),
        )
        check(
            "the manifest carries a record for every line",
            set(man["records"]) == {"GM12878", "H1ESC", "HFFC6"},
        )
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
