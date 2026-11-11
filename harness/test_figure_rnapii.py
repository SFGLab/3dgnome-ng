"""Unit checks for the RNAPII comparison figures, `playground/rnapii/figure_rnapii.py`.

    python harness/test_figure_rnapii.py

The figures read one long table of dataset, kind, arm, metric, value and structure count. The
checks build a small table by hand and read back what the drawing functions put on the page.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "playground" / "rnapii"))

import numpy as np  # noqa: E402
from figure_rnapii import (  # noqa: E402
    candidate_loci,
    collect,
    dataset_label,
    figure_grid,
    figure_main,
)

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}   {detail}")


def table() -> pd.DataFrame:
    rows = []
    for ds, kind in (("GM12878", "line"), ("HFFC6", "line"), ("HG00512", "person")):
        for arm, n in (("C", 10), ("R", 10), ("M", 7)):
            for metric, v in (
                ("scc", 0.3),
                ("pearson", 0.28),
                ("apa_vs_map", 0.9),
                ("partial", -0.15),
            ):
                rows.append(
                    {"dataset": ds, "kind": kind, "arm": arm, "metric": metric, "value": v, "n": n}
                )
    return pd.DataFrame(rows)


def texts(fig) -> list[str]:  # noqa: ANN001
    return [t.get_text() for ax in fig.axes for t in ax.texts] + [
        t.get_text() for ax in fig.axes for t in ax.get_yticklabels() + ax.get_xticklabels()
    ]


def test_label_bj() -> None:
    rec = {
        "HFFC6": {"rnapii_source": "RNAPII from BJ, ENCSR956IXE"},
        "GM12878": {"rnapii_source": "ENCODE"},
    }
    check("HFFC6 is labelled with BJ", "BJ" in dataset_label("HFFC6", rec))
    check("GM12878 keeps its name", dataset_label("GM12878", rec) == "GM12878")


def test_grid(tmp: Path) -> None:
    rec = {"HFFC6": {"rnapii_source": "RNAPII from BJ, ENCSR956IXE"}}
    fig = figure_grid(table(), rec, tmp / "grid.png")
    t = " ".join(texts(fig))
    check(
        "the grid names every dataset and metric",
        all(s in t for s in ("GM12878", "HG00512", "scc", "pearson", "apa_vs_map", "partial")),
    )
    check("the grid labels HFFC6 with BJ", "BJ" in t)
    check("an arm with seven structures is marked n=7", "n=7" in t)
    check("the grid is written", (tmp / "grid.png").is_file())


def test_main(tmp: Path) -> None:
    fig = figure_main(table(), {}, {}, tmp / "main.png")
    check(
        "the main figure is written with panels c to f",
        (tmp / "main.png").is_file() and len(fig.axes) >= 4,
    )


def test_collect(tmp: Path) -> None:
    (tmp / "b.json").write_text(
        json.dumps([{"arm": "GM12878_C_60mb", "n": 10, "scc": 0.36, "pearson": 0.3}])
    )
    (tmp / "e.csv").write_text("arm,metric,value,n\nC,partial,-0.13,10\n")
    man = {
        "battery": [
            {
                "dataset": "GM12878",
                "kind": "line",
                "path": str(tmp / "b.json"),
                "arms": {"GM12878_C_60mb": "C"},
            }
        ],
        "expression": [{"dataset": "GM12878", "kind": "line", "path": str(tmp / "e.csv")}],
    }
    df = collect(man)
    got = {(r.arm, r.metric): r.value for r in df.itertuples()}
    check(
        "collect reads battery JSON and expression CSV into the long table",
        got.get(("C", "scc")) == 0.36
        and got.get(("C", "partial")) == -0.13
        and set(df["n"]) == {10},
        str(got),
    )


def test_loci() -> None:
    n, b = 600, 10_000
    starts = np.arange(n) * b
    obs = np.ones((n, n))
    c_sim = np.ones((n, n))
    r_sim = np.ones((n, n))
    loops = [
        ("chr1", a * b, a * b + 100, "chr1", (a + 20) * b, (a + 20) * b + 100, 5)
        for a in (110, 130, 410)
    ]
    obs[130, 150] = obs[150, 130] = 6.0
    r_sim[130, 150] = r_sim[150, 130] = 6.0
    obs[410, 430] = obs[430, 410] = 6.0
    got = candidate_loci(obs, c_sim, r_sim, starts, b, loops, window_bp=1_000_000, top=3)
    check(
        "the window where the map shows the loop and R differs from C ranks first",
        got and got[0][0] <= 130 * b < got[0][1] and got[0][0] <= 150 * b < got[0][1],
        str(got[:2]),
    )


def main() -> int:
    test_label_bj()
    test_loci()
    with tempfile.TemporaryDirectory() as d:
        test_grid(Path(d))
        test_main(Path(d))
        test_collect(Path(d))
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
