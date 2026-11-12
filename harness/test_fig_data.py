"""Unit checks for the readout tables of the redesigned RNAPII figures, `playground/rnapii/fig_data.py`.

    python harness/test_fig_data.py

Each check writes small readout files by hand in the layout the readouts use and reads them back.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "playground" / "rnapii"))

from fig_data import agreement, boot_draws, link_draws, trio_paired  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")


def _battery(path: Path, line: str, scc: dict[str, float]) -> None:
    rows = [{"arm": f"{line}_{a}_60mb", "n": 10, "scc": v} for a, v in scc.items()]
    path.write_text(json.dumps(rows))


def test_agreement_reads_lines(d: Path) -> None:
    _battery(d / "H1ESC_battery.json", "H1ESC", {"E": 0.1, "C": 0.3})
    (d / "H1ESC_microc.json").write_text(
        json.dumps(
            [
                {"arm": "map", "n": 0, "r_fine": 1.0},
                {"arm": "H1ESC_E_60mb", "n": 10, "r_fine": 0.11},
                {"arm": "H1ESC_C_60mb", "n": 10, "r_fine": 0.12},
            ]
        )
    )
    t = agreement(d)
    got = t.set_index(["line", "arm", "measure"]).value
    check(
        "agreement_reads_lines",
        len(t) == 4 and got[("H1ESC", "C", "hic")] == 0.3 and got[("H1ESC", "E", "microc")] == 0.11,
        str(t.to_dict("records")),
    )


def test_agreement_no_microc_line(d: Path) -> None:
    _battery(d / "GM12878_battery.json", "GM12878", {"E": 0.2, "C": 0.37})
    t = agreement(d)
    gm = t[t.line == "GM12878"]
    check("agreement_no_microc_line", set(gm.measure) == {"hic"} and len(gm) == 2)


def test_trio_relative_pileup(d: Path) -> None:
    (d / "GM19238_pileup.json").write_text(
        json.dumps(
            [
                {"arm": "map", "n_models": 0, "apa": 2.0},
                {"arm": "GM19238_C_60mb", "n_models": 10, "apa": 3.0},
                {"arm": "GM19238_R_60mb", "n_models": 10, "apa": 4.0},
            ]
        )
    )
    _battery(d / "GM19238_battery.json", "GM19238", {"C": 0.32, "R": 0.31})
    t = trio_paired(d).set_index(["person", "arm", "measure"]).value
    check(
        "trio_relative_pileup",
        t[("GM19238", "C", "loop_pileup")] == 1.5 and t[("GM19238", "R", "hic")] == 0.31,
    )


def _genes(path: Path, rng: np.random.Generator, n: int = 40) -> pd.DataFrame:
    total = rng.normal(size=n)
    g = pd.DataFrame(
        {
            "gene_id": [f"G{i}" for i in range(n)],
            "total": total,
            "log2fc": -0.5 * total + rng.normal(size=n),
            "cluster": np.repeat(np.arange(8), n // 8),
        }
    )
    g.loc[3, "log2fc"] = np.nan
    g.to_csv(path, sep="\t", index=False)
    return g


def test_boot_reproduces_point(d: Path) -> None:
    g = _genes(d / "genes.tsv", np.random.default_rng(0))
    ok = g.dropna()
    want = spearmanr(ok.total, ok.log2fc).statistic
    point, draws = boot_draws(d / "genes.tsv", n=200)
    check(
        "boot_reproduces_point",
        abs(point - want) < 1e-12 and len(draws) == 200 and np.isfinite(draws).all(),
        f"{point} {want}",
    )


def _e3d(root: Path, arm: str, r_total: float | None) -> None:
    e = root / f"e3d_{arm}"
    e.mkdir(parents=True, exist_ok=True)
    g = _genes(e / "genes_GM12878_to_H1ESC.tsv", np.random.default_rng(1))
    ok = g.dropna()
    r = spearmanr(ok.total, ok.log2fc).statistic if r_total is None else r_total
    pd.DataFrame([{"pair": "GM12878->H1ESC", "r_total": r}]).to_csv(
        e / "summary.tsv", sep="\t", index=False
    )


def test_link_draws_refuses_mismatch(d: Path) -> None:
    _e3d(d, "C", 0.9)
    try:
        link_draws(d, ("C",), n=50)
        check("link_draws_refuses_mismatch", False, "no error")
    except ValueError as err:
        check("link_draws_refuses_mismatch", "genes_GM12878_to_H1ESC.tsv" in str(err), str(err))


def test_link_draws_sign_flipped(d: Path) -> None:
    _e3d(d, "R", None)
    t = link_draws(d, ("R",), n=50, seed=3)
    _, draws = boot_draws(d / "e3d_R" / "genes_GM12878_to_H1ESC.tsv", n=50, seed=3)
    check(
        "link_draws_sign_flipped",
        set(t.pair) == {"GM12878 vs H1ESC"} and np.allclose(np.sort(t.value), np.sort(-draws)),
    )


def main() -> int:
    for fn in (
        test_agreement_reads_lines,
        test_agreement_no_microc_line,
        test_trio_relative_pileup,
        test_boot_reproduces_point,
        test_link_draws_refuses_mismatch,
        test_link_draws_sign_flipped,
    ):
        with tempfile.TemporaryDirectory() as d:
            if fn is test_agreement_no_microc_line:
                test_agreement_reads_lines(Path(d))
                PASS.pop()
            fn(Path(d))
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
