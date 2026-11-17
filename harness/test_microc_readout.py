"""Unit checks for the Micro-C readout, `playground/rnapii/microc_readout.py`.

python harness/test_microc_readout.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "playground" / "rnapii"))

from microc_readout import fine_agreement, load_slice  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}   {detail}")


def decaying(n: int, rng: np.random.Generator) -> np.ndarray:
    i, j = np.indices((n, n))
    m = 100.0 / (1 + np.abs(i - j)) * rng.lognormal(0, 0.5, size=(n, n))
    return np.triu(m) + np.triu(m, 1).T


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.npz"
        np.savez(
            p,
            bin1=np.array([0, 0, 2]),
            bin2=np.array([0, 3, 2]),
            count=np.array([5.0, 2.0, 7.0]),
            start=1000,
            binsize=500,
            nbins=4,
        )
        m, starts, b = load_slice(p)
        check(
            "a slice loads symmetric with its grid",
            m[0, 3] == 2
            and m[3, 0] == 2
            and m[2, 2] == 7
            and starts.tolist() == [1000, 1500, 2000, 2500]
            and b == 500,
        )
    rng = np.random.default_rng(1)
    obs = decaying(800, rng)
    same = fine_agreement(obs, obs, window=400, lo=4, hi=200)
    noise = fine_agreement(obs, decaying(800, np.random.default_rng(2)), window=400, lo=4, hi=200)
    check("a model equal to the map agrees fully", abs(same - 1.0) < 1e-9, f"{same:.3f}")
    check(
        "an unrelated map with the same decay agrees about zero", abs(noise) < 0.1, f"{noise:.3f}"
    )
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
