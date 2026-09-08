"""Unit checks for thinning a contact map to a target depth.

    python harness/test_thinning.py

A deep map at 25 kb has every pixel present, tens of millions of rows per chromosome, and the
model holds singletons as a list. Dropping pixels under a count threshold removes the far
pairs first and steepens the decay. Random thinning of the counts keeps every separation's
expected share, so the decay the exponent is fitted on survives.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.polymer import fit_contact_exponent  # noqa: E402
from validation.studies.self_corr import thin_counts  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def rows_of(mat: np.ndarray, step: int) -> list[tuple[str, int, str, int, int]]:
    out = []
    n = mat.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            c = int(mat[i, j])
            if c > 0:
                out.append(("chr1", i * step, "chr1", j * step, c))
    return out


def main() -> int:
    print("thinning checks")
    rng = np.random.default_rng(1)
    n, step = 400, 25_000
    raw = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sep = (j - i) * step
            raw[i, j] = raw[j, i] = (
                rng.poisson(3000.0 * (sep / 50_000.0) ** -1.0) if sep <= 3_000_000 else 0
            )
    total = raw[np.triu_indices(n, 1)].sum()
    thinned = thin_counts(raw, target=total / 50.0, seed=0)
    kept = thinned[np.triu_indices(n, 1)].sum()
    check(
        "the total lands on the target",
        abs(kept / (total / 50.0) - 1.0) < 0.03,
        f"{kept / (total / 50.0):.3f}",
    )
    check("counts never grow", bool((thinned <= raw).all()))
    check("symmetric", bool(np.array_equal(thinned, thinned.T)))
    check(
        "deterministic in the seed",
        bool(np.array_equal(thinned, thin_counts(raw, target=total / 50.0, seed=0))),
    )
    sparse = thin_counts(raw, target=total / 500.0, seed=0)
    check(
        "far fewer rows once the far pairs fall under one contact",
        (sparse > 0).sum() < 0.5 * (raw > 0).sum(),
        f"{(sparse > 0).sum() / 2:.0f} of {(raw > 0).sum() / 2:.0f}",
    )
    full = fit_contact_exponent(rows_of(raw, step))
    thin = fit_contact_exponent(rows_of(thinned, step))
    check(
        "the decay survives the thinning",
        thin.ok and abs(thin.slope - full.slope) < 0.05,
        f"{full.slope:.3f} vs {thin.slope:.3f}",
    )
    same = thin_counts(raw, target=total * 2, seed=0)
    check("a target above the depth leaves the map alone", bool(np.array_equal(same, raw)))
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
