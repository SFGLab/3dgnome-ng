"""Whether a pair class lacks contacts from its distance or from its spread.

    python playground/contact_spread.py HOLE_ARM ARM

Holes are taken from the first structure of HOLE_ARM, which should carry no hole beads. Over every
structure of ARM, bead pairs inside one block, split into anchor pairs and subanchor pairs, and inside one hole are binned by genomic separation.
Per class and band the root mean square distance in median bonds is printed, the share of pairs
within the battery's contact radius, and the share a Gaussian chain with the same mean square
distance would have, the regularised gamma function of order 3/2 at rc^2 / (2 sigma^2) with
sigma^2 a third of the mean square. A share well under the Gaussian one at the same mean square
means the conformations spread too little.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import gammainc

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "playground"))

from hole_readout import holes, read_spans  # noqa: E402
from law_spring_readout import nu_from_log, read_models  # noqa: E402
from validation_battery import CONTACT_BEADS  # noqa: E402

BANDS = ((25_000, 50_000), (50_000, 100_000), (100_000, 200_000), (200_000, 400_000), (400_000, 1_000_000))
MAX_PAIRS = 400_000


def owners(mids: np.ndarray, anchor: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per bead, the hole it lies in or -1, and a block label that changes at every hole."""
    hole_id = np.full(len(mids), -1)
    for k, (a, b) in enumerate(h):
        hole_id[(mids > a) & (mids < b)] = k
    edges = np.sort(h[:, 0]) if len(h) else np.zeros(0)
    block_id = np.searchsorted(edges, mids, side="left")
    return hole_id, block_id


def main() -> None:
    hole_arm, arm = Path(sys.argv[1]), Path(sys.argv[2])
    _, mids0, kinds0 = read_models(hole_arm)
    st, en = read_spans(hole_arm)
    h = holes(mids0, st, en, kinds0)
    structs, mids, kinds = read_models(arm)
    hole_id, block_id = owners(mids, kinds == "anchor", h)
    rng = np.random.default_rng(0)
    n = len(mids)
    i = rng.integers(0, n, 4 * MAX_PAIRS)
    j = rng.integers(0, n, 4 * MAX_PAIRS)
    sep = np.abs(mids[i] - mids[j])
    same_hole = (hole_id[i] >= 0) & (hole_id[i] == hole_id[j])
    same_block = (hole_id[i] < 0) & (hole_id[j] < 0) & (block_id[i] == block_id[j])
    print(f"{arm.name}: {len(structs)} structures, {n} beads, {len(h)} holes")
    print("  rms distance in bonds, contact share, Gaussian share at that rms")
    head = "  ".join(f"{lo // 1000}-{up // 1000} kb".rjust(22) for lo, up in BANDS)
    print(f"  {'':>12s}  {head}")
    anc = kinds == "anchor"
    classes = (
        ("one block", same_block),
        ("anchor pairs", same_block & anc[i] & anc[j]),
        ("subanchors", same_block & ~anc[i] & ~anc[j]),
        ("one hole", same_hole),
    )
    for name, cls in classes:
        cells = []
        for lo, up in BANDS:
            sel = cls & (sep >= lo) & (sep < up)
            ii, jj = i[sel][:MAX_PAIRS], j[sel][:MAX_PAIRS]
            if ii.size < 50:
                cells.append("-".rjust(22))
                continue
            d2, hit = [], []
            for p in structs:
                bond = float(np.median(np.linalg.norm(np.diff(p, axis=0), axis=1)))
                d = np.linalg.norm(p[ii] - p[jj], axis=1) / bond
                d2.append(d**2)
                hit.append(d < CONTACT_BEADS)
            msd = float(np.mean(np.concatenate(d2)))
            share = float(np.mean(np.concatenate(hit)))
            gauss = float(gammainc(1.5, CONTACT_BEADS**2 / (2.0 * msd / 3.0)))
            cells.append(f"{msd**0.5:6.1f} {share:7.4f} {gauss:7.4f}".rjust(22))
        print(f"  {name:>12s}  {'  '.join(cells)}")
    floor_line(structs, mids, anc, same_block, i, j, sep, arm)


def floor_line(
    structs: list[np.ndarray],
    mids: np.ndarray,
    anc: np.ndarray,
    same_block: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    sep: np.ndarray,
    arm: Path,
) -> None:
    """The 1 percent and median quantiles of anchor pair distance over the law, per band, the
    law read from the arm's log at one bead per target spacing of 1 kb."""
    logf = arm.with_suffix(".log")
    if not logf.exists():
        return
    nu = nu_from_log(logf)
    cells = []
    for lo, up in BANDS:
        sel = same_block & anc[i] & anc[j] & (sep >= lo) & (sep < up)
        ii, jj, ss = i[sel][:MAX_PAIRS], j[sel][:MAX_PAIRS], sep[sel][:MAX_PAIRS]
        if ii.size < 50:
            cells.append("-".rjust(22))
            continue
        law = np.maximum(1.0, (ss / 1000.0) ** nu)
        r = []
        for p in structs:
            bond = float(np.median(np.linalg.norm(np.diff(p, axis=0), axis=1)))
            r.append(np.linalg.norm(p[ii] - p[jj], axis=1) / bond / law)
        q = np.concatenate(r)
        cells.append(f"{np.quantile(q, 0.01):6.2f} {np.median(q):6.2f}".rjust(22))
    print(f"  {'anchor floor':>12s}  {'  '.join(cells)}   1% and median of distance over law, nu {nu:.3f}")


if __name__ == "__main__":
    main()
