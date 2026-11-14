"""Contacts of the bins between blocks split by the bins they pair with.

    python playground/hole_contact_split.py MCOOL REGION HOLE_ARM ARM [ARM ...]

Holes are taken from the first structure of HOLE_ARM, which should carry no hole beads, as
consecutive anchors more than 1.5 kb apart. A 25 kb bin whose centre lies in a hole is a hole
bin, every other mapped bin a block bin. For Hi-C, raw and balanced, and for each arm's contact
map as the battery builds it, the mean count of hole with hole and hole with block bin pairs is
printed over the mean of block with block pairs at the same separation, by band. The hole with
hole column is split into pairs inside one hole and pairs across two. Also printed is the battery's
SCC on raw counts and the share of its weight each pair class carries.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "playground"))

from hole_readout import holes, read_spans  # noqa: E402
from law_spring_readout import read_models  # noqa: E402
from validation_battery import CONTACT_BEADS  # noqa: E402

from validation.metrics.hic import observed_hic, simulated_contacts  # noqa: E402

RES = 25_000
BANDS = ((1, 2), (2, 4), (4, 8), (8, 16), (16, 40))


def contact_map(arm: Path, bin_starts: np.ndarray) -> np.ndarray:
    structs, mids, _ = read_models(arm)
    c = np.zeros((len(bin_starts), len(bin_starts)))
    for p in structs:
        r = CONTACT_BEADS * float(np.median(np.linalg.norm(np.diff(p, axis=0), axis=1)))
        c += simulated_contacts(p, mids, bin_starts, RES, r)
    return c / len(structs)


def ratios(m: np.ndarray, hole_id: np.ndarray, mapped: np.ndarray) -> list[tuple[float, ...]]:
    """Per band, same hole, two holes and hole with block, each over block with block."""
    n = len(hole_id)
    i, j = np.triu_indices(n, 1)
    ok = mapped[i] & mapped[j]
    hi, hj = hole_id[i] >= 0, hole_id[j] >= 0
    same = hi & hj & (hole_id[i] == hole_id[j])
    two = hi & hj & (hole_id[i] != hole_id[j])
    mix = hi ^ hj
    bb = ~hi & ~hj
    out = []
    for lo, up in BANDS:
        band = ok & (j - i >= lo) & (j - i < up)
        base = m[i[band & bb], j[band & bb]].mean()
        row = []
        for cls in (same, two, mix):
            sel = band & cls
            row.append(float(m[i[sel], j[sel]].mean() / base) if sel.any() and base > 0 else np.nan)
        out.append(tuple(row))
    return out


def scc_share(a: np.ndarray, b: np.ndarray, hole: np.ndarray) -> tuple[float, float, float]:
    """The battery's SCC, and the share of the stratum covariance carried by pairs touching a hole
    bin and by block pairs alone."""
    n = a.shape[0]
    num = den = 0.0
    cov_h = cov_b = 0.0
    for d in range(1, n):
        x, y = np.diagonal(a, d), np.diagonal(b, d)
        if x.size < 2 or x.std() < 1e-12 or y.std() < 1e-12:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        w = x.size * x.std() * y.std()
        num += w * r
        den += w
        t = hole[: n - d] | hole[d:]
        prod = (x - x.mean()) * (y - y.mean())
        cov_h += float(prod[t].sum())
        cov_b += float(prod[~t].sum())
    tot = cov_h + cov_b
    return num / den, cov_h / tot, cov_b / tot


def main() -> None:
    mcool, region, hole_arm = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    arms = [Path(a) for a in sys.argv[4:]]
    raw, bin_starts = observed_hic(mcool, region, RES, balance=False)
    bal, _ = observed_hic(mcool, region, RES, balance=True)
    mapped = bal.sum(0) > 0
    _, mids, kinds = read_models(hole_arm)
    st, en = read_spans(hole_arm)
    h = holes(mids, st, en, kinds)
    centre = bin_starts + RES // 2
    hole_id = np.full(len(bin_starts), -1)
    for k, (a, b) in enumerate(h):
        hole_id[(centre > a) & (centre < b)] = k
    hole = hole_id >= 0
    print(
        f"{region}: {len(h)} holes, {int((hole & mapped).sum())} mapped hole bins, "
        f"{int((~hole & mapped).sum())} mapped block bins, {int((hole & ~mapped).sum())} unmapped hole bins"
    )
    maps = {"Hi-C raw": raw, "Hi-C balanced": bal}
    for a in arms:
        maps[a.name] = contact_map(a, bin_starts)
    print("  mean count over block with block at the same separation; one hole / two holes / hole with block")
    head = "  ".join(f"{lo * 25}-{up * 25} kb".rjust(20) for lo, up in BANDS)
    print(f"  {'':>28s}  {head}")
    for name, m in maps.items():
        cells = "  ".join(
            " ".join(f"{v:5.2f}" if np.isfinite(v) else "    -" for v in row).rjust(20)
            for row in ratios(m, hole_id, mapped)
        )
        print(f"  {name:>28s}  {cells}")
    print("  SCC on raw counts, share of stratum covariance from pairs touching a hole bin / block pairs")
    for a in arms:
        s, ch, cb = scc_share(maps[a.name], raw, hole)
        print(f"  {a.name:>28s}  SCC {s:6.3f}   hole {ch:+.2f}  block {cb:+.2f}")


if __name__ == "__main__":
    main()
