"""Contact maps of a window and the medoid conformation of a locus, panels b and h of
`design/2026-11-12-rnapii-figures.md`.

    python playground/rnapii/model_maps.py <LINE> <models_root> <mcool> <chr1:a-b> \
        <chr1:c-d> <out.npz>

The first region is the map window, the second the locus. Reads `<models_root>/<LINE>_<A>_60mb/*.cif`
for arms E, C and R. Saves the Hi-C map and each arm's summed simulated map at 25 kb, the
window's bin starts, and for C and R the locus beads of the medoid conformation and of every
conformation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from legacy.validation.metrics.hic import observed_hic, simulated_contacts  # noqa: E402
from validation_battery import CONTACT_BEADS, load  # noqa: E402

BINSIZE = 25_000


def _span(region: str) -> tuple[int, int]:
    a, b = region.split(":")[1].replace(",", "").split("-")
    return int(a), int(b)


def ensemble_map(
    cifs: list[Path], bin_starts: np.ndarray, min_cover: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Sum over conformations of the simulated map on `bin_starts`, at the battery's contact
    radius, and the mask of bins holding a bead in every conformation. Beads exist only where
    the models place them, so a bin without one has no simulated contacts rather than few.
    Raises ValueError when the beads cover less than `min_cover` of the window's bins."""
    total = np.zeros((len(bin_starts), len(bin_starts)))
    held = np.ones(len(bin_starts), bool)
    lo, hi = int(bin_starts[0]), int(bin_starts[-1]) + BINSIZE
    for c in cifs:
        p, _, mid, _ = load(c)
        r = CONTACT_BEADS * float(np.median(np.linalg.norm(np.diff(p, axis=0), axis=1)))
        inside = (mid >= lo) & (mid < hi)
        has = np.zeros(len(bin_starts), bool)
        has[np.unique((mid[inside] - lo) // BINSIZE)] = True
        held &= has
        covered = has.mean()
        if covered < min_cover:
            raise ValueError(f"{c}: beads cover {covered:.0%} of {lo}-{hi}")
        total += simulated_contacts(p[inside], mid[inside], bin_starts, BINSIZE, r)
    return total, held


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((a @ rot - b) ** 2).sum(1).mean()))


def medoid(coords: list[np.ndarray]) -> int:
    """Index of the conformation with the least summed RMSD to the others."""
    n = len(coords)
    cost = [sum(_kabsch_rmsd(coords[i], coords[j]) for j in range(n) if j != i) for i in range(n)]
    return int(np.argmin(cost))


def main() -> None:
    line, root, mcool, window, locus, out = sys.argv[1:7]
    hic, starts = observed_hic(mcool, window, BINSIZE, balance=True)
    save: dict[str, np.ndarray] = {"hic": hic, "bin_starts": starts}
    la, lb = _span(locus)
    for arm in ("E", "C", "R"):
        cifs = sorted((Path(root) / f"{line}_{arm}_60mb").glob("*.cif"))
        save[arm], save[f"{arm}_held"] = ensemble_map(cifs, starts)
        print(f"{line} {arm}: {save[f'{arm}_held'].mean():.0%} of bins hold beads")
        if arm == "E":
            continue
        loaded = [load(c) for c in cifs]
        sel = [(m >= la) & (m < lb) for _, _, m, _ in loaded]
        n = min(int(s.sum()) for s in sel)
        k = medoid([p[s][:n] for (p, _, _, _), s in zip(loaded, sel, strict=True)])
        p, st, m, anc = loaded[k]
        save[f"locus_{arm}_xyz"], save[f"locus_{arm}_mid"] = p[sel[k]], m[sel[k]]
        save[f"locus_{arm}_anchor"] = anc[sel[k]]
        for j in range(len(loaded)):
            save[f"ensemble_{arm}_{j}_xyz"] = loaded[j][0][sel[j]]
            save[f"ensemble_{arm}_{j}_mid"] = loaded[j][2][sel[j]]
        print(f"{line} {arm}: medoid {cifs[k].name}, {int(sel[k].sum())} locus beads")
    np.savez_compressed(out, **save)
    print(f"{out}: window {window}, {len(starts)} bins")


if __name__ == "__main__":
    main()
