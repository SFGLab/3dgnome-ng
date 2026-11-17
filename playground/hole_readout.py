"""Readout of the stretches between interaction blocks that carry no beads.

    python playground/hole_readout.py MCOOL REGION COMPARTMENTS_BEDGRAPH ARM_DIR

Holes are found from the first structure of ARM_DIR as consecutive anchors more than 1.5 kb
apart, which inside a block would have subanchors between them. Prints the share of 25 kb bins
inside a hole, the compartment track's mean there and elsewhere, and the Pearson correlation of
the model's MultiMM map with balanced Hi-C over pairs up to 2 Mb, over every bin and with hole
bins left out. The map is the website's, `web/gnome3d_web/contacts.py`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "playground"))
_spec = importlib.util.spec_from_file_location("site_contacts", ROOT / "web/gnome3d_web/contacts.py")
assert _spec is not None and _spec.loader is not None
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)

from law_spring_readout import read_models  # noqa: E402

from validation.metrics.hic import observed_hic  # noqa: E402

RES = 25_000
MAX_SEP = 2_000_000 // RES


def holes(mids: np.ndarray, starts: np.ndarray, ends: np.ndarray, kinds: np.ndarray) -> np.ndarray:
    o = np.argsort(starts, kind="stable")
    s, e, k = starts[o], ends[o], kinds[o]
    gap = s[1:] - e[:-1]
    m = (k[:-1] == "anchor") & (k[1:] == "anchor") & (gap > 1500)
    return np.stack([e[:-1][m], s[1:][m]], 1)


def read_spans(arm: Path) -> tuple[np.ndarray, np.ndarray]:
    f = sorted(arm.glob("*.cif"))[0]
    st, en = [], []
    for line in f.read_text().splitlines():
        if line.startswith("ATOM"):
            x = line.split()
            st.append(int(x[16]))
            en.append(int(x[17]))
    return np.array(st), np.array(en)


def pearson_band(a: np.ndarray, b: np.ndarray, keep: np.ndarray) -> float:
    n = len(a)
    i, j = np.triu_indices(n, 1)
    m = (j - i <= MAX_SEP) & keep[i] & keep[j]
    x, y = a[i[m], j[m]], b[i[m], j[m]]
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def main() -> None:
    mcool, region, comp_path, arm = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
    chrom = region.split(":")[0]
    hic, bin_starts = observed_hic(mcool, region, RES, balance=True)
    n = len(bin_starts)
    lo = int(bin_starts[0])
    structs, mids, kinds = read_models(arm)
    st, en = read_spans(arm)
    h = holes(mids, st, en, kinds)
    centre = bin_starts + RES // 2
    in_hole = np.zeros(n, dtype=bool)
    for a, b in h:
        in_hole |= (centre > a) & (centre < b)
    comp = np.full(n, np.nan)
    for line in open(comp_path):
        x = line.split()
        if x[0] != chrom:
            continue
        if x[3] in ("nan", "NA"):
            continue
        sel = (centre >= int(x[1])) & (centre < int(x[2]))
        comp[sel] = float(x[3])
    model = cm.multimm_map(structs, mids, lo, lo + n * RES, RES)
    mapped = np.isfinite(np.diag(hic)) | (np.nansum(hic, 0) > 0)
    print(f"{region} {arm.name}: {len(h)} holes, {in_hole.sum()} of {n} bins inside one")
    print(f"  compartment mean in holes {np.nanmean(comp[in_hole]):+.3f}, elsewhere {np.nanmean(comp[~in_hole]):+.3f}")
    for name, m in (("holes", in_hole), ("elsewhere", ~in_hole)):
        v = comp[m & np.isfinite(comp)]
        print(f"  {name}: {v.size} bins with a compartment value, share positive {np.mean(v > 0) if v.size else float('nan'):.2f}")
    print(f"  Pearson up to 2 Mb, balanced Hi-C: every bin {pearson_band(hic, model, mapped):.3f}, holes left out {pearson_band(hic, model, mapped & ~in_hole):.3f}")


if __name__ == "__main__":
    main()
