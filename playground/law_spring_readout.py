"""Readout of the law spring trial: how gaps and anchored stretches sit against the law and
against Hi-C, per arm.

    python playground/law_spring_readout.py MCOOL REGION S0_BP ARM_DIR [ARM_DIR ...]

For each arm directory of cif files, with the run log at ARM_DIR.log, prints

  * realised over law distance for subanchor pairs inside one anchorless gap and for pairs with
    five or more anchors between them, at 25 to 50, 50 to 100 and 100 to 200 kb;
  * observed over expected of the model and of Hi-C at 25 kb for anchored and anchorless bin
    pairs, both maps balanced;
  * the stratum adjusted correlation of the MultiMM map against Hi-C up to 2 Mb, raw and
    balanced, beside the same score for a map with no structure, anchors per bin times
    anchors per bin over separation, so a score can be read against what coverage alone earns.

The law's exponent is read from the run log, the bead size is S0_BP. The maps and the score are
the website's, `web/gnome3d_web/contacts.py`.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("site_contacts", ROOT / "web/gnome3d_web/contacts.py")
assert _spec is not None and _spec.loader is not None
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)

from validation.metrics.hic import observed_hic  # noqa: E402

RES = 25_000
BANDS = ((25_000, 50_000), (50_000, 100_000), (100_000, 200_000))
OE_BANDS = ((1, 2), (2, 4), (4, 8), (8, 16))


def read_models(arm: Path) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Positions per structure, bead midpoints and bead kinds, beads in file order."""
    structs: list[np.ndarray] = []
    mids: list[int] = []
    kinds: list[str] = []
    for f in sorted(arm.glob("*.cif")):
        pos, mids, kinds = [], [], []
        for line in f.read_text().splitlines():
            if line.startswith("ATOM"):
                x = line.split()
                pos.append([float(x[10]), float(x[11]), float(x[12])])
                mids.append((int(x[16]) + int(x[17])) // 2)
                kinds.append(x[18])
        structs.append(np.array(pos))
    return structs, np.array(mids), np.array(kinds)


def nu_from_log(log: Path) -> float:
    m = re.search(r"polymer law: exponent ([0-9.]+)", log.read_text())
    if not m:
        raise SystemExit(f"{log} names no measured exponent")
    return float(m.group(1))


def gap_ratios(structs: list[np.ndarray], mids: np.ndarray, kinds: np.ndarray, nu: float, s0: int) -> list[str]:
    order = np.argsort(mids, kind="stable")
    mids, kinds = mids[order], kinds[order]
    structs = [p[order] for p in structs]
    gap = np.cumsum(kinds == "anchor")
    sub = kinds == "subanchor"
    out: dict[tuple[int, str], list[np.ndarray]] = {}
    n = len(mids)
    k = 1
    while k < n:
        i = np.arange(n - k)
        sep = mids[i + k] - mids[i]
        if sep.min() >= BANDS[-1][1]:
            break
        law = np.maximum(1.0, (np.maximum(sep, 1) / s0) ** nu)
        d = np.mean([np.linalg.norm(p[i + k] - p[i], axis=1) for p in structs], axis=0)
        r = d / law
        same = (gap[i] == gap[i + k]) & sub[i] & sub[i + k]
        many = (gap[i + k] - gap[i]) >= 5
        for lo, hi in BANDS:
            b = (sep >= lo) & (sep < hi)
            out.setdefault((lo, "same"), []).append(r[b & same])
            out.setdefault((lo, "many"), []).append(r[b & many])
        k += 1
    lines = []
    for lo, hi in BANDS:
        a = np.concatenate(out[(lo, "same")])
        b = np.concatenate(out[(lo, "many")])
        lines.append(
            f"    {lo // 1000}-{hi // 1000} kb: inside a gap {np.median(a) if a.size else float('nan'):.2f}"
            f" (n={a.size}), five or more anchors between {np.median(b) if b.size else float('nan'):.2f} (n={b.size})"
        )
    return lines


def oe(a: np.ndarray) -> np.ndarray:
    n = len(a)
    o = np.zeros_like(a)
    for k in range(1, n):
        dg = np.diagonal(a, k)
        mu = dg[dg > 0].mean() if (dg > 0).any() else 0.0
        i = np.arange(n - k)
        o[i, i + k] = dg / mu if mu > 0 else 0.0
    return o


def main() -> None:
    mcool, region, s0 = sys.argv[1], sys.argv[2], int(sys.argv[3])
    chrom, rest = region.split(":")
    start, end = (int(x) for x in rest.replace(",", "").split("-"))
    hic, bin_starts = observed_hic(mcool, region, RES, balance=False)
    lo_bin = int(bin_starts[0])
    n = len(bin_starts)
    hic_i = cm.ice(hic)
    sep = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) + 1.0
    print(f"{region} at {RES // 1000} kb, {n} bins, Hi-C {mcool}")
    for arm in (Path(a) for a in sys.argv[4:]):
        structs, mids, kinds = read_models(arm)
        nu = nu_from_log(Path(str(arm).rstrip("/") + ".log"))
        model = cm.multimm_map(structs, mids, lo_bin, lo_bin + n * RES, RES)
        model_i = cm.ice(model)
        bins = (mids[kinds == "anchor"] - lo_bin) // RES
        na = np.bincount(bins[(bins >= 0) & (bins < n)], minlength=n)[:n].astype(float)
        null = np.outer(na + 1, na + 1) / sep
        anchored = na > 0
        print(f"\n{arm.name}: {len(structs)} structures, {len(mids)} beads, nu {nu:.3f}, {int(anchored.sum())} of {n} bins hold an anchor")
        print(f"  SCC raw {cm.scc(hic, model, RES):.3f}  balanced {cm.scc(hic_i, model_i, RES):.3f}   no structure null raw {cm.scc(hic, null, RES):.3f}  balanced {cm.scc(hic_i, cm.ice(null), RES):.3f}")
        sel = np.ix_(anchored, anchored)
        print(f"  SCC balanced on anchored bins only {cm.scc(cm.ice(hic[sel]), cm.ice(model[sel]), RES):.3f}")
        print("  realised over law")
        for line in gap_ratios(structs, mids, kinds, nu, s0):
            print(line)
        ho, mo = oe(hic_i), oe(model_i)
        print("  observed over expected, median, Hi-C / model")
        for lo, hi in OE_BANDS:
            parts = []
            for name, ci, cj in (("anchored", anchored, anchored), ("anchorless", ~anchored, ~anchored)):
                mask = np.zeros((n, n), dtype=bool)
                for k in range(lo, hi):
                    i = np.arange(n - k)
                    m = ci[i] & cj[i + k]
                    mask[i[m], i[m] + k] = True
                parts.append(f"{name} {np.median(ho[mask]):.2f} / {np.median(mo[mask]):.2f}" if mask.any() else f"{name} none")
            print(f"    {lo * RES // 1000}-{hi * RES // 1000} kb  " + "   ".join(parts))
    del chrom, start, end


if __name__ == "__main__":
    main()
