"""Mock of the website's contact map with the coverage mask, beside the map as it is drawn now.

    python playground/coverage_mask_mock.py MCOOL REGION ARM_DIR OUT_PNG

Upper half balanced Hi-C, lower half the model's MultiMM map balanced, colours as
`web/frontend/src/job/heatmap.ts`. Now: bins with no bead grey in the model half only, and the
score over every bin. Proposed: bins with no bead hatched grey in both halves and marked on a
strip above the map, and the score over pairs of covered bins with the coverage beside it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "playground"))
_spec = importlib.util.spec_from_file_location("site_contacts", ROOT / "web/gnome3d_web/contacts.py")
assert _spec is not None and _spec.loader is not None
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)

from law_spring_readout import read_models  # noqa: E402

from legacy.validation.metrics.hic import observed_hic  # noqa: E402

RES = 25_000
NO_DATA = np.array([226, 225, 221]) / 255
HATCH = np.array([200, 198, 192]) / 255
STOPS = np.array([[255, 255, 255], [241, 120, 60], [110, 18, 8]]) / 255


def ramp(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((np.log10(np.maximum(v, 1e-12)) - lo) / max(hi - lo, 1e-9), 0, 1)
    a = np.where(t[..., None] < 0.5, STOPS[0], STOPS[1])
    b = np.where(t[..., None] < 0.5, STOPS[1], STOPS[2])
    f = np.where(t < 0.5, t * 2, (t - 0.5) * 2)[..., None]
    return a + (b - a) * f


def log_range(m: np.ndarray) -> tuple[float, float]:
    v = m[np.triu_indices(len(m), 1)]
    v = v[v > 0]
    return float(np.log10(np.percentile(v, 5))), float(np.log10(np.percentile(v, 99)))


def scc_masked(a: np.ndarray, b: np.ndarray, keep: np.ndarray, res: int, max_bp: int = 2_000_000) -> float:
    """`contacts.scc` over pairs whose two bins are both kept."""
    n = len(a)
    num = den = 0.0
    for k in range(1, min(n - 1, max_bp // res) + 1):
        i = np.arange(n - k)
        m = keep[i] & keep[i + k]
        x, y = a[i[m], i[m] + k], b[i[m], i[m] + k]
        if x.size < 2 or x.std() < 1e-12 or y.std() < 1e-12:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        w = x.size * float(x.std()) * float(y.std())
        num += w * r
        den += w
    return num / den


def image(hic: np.ndarray, model: np.ndarray, covered: np.ndarray, masked: bool) -> np.ndarray:
    n = len(hic)
    up, lo = ramp(hic, *log_range(hic)), ramp(model, *log_range(model))
    i, j = np.indices((n, n))
    img = np.where((j >= i)[..., None], up, lo)
    model_empty = ~covered[i] | ~covered[j]
    hic_empty = ~(hic.sum(0) > 0)
    img[(j < i) & model_empty] = NO_DATA
    img[(j >= i) & (hic_empty[i] | hic_empty[j])] = NO_DATA
    if masked:
        hatch = model_empty & (((i + j) // 3) % 3 == 0)
        img[model_empty] = NO_DATA
        img[hatch] = HATCH
    img[i == j] = NO_DATA
    return img


def main() -> None:
    mcool, region, arm, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
    start, end = (int(x) for x in region.split(":")[1].split("-"))
    hic_raw, bin_starts = observed_hic(mcool, region, RES, balance=False)
    lo = int(bin_starts[0])
    n = len(bin_starts)
    structs, mids, _ = read_models(arm)
    idx = (mids - lo) // RES
    covered = np.zeros(n, dtype=bool)
    covered[np.unique(idx[(idx >= 0) & (idx < n)])] = True
    model = cm.multimm_map(structs, mids, lo, lo + n * RES, RES)
    counted = cm.model_map(structs, mids, lo, lo + n * RES, RES, cm.median_bond(structs))
    hic_i, model_i = cm.ice(hic_raw), cm.ice(model)
    s_all = cm.scc(hic_raw, counted, RES)
    s_cov = scc_masked(hic_raw, counted, covered, RES)
    share = covered.mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 7.4), gridspec_kw={"wspace": 0.12})
    for ax, masked in zip(axes, (False, True), strict=True):
        ax.imshow(image(hic_i, model_i, covered, masked), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(n - 1, -n * 0.035, "Hi-C, balanced", ha="right", va="bottom", fontsize=9)
        ax.text(0, n * 1.012, "This job, 1/(d+1)³ between bins", ha="left", va="top", fontsize=9)
        if masked:
            for a, b in _runs(~covered):
                ax.add_patch(plt.Rectangle((a - 0.5, -n * 0.03), b - a, n * 0.018, color="#8a877f", clip_on=False))
            ax.text(0, -n * 0.035, "no loop data", ha="left", va="bottom", fontsize=9, color="#5c5a54")
            title = "Proposed"
            card = (
                f"Agreement {s_cov:.2f}  stratum adjusted correlation\n"
                f"Compared on {covered.sum()} of {n} bins ({100 * share:.0f}%). Hatched bins hold\n"
                "no bead: no loop reaches them, so the model has nothing there\n"
                "and they are left out of the score."
            )
        else:
            title = "Now"
            card = f"Agreement {s_all:.2f}  stratum adjusted correlation\nScored on every bin, beadless bins counted as no contact."
        ax.set_title(f"{title}, {region} at 25 kb", fontsize=11, loc="left", pad=22)
        ax.text(0, n * 1.06, card, ha="left", va="top", fontsize=9.5, family="monospace")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"{region}: {covered.sum()} of {n} bins covered, SCC every bin {s_all:.3f}, covered bins {s_cov:.3f} -> {out}")


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


if __name__ == "__main__":
    main()
