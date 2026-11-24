"""Draws the picture of the contacts example of the title page as the contact map tab of a job draws it.

    python web/design/render_contacts.py [--log] CONTACTS.json OUT.png

CONTACTS.json is the answer of `/api/jobs/<key>/contacts` for the region of the example with the
halves the tab opens on, Hi-C above and the contacts of the structures within the median bond
below. A half arrives as its rows from the diagonal on. Each half has its own colour range, as
`heatmap.ts` and `mapDraw.ts` have it: linear from zero to the 99th percentile of its nonzero
values, or with `--log` on a log scale from the 5th percentile, the choice Log scale of the tab.
The picture of the title page is the log one, which shows the blocks at the size of a tile. The
main diagonal and a bin without contacts are drawn as no data. The bins without contacts at the two
ends of the region are left out, and the diagonal runs from corner to corner.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

STOPS = np.array([[255, 255, 255], [241, 120, 60], [110, 18, 8]], dtype=float)
NO_DATA = (226, 225, 221)
SIDE = 932


def ramp(v: np.ndarray, lo: float, hi: float, log: bool) -> np.ndarray:
    """Returns the colours of the values on the range from `lo` to `hi`, as `colour` of heatmap.ts."""
    floor = lo if lo > 0 else hi / 1000

    def f(x: np.ndarray | float) -> np.ndarray:
        return np.log(np.maximum(x, floor)) if log else np.asarray(x, dtype=float)

    span = float(f(hi) - f(lo))
    t = np.clip((f(v) - f(lo)) / span if span > 0 else np.zeros_like(v), 0, 1) * (len(STOPS) - 1)
    i = np.minimum(len(STOPS) - 2, np.floor(t).astype(int))
    u = (t - i)[..., None]
    return np.rint(STOPS[i] + u * (STOPS[i + 1] - STOPS[i])).astype(np.uint8)


def percentile(m: np.ndarray, p: float) -> float:
    """Returns a percentile of the nonzero values of a half, as `percentile` of heatmap.ts."""
    x = np.sort(m[m > 0])
    return float(x[min(len(x) - 1, int(p * len(x)))]) if len(x) else 1.0


def half_range(m: np.ndarray, log: bool) -> tuple[float, float]:
    """Returns the colour range of a half, as `halfRange` of heatmap.ts."""
    hi = percentile(m, 0.99)
    return (min(percentile(m, 0.05), hi / 10) if log else 0.0), hi


def square(once: list[list[float]]) -> np.ndarray:
    """Returns the symmetric map of the rows sent once, row i from bin i on."""
    n = len(once)
    m = np.zeros((n, n))
    for i, row in enumerate(once):
        m[i, i:] = row
        m[i:, i] = row
    return m


def main() -> None:
    log = sys.argv[1] == "--log"
    source, out = sys.argv[2:4] if log else sys.argv[1:3]
    table = json.loads(Path(source).read_text())
    upper, lower = square(table["upper"]), square(table["lower"])
    n = len(upper)
    off = ~np.eye(n, dtype=bool)
    above = np.triu(np.ones((n, n), dtype=bool), 1)
    up, low = half_range(upper, log), half_range(lower, log)
    img = np.where(above[..., None], ramp(upper, *up, log), ramp(lower, *low, log))
    empty_up, empty_low = ~((upper > 0) & off).any(axis=1), ~((lower > 0) & off).any(axis=1)
    none = np.where(above, empty_up[:, None] | empty_up[None, :], empty_low[:, None] | empty_low[None, :]) | ~off
    img[none] = NO_DATA
    live = np.flatnonzero(~(empty_up | empty_low))
    a, b = int(live[0]), int(live[-1]) + 1
    Image.fromarray(np.ascontiguousarray(img[a:b, a:b]), "RGB").resize((SIDE, SIDE), Image.NEAREST).save(out)
    print(f"wrote {out}: bins {a} to {b} of {n}, ranges {up[0]:.3g} to {up[1]:.3g} above and {low[0]:.3g} to {low[1]:.3g} below")


if __name__ == "__main__":
    main()
