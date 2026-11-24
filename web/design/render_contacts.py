"""Draws the picture of the contacts example of the title page as the contact map tab of a job draws it.

    python web/design/render_contacts.py CONTACTS.json OUT.png

CONTACTS.json is the answer of `/api/jobs/<key>/contacts` for the region of the example with the
halves the tab opens on, Hi-C above and the contacts of the structures within the median bond
below. A half arrives as its rows from the diagonal on. Each half has its own linear colour range, from zero to the 99th percentile of its nonzero
values, as `heatmap.ts` and `mapDraw.ts` have it. The main diagonal and a bin without contacts are
drawn as no data.
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


def ramp(v: np.ndarray, hi: float) -> np.ndarray:
    """Returns the colours of the values on the linear range from zero to `hi`."""
    t = np.clip(v / hi if hi > 0 else np.zeros_like(v), 0, 1) * (len(STOPS) - 1)
    i = np.minimum(len(STOPS) - 2, np.floor(t).astype(int))
    u = (t - i)[..., None]
    return np.rint(STOPS[i] + u * (STOPS[i + 1] - STOPS[i])).astype(np.uint8)


def top(m: np.ndarray) -> float:
    """Returns the 99th percentile of the nonzero values of a half, the top of its range."""
    x = np.sort(m[m > 0])
    return float(x[min(len(x) - 1, int(0.99 * len(x)))]) if len(x) else 1.0


def square(once: list[list[float]]) -> np.ndarray:
    """Returns the symmetric map of the rows sent once, row i from bin i on."""
    n = len(once)
    m = np.zeros((n, n))
    for i, row in enumerate(once):
        m[i, i:] = row
        m[i:, i] = row
    return m


def main() -> None:
    table = json.loads(Path(sys.argv[1]).read_text())
    upper, lower = square(table["upper"]), square(table["lower"])
    n = len(upper)
    off = ~np.eye(n, dtype=bool)
    above = np.triu(np.ones((n, n), dtype=bool), 1)
    img = np.where(above[..., None], ramp(upper, top(upper)), ramp(lower, top(lower)))
    empty_up, empty_low = ~((upper > 0) & off).any(axis=1), ~((lower > 0) & off).any(axis=1)
    none = np.where(above, empty_up[:, None] | empty_up[None, :], empty_low[:, None] | empty_low[None, :]) | ~off
    img[none] = NO_DATA
    Image.fromarray(img, "RGB").resize((SIDE, SIDE), Image.NEAREST).save(sys.argv[2])
    print(f"wrote {sys.argv[2]}: {n} bins, ranges 0 to {top(upper):.3g} above and 0 to {top(lower):.3g} below")


if __name__ == "__main__":
    main()
