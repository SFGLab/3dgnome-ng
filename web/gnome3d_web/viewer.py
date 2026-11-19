"""The drawing data of one structure: the chain as a smooth curve and a box around each
promoter and enhancer.

The beads are smoothed with `gnome3d.geometry.smooth_chain`. Every point of the curve has a
genomic position, linear between the midpoints of the two beads of its bond. The box of an
element is the padded axis aligned box of the curve points inside its interval.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gnome3d.geometry import smooth_chain

MAX_GAP = 5_000


def _beads(cif: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [ln.split() for ln in Path(cif).read_text().splitlines() if ln.startswith("ATOM")]
    pos = np.array([[float(r[10]), float(r[11]), float(r[12])] for r in rows])
    mid = np.array([(int(r[16]) + int(r[17])) // 2 for r in rows])
    o = np.argsort(mid, kind="stable")
    return pos[o], mid[o]


def viewer_payload(cif: Path, elements: list[dict[str, object]], per_bond: int = 4, pad: float = 0.15) -> dict[str, object]:
    """The smoothed chain and the element boxes of one structure, as JSON ready lists."""
    pos, mid = _beads(cif)
    curve = smooth_chain(pos, per_bond)
    if len(curve) == len(pos):
        bp = mid.astype(np.float64)
    else:
        t = np.arange(per_bond) / per_bond
        bp = np.concatenate([(mid[:-1, None] + t[None, :] * np.diff(mid)[:, None]).ravel(), [float(mid[-1])]])
    boxes = []
    for e in elements:
        a, b = int(e["start"]), int(e["end"])  # type: ignore[call-overload]
        idx = np.flatnonzero((bp >= a) & (bp <= b))
        if idx.size == 0:
            c = (a + b) / 2
            k = int(np.argmin(np.abs(bp - c)))
            if abs(bp[k] - c) > MAX_GAP:
                continue
            idx = np.arange(max(0, k - per_bond // 2), min(len(curve), k + per_bond // 2 + 1))
        pts = curve[idx]
        boxes.append(
            {
                "kind": e["kind"],
                "name": e.get("name", ""),
                "min": np.round(pts.min(axis=0) - pad, 3).tolist(),
                "max": np.round(pts.max(axis=0) + pad, 3).tolist(),
            }
        )
    return {"chain": np.round(curve, 3).tolist(), "boxes": boxes}
