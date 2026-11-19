"""Write the drawings of the front page's example from finished jobs.

    python web/scripts/make_previews.py <api base> <a bp> <b bp> <name>=<job url> ... > previews.json

For each job: the structure whose distance between the two positions is nearest the ensemble's
mean, turned so the line from the first position to the second runs left to right and both
positions are in front of the chain and the distance over the ensemble as the site's own
distance route gives it. The first position is the origin of every drawing and all drawings
share one scale. A longer line between the positions is a longer distance. Standard library
only.

    python web/scripts/make_previews.py http://127.0.0.1:8770/new/api 12063302 12449484 \\
        rnapii=/jobs/85c9b9a86ed1d864 ctcf=/jobs/f06f5487896aaca7
"""

from __future__ import annotations

import json
import math
import sys
import urllib.request

V = tuple[float, float, float]


def get(url: str) -> dict:
    return json.load(urllib.request.urlopen(url, timeout=120))


def sub(a: V, b: V) -> V:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a: V, b: V) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def unit(a: V) -> V:
    n = math.sqrt(dot(a, a)) or 1.0
    return (a[0] / n, a[1] / n, a[2] / n)


def cross(a: V, b: V) -> V:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


#: Drawing units per bead unit, the same for every drawing.
SCALE = 40.0
#: Depth steps of a drawing, the farthest first.
LAYERS = 6
#: A bead this close to a marked bead on the drawing and nearer the viewer hides it, in bead units.
HIDES = 1.6
#: Beads either side of a marked bead that belong to its mark.
REACH = 5


def view(points: list[V], ia: int, ib: int) -> tuple[V, V, V]:
    """Axes of the drawing: right, down and toward the viewer. The first runs from a to b. The
    turn about it is the one that leaves the fewest beads in front of the two marked beads and
    among equals the one that spreads the chain between them widest on the drawing."""
    e1 = unit(sub(points[ib], points[ia]))
    u = unit(cross(e1, (0.3, 0.5, 0.8)))
    v = cross(e1, u)
    between = points[min(ia, ib) : max(ia, ib) + 1]
    best: tuple[tuple[float, float], tuple[V, V, V]] | None = None
    for step in range(72):
        t = math.radians(5 * step)
        e2: V = tuple(math.cos(t) * u[k] + math.sin(t) * v[k] for k in range(3))  # type: ignore[assignment]
        e3 = cross(e2, e1)
        hidden = 0
        for mark in (ia, ib):
            m = points[mark]
            for i, p in enumerate(points):
                if abs(i - mark) <= REACH:
                    continue
                d = sub(p, m)
                if dot(d, e3) > 0.3 and math.hypot(dot(d, e1), dot(d, e2)) < HIDES:
                    hidden += 1
        mid = sum(dot(p, e2) for p in between) / len(between)
        spread = sum((dot(p, e2) - mid) ** 2 for p in between) / len(between)
        key = (float(hidden), -spread)
        if best is None or key < best[0]:
            best = (key, (e1, e2, e3))
    assert best is not None
    return best[1]


def drawing(base: str, url: str, a_bp: int, b_bp: int) -> dict:
    d = get(f"{base}{url}/distance?a={a_bp}&b={b_bp}")
    k = min(range(d["n"]), key=lambda i: abs(d["values"][i] - d["mean"]))
    s = get(f"{base}{url}/structure/{k + 1}")
    beads = [(float(x), float(y), float(z)) for _, _, x, y, z, _ in s["beads"]]
    ia, ib = d["a"]["bead"], d["b"]["bead"]
    pts = [sub(p, beads[ia]) for p in beads]
    e1, e2, e3 = view(pts, ia, ib)
    depth = [dot(p, e3) for p in pts]
    lo, hi = min(depth), max(depth)
    return {
        # x and y in drawing units from the first position and the depth step.
        "points": [[round(SCALE * dot(p, e1)), round(SCALE * dot(p, e2)), min(LAYERS - 1, int(LAYERS * (z - lo) / ((hi - lo) or 1.0)))] for p, z in zip(pts, depth, strict=True)],
        "a": ia,
        "b": ib,
        "mean": round(d["mean"], 1),
        "sd": round(d["sd"], 1),
        "n": d["n"],
        "structure": k + 1,
        "here": round(d["values"][k], 1),
    }


def main() -> None:
    base, a_bp, b_bp = sys.argv[1].rstrip("/"), int(sys.argv[2]), int(sys.argv[3])
    jobs = dict(arg.split("=", 1) for arg in sys.argv[4:])
    json.dump({name: drawing(base, url, a_bp, b_bp) for name, url in jobs.items()}, sys.stdout, separators=(",", ":"))


if __name__ == "__main__":
    main()
