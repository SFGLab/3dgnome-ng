"""Write the drawings of the front page's examples from finished jobs.

    python web/scripts/make_previews.py <api base> <a bp> <b bp> <name>=<job url> ... > previews.json

For each job: the structure whose distance between the two positions is nearest the ensemble's
mean, its beads projected to a plane that holds the line between the two positions, and the
distance over the ensemble as the site's own distance route gives it. All drawings share one
scale, so a longer line between the positions is a longer distance. Standard library only.

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


def plane(points: list[V], a: V, b: V) -> tuple[V, V, V]:
    """Axes of the drawing. The first runs from a to b. The second is the direction across it
    along which the beads spread most, found by power iteration. The third is the depth."""
    e1 = unit(sub(b, a))
    across = [sub(p, tuple(dot(p, e1) * c for c in e1)) for p in points]  # type: ignore[arg-type]
    v: V = unit(cross(e1, (0.3, 0.5, 0.8)))
    for _ in range(60):
        w = [0.0, 0.0, 0.0]
        for p in across:
            k = dot(p, v)
            w[0] += k * p[0]
            w[1] += k * p[1]
            w[2] += k * p[2]
        v = unit((w[0], w[1], w[2]))
    return e1, v, unit(cross(e1, v))


def drawing(base: str, url: str, a_bp: int, b_bp: int) -> dict:
    d = get(f"{base}{url}/distance?a={a_bp}&b={b_bp}")
    k = min(range(d["n"]), key=lambda i: abs(d["values"][i] - d["mean"]))
    s = get(f"{base}{url}/structure/{k + 1}")
    beads = [(float(x), float(y), float(z)) for _, _, x, y, z, _ in s["beads"]]
    c = tuple(sum(p[i] for p in beads) / len(beads) for i in range(3))
    pts = [sub(p, c) for p in beads]  # type: ignore[arg-type]
    ia, ib = d["a"]["bead"], d["b"]["bead"]
    e1, e2, e3 = plane(pts, pts[ia], pts[ib])
    flat = [(dot(p, e1), dot(p, e2), dot(p, e3)) for p in pts]
    return {"flat": flat, "a": ia, "b": ib, "mean": round(d["mean"], 1), "sd": round(d["sd"], 1), "n": d["n"], "structure": k + 1, "here": round(d["values"][k], 1)}


def main() -> None:
    base, a_bp, b_bp = sys.argv[1].rstrip("/"), int(sys.argv[2]), int(sys.argv[3])
    jobs = dict(arg.split("=", 1) for arg in sys.argv[4:])
    raw = {name: drawing(base, url, a_bp, b_bp) for name, url in jobs.items()}
    # One scale for every drawing: the largest half extent maps to 480 of a 1000 wide box.
    reach = max(max(abs(v) for p in r["flat"] for v in p[:2]) for r in raw.values())
    out = {}
    for name, r in raw.items():
        depth = [p[2] for p in r["flat"]]
        lo, hi = min(depth), max(depth)
        out[name] = {
            # x, y in a box of 1000 by 1000 with the centre at 500, and the depth in three steps.
            "points": [[round(500 + 480 * p[0] / reach), round(500 - 480 * p[1] / reach), min(2, int(3 * (p[2] - lo) / ((hi - lo) or 1.0)))] for p in r["flat"]],
            **{k: r[k] for k in ("a", "b", "mean", "sd", "n", "structure", "here")},
        }
    json.dump(out, sys.stdout, separators=(",", ":"))


if __name__ == "__main__":
    main()
