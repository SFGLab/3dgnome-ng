"""Capture what the arcs stage's target matrix is built from, then exit hard.

The matrix alone cannot be re-derived under a different distance law, because a target does not
say which PET count and which span produced it. This captures the inputs instead, the anchor
midpoints and the arcs, so a matrix can be rebuilt under any law offline and solved in seconds.

    python playground/capture_arc_inputs.py <config.ini> <region> <n_blocks> <out.pkl>
"""

import os
import pickle
import sys
import threading
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from gnome3d.pipeline.coarse import build  # noqa: E402

CAP: list[dict[str, object]] = []
LOCK = threading.Lock()
TARGET = int(sys.argv[3])
OUT = sys.argv[4]
_real = build.calc_anchor_expected_distances


def spy(state, active_region, chr_, anchor_heatmap=None, *a, **k):  # type: ignore[no-untyped-def]
    mat = _real(state, active_region, chr_, anchor_heatmap, *a, **k)
    clusters = state.clusters
    mids = [int(clusters[ci].genomic_pos) for ci in active_region]
    cluster_to_active = {ci: ai for ai, ci in enumerate(active_region)}
    chr_arcs = state.arcs.get(chr_, [])
    arcs: list[tuple[int, int, int]] = []
    for ai, ci in enumerate(active_region):
        for arc_local in clusters[ci].arcs:
            if arc_local >= len(chr_arcs):
                continue
            arc = chr_arcs[arc_local]
            other = arc.end if arc.start == ci else arc.start
            if other < ci or other not in cluster_to_active:
                continue
            arcs.append((ai, cluster_to_active[other], int(arc.score)))
    with LOCK:
        CAP.append(
            {
                "mids": mids,
                "arcs": arcs,
                "mat": np.asarray(mat, np.float64).copy(),
                "heat": None if anchor_heatmap is None else np.asarray(anchor_heatmap).copy(),
            }
        )
        print(f"  captured block N={len(mids)} arcs={len(arcs)}  ({len(CAP)}/{TARGET})", flush=True)
        if len(CAP) >= TARGET:
            keep = sorted(CAP, key=lambda c: -len(c["mids"]))[:6]  # type: ignore[arg-type]
            with open(OUT, "wb") as fh:
                pickle.dump(keep, fh)
            print(f"saved {len(keep)}, N = {[len(c['mids']) for c in keep]}", flush=True)
            sys.stdout.flush()
            os._exit(0)
    return mat


build.calc_anchor_expected_distances = spy

from gnome3d.cli import main  # noqa: E402

sys.argv = [
    "cli",
    "--config",
    sys.argv[1],
    "--region",
    sys.argv[2],
    "--data-dir",
    "data/GM12878",
    "-n",
    "1",
    "--out",
    "/tmp",
]
main()
