"""Validate the vectorized `build_contact_heatmaps` == the original O(N^2) loops.

`coarse.build.build_contact_heatmaps` was rewritten from per-singleton + N^2
Python loops to numpy (searchsorted + ordered `np.add.at` + outer products).
This asserts byte/float-exact equality against a reference re-implementation of
the original loops, run over every real IB active-region of the bundled region
(both heatmaps: anchor + subanchor).

The interesting exactness point is the singleton accumulation: `np.add.at` is fed
the (si,ei)/(ei,si) entries interleaved per singleton so its ordered accumulation
reproduces the loop's float sum order exactly.

    python -u playground/refactor/validate_contact_heatmaps_vec.py
"""

from __future__ import annotations

import bisect
import sys

import numpy as np

sys.path.insert(0, ".")
from _common import load_region  # noqa: E402

from gnome3d.hierarchy import Level, set_level  # noqa: E402
from gnome3d.pipeline import coarse as cb  # noqa: E402
from gnome3d.pipeline.coarse.build import (  # noqa: E402
    build_contact_heatmaps,
    subanchor_counts_per_arc,
)


def _ref_build(state, active_region, chr_):
    """Reference: the pre-vectorization loop implementation."""
    clusters = state.clusters
    n_anchors = len(active_region)
    counts = subanchor_counts_per_arc(state, active_region)
    N = n_anchors + sum(counts)

    anchor_offsets = [0]
    for c in counts:
        anchor_offsets.append(anchor_offsets[-1] + 1 + c)

    anchor_lens, gap_lens = [], []
    region_start = clusters[active_region[0]].start
    region_end = clusters[active_region[-1]].end
    breaks = [region_start]
    anchor_lens.append(clusters[active_region[0]].end - clusters[active_region[0]].start)
    for i in range(n_anchors - 1):
        ca_end = clusters[active_region[i]].end
        cb_start = clusters[active_region[i + 1]].start
        gap = max(cb_start - ca_end, 0)
        gap_lens.append(gap)
        anchor_lens.append(clusters[active_region[i + 1]].end - clusters[active_region[i + 1]].start)
        c = counts[i]
        if c >= 1:
            breaks.append(ca_end)
            for j in range(1, c):
                breaks.append(ca_end + int(gap * j / c))
            breaks.append(cb_start)
        else:
            breaks.append((ca_end + cb_start) // 2)
    breaks.append(region_end)

    h_sub = np.zeros((N, N), dtype=np.float64)
    for c1, p1, c2, p2, sc in state.singletons:
        if c1 != chr_ or c2 != chr_:
            continue
        if p1 < region_start or p1 > region_end or p2 < region_start or p2 > region_end:
            continue
        si = bisect.bisect_right(breaks, p1) - 1
        ei = bisect.bisect_right(breaks, p2) - 1
        if si < 0 or ei < 0 or si >= N or ei >= N or si == ei:
            continue
        h_sub[si, ei] += sc
        h_sub[ei, si] += sc

    h_anchor = np.zeros((n_anchors, n_anchors), dtype=np.float64)
    for i in range(n_anchors):
        ai, al_i = anchor_offsets[i], max(anchor_lens[i], 1)
        for j in range(i + 1, n_anchors):
            aj, al_j = anchor_offsets[j], max(anchor_lens[j], 1)
            val = h_sub[ai, aj] / (al_i * al_j / 1e6)
            h_anchor[i, j] = h_anchor[j, i] = val

    bin_is_anchor = [False] * N
    bin_arc_idx = [-1] * N
    for k in range(n_anchors):
        bin_is_anchor[anchor_offsets[k]] = True
    for i, c in enumerate(counts):
        for j in range(c):
            bin_arc_idx[anchor_offsets[i] + 1 + j] = i

    avg_count = float(h_sub.mean())
    if avg_count > 1e-6:
        h_sub /= avg_count
        bin_sizes = np.empty(N, dtype=np.float64)
        for k in range(N):
            if bin_is_anchor[k]:
                bin_sizes[k] = max(anchor_lens[anchor_offsets.index(k)], 1) / 1000.0
            else:
                arc_i = bin_arc_idx[k]
                gl = gap_lens[arc_i] if 0 <= arc_i < len(gap_lens) else 1
                c = counts[arc_i] if 0 <= arc_i < len(counts) else 1
                bin_sizes[k] = max(gl / max(c, 1), 1) / 1000.0
        for i in range(N):
            for j in range(i + 1, N):
                denom = bin_sizes[i] * bin_sizes[j]
                if denom > 0.0:
                    h_sub[i, j] = h_sub[j, i] = h_sub[i, j] / denom
    return h_anchor, h_sub


def main() -> int:
    s, bed, data, _ = load_region()
    chr_ = bed.chr

    # Build over a wider region too (more IBs of varying anchor counts) — no MC,
    # just the hierarchy, so it's cheap.
    from gnome3d.data import ContactData
    from gnome3d.io import parse_region

    states = [cb.build_state(s, data, [chr_], bed)]
    wbed = parse_region("chr1:1000000-30000000")
    wdata = ContactData.from_files(s, [wbed.chr], wbed)
    states.append(cb.build_state(s, wdata, [wbed.chr], wbed))

    active_regions = []
    for state in states:
        seg_level = set_level(
            Level.SEGMENT - Level.CHROMOSOME, state.chr_root, state.clusters, state.chrs
        )
        for seg in seg_level.get(state.chrs[0], []):
            for ib in state.clusters[seg].children:
                ar = list(state.clusters[ib].children)
                if len(ar) > 1:
                    active_regions.append((state, ar))

    ok = True
    total = 0
    for state, ar in active_regions:
        a_new, s_new = build_contact_heatmaps(state, ar, state.chrs[0])
        a_ref, s_ref = _ref_build(state, ar, state.chrs[0])
        eq = np.array_equal(a_new, a_ref) and np.array_equal(s_new, s_ref)
        ok &= eq
        total += 1
        if not eq:
            worst = max(float(np.max(np.abs(a_new - a_ref))), float(np.max(np.abs(s_new - s_ref))))
            print(f"  FAIL IB (n_anchors={len(ar)}): max |diff| = {worst:.2e}")

    print(f"  {total} IB active-regions compared (anchor + subanchor heatmaps), exact equality")
    print("PASS (vectorized == loop, byte-exact)" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
