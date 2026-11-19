"""Region parsing and sampling shared by the validation studies."""

from __future__ import annotations

import collections
import random

from gnome3d.io import parse_region
from gnome3d.types import BedRegion


def parse_region_arg(region: str) -> tuple[list[str], BedRegion | None]:
    bed = parse_region(region)
    if bed is None:
        chrom = region.strip()
        if not chrom:
            raise ValueError(f"cannot parse region: {region!r}")
        return [chrom], None
    return [bed.chr], bed


_chrs_and_region = parse_region_arg


def enumerate_regions(
    breakpoints_path: str,
    n: int,
    chroms: list[str] | None = None,
    min_ibs: int = 2,
    max_ibs: int = 6,
    max_mb: float = 6.0,
    seed: int = 0,
) -> list[str]:
    """Sample n regions that each span multiple segments, roughly interaction blocks, with a varied
    IB count across chromosomes so inter-IB packing from excluded volume, confinement and IB-MC is
    actually exercised. A single-segment region is one IB and gives those features almost nothing to
    do.

    A region spans k consecutive breakpoint-segments where k approximates the IB count and lies in
    [min_ibs, max_ibs], capped at max_mb. Regions are stratified by k for an IB-count spread,
    shuffled within each class for a chromosome spread, and chosen greedily non-overlapping per
    chromosome. The result is deterministic given seed. Passing chroms=None uses every chromosome in
    the breakpoints file.
    """
    pts_by_chr: dict[str, list[int]] = collections.defaultdict(list)
    with open(breakpoints_path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                pts_by_chr[p[0]].append(int(p[1]))
    targets = chroms if chroms else sorted(pts_by_chr)
    by_k: dict[int, list[tuple[str, int, int, int]]] = collections.defaultdict(list)
    for c in targets:
        ps = sorted(pts_by_chr.get(c, []))
        for i in range(len(ps)):
            for k in range(min_ibs, max_ibs + 1):
                j = i + k
                if j >= len(ps):
                    break
                if (ps[j] - ps[i]) / 1e6 > max_mb:  # bigger k only grows the span so stop scanning
                    break
                by_k[k].append((c, ps[i], ps[j], k))
    if not by_k:
        return []
    rng = random.Random(seed)
    for v in by_k.values():
        rng.shuffle(v)  # chromosome spread within an IB-count class
    chosen: list[tuple[str, int, int, int]] = []
    used: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    order = sorted(by_k)
    pos = dict.fromkeys(order, 0)
    progressed = True
    while len(chosen) < n and progressed:
        progressed = False
        for k in order:  # round-robin across IB-count classes for an even IB-count spread
            while pos[k] < len(by_k[k]):
                c, a, b, kk = by_k[k][pos[k]]
                pos[k] += 1
                if not any(a < ub and lb < b for lb, ub in used[c]):  # non-overlapping per chrom
                    chosen.append((c, a, b, kk))
                    used[c].append((a, b))
                    progressed = True
                    break
            if len(chosen) >= n:
                break
    chosen.sort(key=lambda s: (s[0], s[1]))
    return [f"{c}:{a}-{b}" for c, a, b, _ in chosen]
