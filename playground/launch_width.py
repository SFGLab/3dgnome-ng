"""How wide the smooth launches are, and what they would cost merged.

Cost per step in the JAX smooth kernel is nearly flat in both the launch width and the bead
count, so a launch of sixty four chains costs about what a launch of one costs. The only thing
that decides throughput is therefore how many interaction blocks are in a launch, and that is
set by the batch key rather than by any kernel.

Reads a run log and reports two things. The per launch cost broken down by width, which shows
where the time is going. And what each dispatch would cost if its groups ran as one launch,
which is the slowest chain's step count rather than the sum of every group's.

    python playground/launch_width.py <run.log> [...]
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

SHAPE = re.compile(r"smooth\[mc\]: (\d+) IBs x (\d+) beads")
DONE = re.compile(r"smooth\[mc\]: (\d+) IBs in ([\d.]+)s - (\d+) rounds \((\d+) steps\)")
GROUP = re.compile(r"smooth\[batch\]: (\d+) nodes \(group (\d+)/(\d+)")
ESTIMATE = re.compile(r"estimate_dist\[batch\]:")

# A merged launch pads to its largest member, and the biggest real launches cost 14.2 us per
# step at 32x16384 and 16.9 at 1x12800. Twenty is a conservative stand-in for that.
MERGED_US = 20e-6


def main(path: str) -> None:
    launches: list[tuple[int, int, int, float, int]] = []
    dispatches: list[list[tuple[int, float, int]]] = []
    current: list[tuple[int, float, int]] = []
    shape: tuple[int, int] | None = None
    in_estimate = False

    for line in Path(path).read_text().splitlines():
        if ESTIMATE.search(line):
            in_estimate = True
        m = GROUP.search(line)
        if m:
            in_estimate = False
            if m.group(2) == "1" and current:
                dispatches.append(current)
                current = []
        m = SHAPE.search(line)
        if m:
            shape = (int(m.group(1)), int(m.group(2)))
            continue
        m = DONE.search(line)
        if m and shape:
            k, secs, steps = int(m.group(1)), float(m.group(2)), int(m.group(4))
            beads = shape[1] if shape[0] == k else 0
            launches.append((k, beads, 0 if in_estimate else 1, secs, steps))
            if not in_estimate:
                current.append((k, secs, steps))
    if current:
        dispatches.append(current)

    if not launches:
        print(f"{Path(path).name}: no smooth launches")
        return
    total = sum(x[3] for x in launches)
    print(f"\n=== {Path(path).name}  {len(launches)} launches, {total:.0f}s ===\n")

    by: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for k, _b, is_smooth, secs, steps in launches:
        by[k][0] += secs
        by[k][1] += steps
        by[k][2] += is_smooth
    print(f"{'IBs':>5s} {'stage':>14s} {'seconds':>9s} {'share':>7s} {'us/step':>9s} {'per IB':>8s}")
    for k in sorted(by):
        secs, steps, n_smooth = by[k]
        stage = "smooth" if n_smooth else "estimate_dist"
        per = secs / steps * 1e6
        print(
            f"{k:>5d} {stage:>14s} {secs:>9.0f} {100 * secs / total:>6.1f}% "
            f"{per:>9.2f} {per / k:>8.2f}"
        )

    if not dispatches:
        return
    print(f"\n{'dispatch':>9s} {'groups':>7s} {'IBs':>5s} {'now':>9s} {'merged':>9s} {'gain':>6s}")
    now_all = merged_all = 0.0
    for i, d in enumerate(dispatches, 1):
        now = sum(x[1] for x in d)
        merged = max(x[2] for x in d) * MERGED_US
        now_all += now
        merged_all += merged
        print(
            f"{i:>9d} {len(d):>7d} {sum(x[0] for x in d):>5d} "
            f"{now:>8.1f}s {merged:>8.1f}s {now / merged:>5.1f}x"
        )
    print(f"{'total':>9s} {'':>7s} {'':>5s} {now_all:>8.0f}s {merged_all:>8.0f}s {now_all / merged_all:>5.1f}x")


for arg in sys.argv[1:]:
    main(arg)
