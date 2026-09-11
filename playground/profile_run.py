"""Where a reconstruction's wall time goes, from its own log.

Sums the dispatch lines the executor writes, so the answer is the run's own accounting rather
than an estimate. Reports each stage's share, and for the batched stages the per launch detail:
how many regions, how wide, how many rounds they needed and how much of the batch was spent
waiting for its slowest chain.

    python playground/profile_run.py <run.log> [...]
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

DISPATCH = re.compile(r"(\w+)\[(\w+)\]: (\d+) nodes in ([\d.]+)s")
LAUNCH = re.compile(r"(\w+)\[(\w+)\]: (\d+) IBs in ([\d.]+)s - (\d+) rounds \((\d+) steps\)")
WASTED = re.compile(r"(\d+)% wasted")
SHAPE = re.compile(r"(\w+)\[(\w+)\]: (\d+) IBs x (\d+) beads")

for path in sys.argv[1:]:
    text = Path(path).read_text().splitlines()
    stage: dict[str, float] = defaultdict(float)
    count: dict[str, int] = defaultdict(int)
    launches: list[tuple[str, int, int, int, int, int]] = []
    pending_shape: dict[str, tuple[int, int]] = {}

    for line in text:
        m = SHAPE.search(line)
        if m:
            pending_shape[m.group(1)] = (int(m.group(3)), int(m.group(4)))
        m = LAUNCH.search(line)
        if m:
            name = m.group(1)
            ibs, beads = pending_shape.get(name, (int(m.group(3)), 0))
            w = WASTED.search(line)
            launches.append(
                (name, ibs, beads, int(m.group(5)), int(m.group(6)), int(w.group(1)) if w else -1)
            )
        m = DISPATCH.search(line)
        if m:
            key = f"{m.group(1)}[{m.group(2)}]"
            stage[key] += float(m.group(4))
            count[key] += 1

    total = sum(stage.values())
    print(f"\n=== {Path(path).name} ===")
    if not total:
        print("no dispatch lines")
        continue
    print(f"{'stage':>22s} {'seconds':>9s} {'share':>7s} {'dispatches':>11s}")
    for k in sorted(stage, key=lambda k: -stage[k]):
        print(f"{k:>22s} {stage[k]:>9.0f} {100 * stage[k] / total:>6.1f}% {count[k]:>11d}")
    print(f"{'total accounted':>22s} {total:>9.0f}")

    if launches:
        print(f"\n{'launch':>10s} {'IBs':>5s} {'beads':>7s} {'rounds':>7s} {'Msteps':>8s} {'wasted':>7s}")
        for name, ibs, beads, rounds, steps, wasted in sorted(launches, key=lambda x: -x[4])[:12]:
            w = f"{wasted}%" if wasted >= 0 else "-"
            print(f"{name:>10s} {ibs:>5d} {beads:>7d} {rounds:>7d} {steps / 1e6:>8.1f} {w:>7s}")
        by_stage: dict[str, list[int]] = defaultdict(list)
        for name, _ibs, _beads, _rounds, steps, _w in launches:
            by_stage[name].append(steps)
        print("\nsteps per stage, which is what a kernel optimisation divides:")
        for k, v in sorted(by_stage.items(), key=lambda kv: -sum(kv[1])):
            print(f"{k:>10s} {sum(v) / 1e6:>10.1f} Mstep over {len(v)} launches")
