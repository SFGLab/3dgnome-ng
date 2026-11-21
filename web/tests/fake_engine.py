"""Stands in for `python -m gnome3d.cli` in tests: writes N straight chain cifs over the region,
or fails when the region starts with chr9. FAKE_ENGINE_SLEEP holds it for that many seconds and
FAKE_ENGINE_TRACE receives its start, end, CUDA_VISIBLE_DEVICES and the cores it may use."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gnome3d.io import write_cif  # noqa: E402
from gnome3d.types import BeadOut  # noqa: E402

args = sys.argv[1:]
region = args[args.index("--region") + 1]
members = args[args.index("--members") + 1]
out = Path(args[args.index("--out") + 1])
chrom, span = region.split(":")
trace = os.environ.get("FAKE_ENGINE_TRACE")
start = time.time()
time.sleep(float(os.environ.get("FAKE_ENGINE_SLEEP", "0")))
if chrom == "chr9":
    print("engine exploded", file=sys.stderr)
    sys.exit(3)
a, b = (int(x) for x in span.split("-"))
n = int(members.split("-")[1]) + 1
out.mkdir(parents=True, exist_ok=True)
for k in range(n):
    beads = [BeadOut(s, s + 2000, (s - a) / 2000.0 * (k + 1), 0.0, 0.0, "anchor" if s % 20000 == 0 else "subanchor") for s in range(a, b, 2000)]
    write_cif(str(out / f"{chrom}_{a}_{b}_s{k + 1}.cif"), beads)
if trace:
    with open(trace, "a") as fh:
        cores = ",".join(str(c) for c in sorted(os.sched_getaffinity(0))) if hasattr(os, "sched_getaffinity") else "-"
        fh.write(f"{start} {time.time()} {os.environ.get('CUDA_VISIBLE_DEVICES', '')} {cores}\n")
print(f"wrote {n} structures")
