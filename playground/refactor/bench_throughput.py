"""Throughput harness for the task-DAG pipeline on real data (GPU or CPU).

Times a `reconstruct` run on a region and breaks it down into:
  * build    - `coarse.build_state` (hierarchy build) — the cheap preamble.
  * stages   - per kind (COARSE / ARCS / DENSIFY / HEAT_DIST / SMOOTH): wall,
               kernel *launches*, and total *chains* (= IBs x restarts/trials).
               The coarse spine now runs *inside* the executor as the COARSE
               kind; the launches-vs-chains ratio on the IB kinds is the
               GPU-saturation signal: few wide launches over many chains = the
               batching is doing its job.
  * assemble - bead collection.

`GNOME3D_MC_PROFILE` does NOT capture the pipeline (the stages bypass the mc
dispatch where that profiling lives), so this instruments the executor's runner
layer directly instead.

Usage (on the CUDA box):
    python playground/refactor/bench_throughput.py \
        --config kaustav_models/config.ini --region chr1:1-20000000 \
        --data-dir kaustav_models/

    # add --compare-numba to A/B the SerialExecutor (numba) on the same region
    # add --warmup to discard a first (JIT-compiling) run and report steady state
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict

sys.path.insert(0, ".")
from gnome3d import log  # noqa: E402
from gnome3d.data import ContactData  # noqa: E402
from gnome3d.io import parse_region  # noqa: E402
from gnome3d.pipeline import StageKind, registry  # noqa: E402
from gnome3d.pipeline import coarse as cb  # noqa: E402
from gnome3d.pipeline.coarse.stages import build_coarse_dag  # noqa: E402
from gnome3d.pipeline.executor import Executor, SerialExecutor  # noqa: E402
from gnome3d.pipeline.ib import ib_node_id  # noqa: E402
from gnome3d.reconstruct import _beads, pick_executor  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

_SMOOTH = StageKind.SMOOTH.value


def _instrument():
    """Wrap each kind's registered runners to accumulate (wall, launches, chains).
    Returns (stats, restore)."""
    stats: dict = defaultdict(lambda: {"wall": 0.0, "launches": 0, "chains": 0})
    saved: dict = {}

    def timed(kind, mode, fn):
        def wrapped(arg):
            chains = len(arg) if mode == "batch" else 1
            t = time.perf_counter()
            out = fn(arg)
            st = stats[(kind.value, mode)]
            st["wall"] += time.perf_counter() - t
            st["launches"] += 1
            st["chains"] += chains
            return out

        return wrapped

    for kind in StageKind:
        try:
            r = registry.runners_for(kind)
        except KeyError:
            continue
        saved[kind] = (r.serial, r.batch)
        if r.serial is not None:
            r.serial = timed(kind, "serial", r.serial)
        if r.batch is not None:
            r.batch = timed(kind, "batch", r.batch)

    def restore():
        for kind, (s, b) in saved.items():
            rr = registry.runners_for(kind)
            rr.serial, rr.batch = s, b

    return stats, restore


def _run_timed(settings, data, chrs, region, executor: Executor, seed_offset: int = 0):
    """reconstruct(), timed into coarse / executor / assemble, with per-kind stats."""
    stats, restore = _instrument()
    try:
        t0 = time.perf_counter()
        state = cb.build_state(settings, data, chrs, region)
        dag, ib_sink = build_coarse_dag(state, seed_offset)
        t_build = time.perf_counter() - t0

        t1 = time.perf_counter()
        outputs = executor.run(dag)  # coarse spine (COARSE kind) + per-IB chains
        t_exec = time.perf_counter() - t1

        t2 = time.perf_counter()
        beads = 0
        for ibs in ib_sink:
            beads += len(_beads(outputs[ib_node_id(ibs.ib_id, _SMOOTH)]))
        t_assemble = time.perf_counter() - t2
    finally:
        restore()
    return {
        "n_ibs": len(ib_sink),
        "beads": beads,
        "build": t_build,
        "exec": t_exec,
        "assemble": t_assemble,
        "total": t_build + t_exec + t_assemble,
        "stats": dict(stats),
    }


def _report(label: str, r: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"  IBs: {r['n_ibs']}   beads: {r['beads']}")
    print(f"  build_state:        {r['build']:8.2f}s")
    print(f"  stages (executor):  {r['exec']:8.2f}s")
    for (kind, mode), st in sorted(r["stats"].items()):
        sat = st["chains"] / max(st["launches"], 1)
        print(
            f"     {kind:<10}[{mode}]  {st['wall']:7.2f}s  "
            f"{st['launches']:>4} launches  {st['chains']:>6} chains  ({sat:.0f} chains/launch)"
        )
    print(f"  assemble:           {r['assemble']:8.2f}s")
    print(f"  TOTAL:              {r['total']:8.2f}s   ({r['beads'] / max(r['total'], 1e-9):.0f} beads/s)")


def main() -> int:
    ap = argparse.ArgumentParser(description="pipeline throughput harness")
    ap.add_argument("--config", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--warmup", action="store_true", help="discard a first JIT-compiling run")
    ap.add_argument("--compare-numba", action="store_true", help="also time SerialExecutor (numba)")
    args = ap.parse_args()

    log.setup(0)
    s = Settings()
    if not s.load_ini(args.config):
        raise SystemExit(f"cannot load {args.config}")
    if args.data_dir:
        s.data_dir = args.data_dir
    bed = parse_region(args.region)
    chrs = [bed.chr] if bed else [args.region.strip()]
    data = ContactData.from_files(s, chrs, bed)

    executor = pick_executor(s)
    print(f"region {args.region}  backend={s.mc_backend}  executor={type(executor).__name__}")

    if args.warmup:
        print("warmup (compiling kernels)...")
        _run_timed(s, data, chrs, bed, executor)

    _report(f"{type(executor).__name__} ({s.mc_backend})", _run_timed(s, data, chrs, bed, executor))

    if args.compare_numba and not isinstance(executor, SerialExecutor):
        print("\ncomparing against SerialExecutor (numba)...")
        _report("SerialExecutor (numba)", _run_timed(s, data, chrs, bed, SerialExecutor()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
