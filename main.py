#!/usr/bin/env python3
"""
main.py - 3D structure prediction from contact data using 3dgnome data format.
    python main.py --config data/GM12878/config.ini \
                   --region chr1:18288319-20307135  \
                   --n 1                            \
                   --out out/
"""

import argparse
import os
import sys
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures.thread import _worker, _threads_queues
from pathlib import Path


class _DaemonThreadPoolExecutor(ThreadPoolExecutor):
    def _adjust_thread_count(self):
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = "%s_%d" % (self._thread_name_prefix or self, num_threads)
            t = threading.Thread(
                name=thread_name,
                target=_worker,
                args=(weakref.ref(self, weakref_cb),
                      self._work_queue,
                      self._initializer,
                      self._initargs),
                daemon=True,
            )
            t.start()
            self._threads.add(t)
            _threads_queues[t] = self._work_queue

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.settings import Settings
from src.io import parse_region, write_cif
from src.data import ContactData
from src.solver import Solver


def _run_structure(
    i: int,
    n: int,
    s: Settings,
    data: ContactData,
    chrs_list: list,
    region,
    out_dir: Path,
    entry_base: str,
) -> int:
    """Build and write one independent structure. Returns bead count."""
    print(f"\n[main] structure {i + 1}/{n}")
    solver = Solver(s)
    solver.load(data, chrs_list, region)
    solver.reconstruct_heatmap()
    solver.reconstruct_arcs()

    beads = solver.get_leaf_positions(chrs_list[0])
    if not beads:
        raise RuntimeError(f"Structure {i + 1}: no leaf beads returned")

    cif_path = out_dir / f"{entry_base}_s{i + 1}.cif"
    write_cif(str(cif_path), beads, entry_id=f"{entry_base}_s{i + 1}")
    print(f"[main] wrote {cif_path}  ({len(beads)} beads)")
    return len(beads)


def main():
    parser = argparse.ArgumentParser(description="3dgnome-ng structure prediction")
    parser.add_argument("--config", required=True, help="Path to config.ini")
    parser.add_argument("--region", required=True,
                        help="Genomic region, e.g. chr1:18288319-20307135")
    parser.add_argument("-n", "--n-structures", type=int, default=1,
                        help="Number of independent structures to generate (default 1)")
    parser.add_argument("--data-dir", default=None,
                        help="Override data_dir from config")
    parser.add_argument("--out", default=".", help="Output directory (default: .)")
    args = parser.parse_args()

    # 1. Parse region
    bed_region = parse_region(args.region)
    if bed_region is None:
        chrom = args.region.strip()
        if not chrom:
            sys.exit(f"Cannot parse region: {args.region!r}")
        chrs_list = [chrom]
        bed_region = None
    else:
        chrs_list = [bed_region.chr]

    # 2. Load settings
    s = Settings()
    if not s.load_ini(args.config):
        sys.exit(f"Failed to load config: {args.config!r}")
    if args.data_dir:
        s.data_dir = args.data_dir
    print(f"[main] config: {args.config}  data_dir: {s.data_dir}")

    # 3. Load all input data from files (shared read-only across workers)
    data = ContactData.from_files(s, chrs_list, bed_region)

    # 4. Prepare output
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    entry_base = args.region.replace(":", "_").replace("-", "_")

    # 5. Run n independent MC trajectories in parallel.
    #    Each worker creates its own Solver (separate cluster state).
    #    Numba releases the GIL during MC, so threads genuinely overlap.
    n = args.n_structures
    n_workers = min(n, os.cpu_count() or 1)
    print(f"[main] running {n} structure(s) with {n_workers} worker(s)")

    pool = _DaemonThreadPoolExecutor(max_workers=n_workers)
    futures = {
        pool.submit(_run_structure, i, n, s, data, chrs_list, bed_region, out_dir, entry_base): i
        for i in range(n)
    }
    try:
        for fut in as_completed(futures):
            fut.result()
    except KeyboardInterrupt:
        for f in futures:
            f.cancel()
        pool.shutdown(wait=False)
        raise
    else:
        pool.shutdown(wait=True)

    print(f"\n[main] done - {n} structure(s) written to {out_dir}/")


if __name__ == "__main__":
    main()
