"""Gate: ThreadedExecutor == SerialExecutor, byte-exact.

The `ThreadedExecutor` runs independent per-IB stage-nodes across a thread pool
(numba kernels are nogil + thread-local RNG; the per-IB Python noise re-seeds a
thread-local RNG).  Each node fully seeds its own RNG from `Seeded.seed`, so its
output depends only on (inputs, seed) — not on which thread runs it or when.  So
threaded reconstruction must be byte-identical to serial, and stable across reruns
(thread scheduling varies, results don't).  The coarse spine stays inline.

    python -u playground/refactor/validate_threaded.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")
from _common import load_region  # noqa: E402

from gnome3d.pipeline.executor import SerialExecutor, ThreadedExecutor  # noqa: E402
from gnome3d.reconstruct import reconstruct  # noqa: E402


def _key(beads):
    return [(b.start, b.end, b.kind, float(b.x), float(b.y), float(b.z)) for b in beads]


def main() -> int:
    s, bed, data, _ = load_region()
    chrs = [bed.chr]

    serial = reconstruct(s, data, chrs, bed, executor=SerialExecutor())[bed.chr]
    thr = reconstruct(s, data, chrs, bed, executor=ThreadedExecutor(max_workers=4))[bed.chr]
    thr2 = reconstruct(s, data, chrs, bed, executor=ThreadedExecutor(max_workers=8))[bed.chr]

    eq = len(serial) == len(thr) and _key(serial) == _key(thr)
    det = len(thr) == len(thr2) and _key(thr) == _key(thr2)
    ok = eq and det
    print(f"  {'ok ' if eq else 'FAIL'} threaded(4) == serial ({len(thr)} vs {len(serial)} beads)")
    print(f"  {'ok ' if det else 'FAIL'} threaded stable across workers=4 vs 8 (scheduling-independent)")
    print("PASS (ThreadedExecutor == SerialExecutor, byte-exact)" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
