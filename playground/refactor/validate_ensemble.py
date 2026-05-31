"""Gate: ensemble-batched reconstruction == per-member standalone reconstruct.

`reconstruct_ensemble` runs the coarse spine per member, then batches *all*
members' IB chains into one DAG (so same-bucket IBs across members fill one
launch on GPU).  This must not change results: member m must be byte-identical to
a standalone `reconstruct(seed_offset = m * MEMBER_SEED_STRIDE)` — the coarse
spine + IB seeds match, and IB chains re-seed per node (independent of batching).

Checked here under `SerialExecutor` (byte-exact); the GPU batch-width speedup is
validated separately on the CUDA box.

    python -u playground/refactor/validate_ensemble.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")
from _common import load_region  # noqa: E402

from gnome3d.pipeline.executor import SerialExecutor  # noqa: E402
from gnome3d.reconstruct import (  # noqa: E402
    MEMBER_SEED_STRIDE,
    reconstruct,
    reconstruct_ensemble,
)


def _key(beads):
    return [(b.start, b.end, b.kind, float(b.x), float(b.y), float(b.z)) for b in beads]


def main() -> int:
    s, bed, data, _ = load_region()
    chrs = [bed.chr]
    n = 3

    ens = reconstruct_ensemble(s, data, chrs, bed, n=n, executor=SerialExecutor())

    ok = len(ens) == n
    print(f"  {'ok ' if ok else 'FAIL'} ensemble returned {len(ens)} members (want {n})")
    for m in range(n):
        ref = reconstruct(s, data, chrs, bed, executor=SerialExecutor(),
                          seed_offset=m * MEMBER_SEED_STRIDE)
        em, rm = ens[m].get(bed.chr, []), ref.get(bed.chr, [])
        eq = len(em) == len(rm) and _key(em) == _key(rm)
        ok &= eq
        print(f"  {'ok ' if eq else 'FAIL'} member {m}: {len(em)} beads == standalone ({len(rm)})")

    print("PASS (ensemble batching == per-member reconstruct, byte-exact)" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
