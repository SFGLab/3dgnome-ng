"""Gate: the unified self-expanding DAG == the legacy reconstruct, byte-exact.

Step 3b moves the coarse half (hierarchy -> chr -> segment -> ib positioning)
out of a driver object and into pipeline stages, assembled by
`reconstruct.reconstruct_unified` into a single DAG whose IB-positioning node
`expand`s into the per-IB chains.  The coarse spine is a linear chain that
consumes the *same* global RNG stream in the *same* order as the old engine, and
the IB chains re-seed per `Seeded.seed`, so the two paths must produce identical
beads — same layout AND same coordinates (not just statistics).

This asserts full equality (start/end/kind/x/y/z per bead, per chromosome) so the
unified DAG can become the default in Step 4 with confidence.

    python playground/refactor/validate_coarse_dag.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")
from _common import load_region  # noqa: E402

from gnome3d.reconstruct import reconstruct, reconstruct_unified  # noqa: E402


def _key(beads):
    return [(b.start, b.end, b.kind, float(b.x), float(b.y), float(b.z)) for b in beads]


def main() -> int:
    s, bed, data, _ = load_region()

    old = reconstruct(s, data, [bed.chr], bed)
    new = reconstruct_unified(s, data, [bed.chr], bed)

    ok = True
    chrs_ok = set(old) == set(new)
    ok &= chrs_ok
    print(f"  {'ok ' if chrs_ok else 'FAIL'} same chromosomes ({sorted(old)} vs {sorted(new)})")

    total = 0
    for chr_ in sorted(old):
        ob, nb = old[chr_], new.get(chr_, [])
        n_ok = len(ob) == len(nb)
        eq = n_ok and _key(ob) == _key(nb)
        ok &= eq
        total += len(ob)
        flag = "ok " if eq else "FAIL"
        if not eq and n_ok:
            # find first differing bead for a useful message
            diffs = [i for i, (a, b) in enumerate(zip(_key(ob), _key(nb), strict=True)) if a != b]
            print(f"  {flag} {chr_}: {len(ob)} beads, {len(diffs)} differ (first @ {diffs[0]})")
        else:
            print(f"  {flag} {chr_}: {len(ob)} beads (new {len(nb)})")

    print(f"  total beads compared: {total}")
    print("PASS (unified DAG == legacy reconstruct, byte-exact)" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
