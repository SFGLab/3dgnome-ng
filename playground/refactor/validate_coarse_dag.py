"""Gate: the unified coarse DAG dispatches + runs every branch, deterministically.

`reconstruct` is now the single self-expanding DAG (the coarse spine +
`build_coarse_dag`'s branch selection, fanning out into the per-IB chains).  This
exercises each coarse-dispatch branch — origin (single-chr, <=1 segment), walk
(`random_walk`), segment (multi-segment, heatmap MC) — and asserts each:
  * selects the expected branch,
  * produces the expected RNG-independent bead *count* (the layout), and
  * is deterministic (a rerun is byte-identical).

(The byte-exact equivalence of this unified DAG to the former imperative driver
was proved against that driver before it was removed — origin/walk/segment all
identical; this gate is the forward regression net.  The multi-chr `chr+segment`
branch needs a whole-genome fixture and is covered by the genome run.)

MC steps are capped (this checks dispatch/layout/determinism, not convergence) so
the big segment-branch region stays fast.

    python -u playground/refactor/validate_coarse_dag.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")
from _common import CONFIG, DATA_DIR  # noqa: E402

from gnome3d import log  # noqa: E402
from gnome3d.data import ContactData  # noqa: E402
from gnome3d.hierarchy import Level, set_level  # noqa: E402
from gnome3d.io import parse_region  # noqa: E402
from gnome3d.pipeline import coarse as cb  # noqa: E402
from gnome3d.reconstruct import reconstruct  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402


def _key(beads):
    return [(b.start, b.end, b.kind, float(b.x), float(b.y), float(b.z)) for b in beads]


def _branch(state) -> str:
    """Which coarse-dispatch branch this state takes (mirrors build_coarse_dag)."""
    lvl = set_level(Level.SEGMENT - Level.CHROMOSOME, state.chr_root, state.clusters, state.chrs)
    total_segs = sum(len(v) for v in lvl.values())
    if state.s.random_walk:
        return "walk"
    if len(state.chrs) == 1 and total_segs <= 1:
        return "origin"
    if len(state.chrs) > 1:
        return "chr+segment"
    return "segment"


def _settings(mutate=None) -> Settings:
    s = Settings()
    if not s.load_ini(CONFIG):
        raise SystemExit(f"cannot load {CONFIG} (run from repo root)")
    s.data_dir = DATA_DIR
    s.mc_backend = "numba"
    # Cap the smooth MC hard — this gate checks dispatch/layout/determinism, not
    # convergence, so the big segment-branch region stays fast.
    s.mc_stop_steps_smooth = 200
    if mutate is not None:
        mutate(s)
    return s


def _case(region_str: str, want_branch: str, want_beads: int, mutate=None) -> tuple[bool, str]:
    s = _settings(mutate)
    bed = parse_region(region_str)
    data = ContactData.from_files(s, [bed.chr], bed)

    branch = _branch(cb.build_state(s, data, [bed.chr], bed))
    out = reconstruct(s, data, [bed.chr], bed)[bed.chr]
    rerun = reconstruct(s, data, [bed.chr], bed)[bed.chr]

    branch_ok = branch == want_branch
    beads_ok = len(out) == want_beads
    det_ok = len(out) == len(rerun) and _key(out) == _key(rerun)
    ok = branch_ok and beads_ok and det_ok
    detail = f"branch={branch:11s} beads={len(out)} det={det_ok}"
    if not branch_ok:
        detail += f" (want branch {want_branch})"
    if not beads_ok:
        detail += f" (want {want_beads} beads)"
    return ok, detail


def main() -> int:
    log.setup(0)
    # (label, region, want_branch, want_beads, settings mutator)
    cases = [
        ("origin", "chr1:18288319-20307135", "origin", 1673, None),
        ("walk", "chr1:18288319-20307135", "walk", 1673, lambda s: setattr(s, "random_walk", True)),
        ("segment", "chr1:1000000-30000000", "segment", 21733, None),
    ]
    ok = True
    for label, region, branch, beads, mut in cases:
        c_ok, detail = _case(region, branch, beads, mut)
        ok &= c_ok
        print(f"  {'ok ' if c_ok else 'FAIL'} {label:8s} {detail}")

    print("PASS (all coarse branches dispatch + run + deterministic)" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
