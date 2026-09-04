"""Unit checks that the estimate stage's replicates are seeded from their own identity.

    python harness/test_estimate_seeds.py

The estimate stage expands each interaction block into a set of dry smooth replicates that
start from differently perturbed copies of the same chain. Those replicates go into the same
batched kernel as everything else in the group, and that kernel now seeds each chain from the
seed its problem carries. A replicate that carries no seed falls back to its slot in the launch,
which puts the grouping back into the draw for this stage alone.

So each replicate has to carry a seed of its own, distinct from its siblings, fixed by its
parent block and its replicate number, and unaffected by which other blocks share the batch.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.pipeline.ib.estimate_dist import _expand_replicates  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def problem(n: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    pos = np.cumsum(rng.normal(size=(n, 3)), axis=0).astype(np.float32)
    fixed = np.zeros(n, dtype=bool)
    fixed[0] = fixed[-1] = True
    s = Settings()
    s.subanchor_estimate_replicates = 2
    s.subanchor_estimate_steps = 2
    return {
        "pos": pos,
        "fixed": fixed,
        "dtn": np.linalg.norm(np.diff(pos, axis=0), axis=1).astype(np.float32),
        "step_size": 0.5,
        "settings": s,
        "seed": seed,
    }


def seeds_of(problems: list[dict[str, object]]) -> list[int]:
    expanded, _spans = _expand_replicates(problems, 4)
    return [int(e["seed"]) for e in expanded]


def test_every_replicate_carries_a_seed() -> None:
    got = seeds_of([problem(64, 7)])
    check("each replicate carries a seed", len(got) == 4 and all(isinstance(x, int) for x in got))
    check("sibling replicates get different seeds", len(set(got)) == 4, f"{got}")


def test_seeds_follow_the_parent_not_the_batch() -> None:
    """A block's replicate seeds are the same whoever else is in the batch."""
    alone = seeds_of([problem(64, 7)])
    first = seeds_of([problem(64, 7), problem(64, 9)])[:4]
    second = seeds_of([problem(64, 9), problem(64, 7)])[4:]
    check("a block's replicate seeds do not depend on its partners", alone == first, f"{alone}")
    check("nor on its position in the batch", alone == second, f"{second}")


def test_different_blocks_get_different_seeds() -> None:
    a, b = seeds_of([problem(64, 7)]), seeds_of([problem(64, 9)])
    check("two blocks draw disjoint replicate seeds", not set(a) & set(b), f"{a} vs {b}")


def main() -> int:
    print("estimate replicate seeding checks\n")
    test_every_replicate_carries_a_seed()
    test_seeds_follow_the_parent_not_the_batch()
    test_different_blocks_get_different_seeds()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
