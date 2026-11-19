"""Unit checks that a batched smooth result does not depend on the composition of the batch.

    python tests/unit/test_batch_seeding.py

Every check compares launches of the same width. The width changes the float32 reduction
order, which `test_width_changes_only_the_last_bits` measures.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gnome3d.mc.jax.smooth import _chunk_plan, mc_smooth_jax_batch  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def settings(steps: int = 400) -> Settings:
    s = Settings()
    s.use_excluded_volume = True
    s.exclusion_apply_to_smooth = True
    s.exclusion_weight = 0.5
    s.exclusion_radius_smooth = 1.5
    s.exclusion_skip_neighbors = 1
    s.use_confinement = False
    s.spring_stretch = s.spring_squeeze = 0.1
    s.max_temp_smooth = 2.0
    s.mc_stop_steps_smooth = steps
    # The stop is on the accept count. A chain then plateaus after a known number of rounds.
    s.mc_stop_improvement_smooth = 0.995
    s.mc_stop_successes_smooth = 1
    s.mc_executor_jax_bucket_shapes = True
    return s


def block(n: int, seed: int, spread: float = 1.0) -> dict[str, object]:
    """One interaction block, a random walk chain with both ends fixed as anchors."""
    rng = np.random.default_rng(seed)
    pos = np.cumsum(rng.normal(0.0, spread, size=(n, 3)), axis=0).astype(np.float32)
    dtn = np.linalg.norm(np.diff(pos.astype(np.float64), axis=0), axis=1).astype(np.float32)
    fixed = np.zeros(n, dtype=bool)
    fixed[0] = fixed[-1] = True
    return {"pos": pos, "dtn": dtn, "fixed": fixed, "step_size": 0.5, "seed": seed}


def run(problems: list[dict[str, object]], s: Settings) -> list[np.ndarray]:
    return [p for _score, p in mc_smooth_jax_batch([dict(p) for p in problems], s)]


def test_slot_does_not_matter() -> None:
    s = settings()
    a, b = block(256, 11), block(256, 22)
    ab, ba = run([a, b], s), run([b, a], s)
    check(
        "a block anneals the same whichever slot it takes in the launch",
        np.array_equal(ab[0], ba[1]) and np.array_equal(ab[1], ba[0]),
        f"max drift {max(np.abs(ab[0] - ba[1]).max(), np.abs(ab[1] - ba[0]).max()):.3e}",
    )


def test_partner_does_not_matter() -> None:
    s = settings()
    a = block(256, 11)
    with_b = run([a, block(256, 22)], s)[0]
    with_c = run([a, block(256, 33)], s)[0]
    check(
        "a block anneals the same next to one partner as next to another",
        np.array_equal(with_b, with_c),
        f"max drift {np.abs(with_b - with_c).max():.3e}",
    )


def test_a_slower_partner_does_not_over_anneal_it() -> None:
    s = settings()
    a = block(256, 11)
    slow_a = run([a, block(1024, 98, spread=8.0)], s)[0]
    slow_b = run([a, block(1024, 99, spread=12.0)], s)[0]
    check(
        "a converged block is unchanged by how long its partner keeps annealing",
        np.array_equal(slow_a, slow_b),
        f"max drift {np.abs(slow_a - slow_b).max():.3e}",
    )


def test_padding_does_not_leak() -> None:
    s = settings()
    a = block(300, 11)
    small = run([a, block(400, 55)], s)[0]  # the 512 bucket
    large = run([a, block(3000, 77)], s)[0]  # the 4096 bucket
    check(
        "a block padded to a larger bead extent anneals the same",
        np.array_equal(small, large),
        f"max drift {np.abs(small - large).max():.3e}",
    )


def test_width_changes_only_the_last_bits() -> None:
    """The launch width changes the order of the float32 sums and nothing else. The first
    difference is one unit in the last place of the score, which the Monte Carlo then amplifies.
    """
    s = settings(steps=100)
    s.mc_stop_improvement_smooth = 0.0
    s.mc_stop_successes_smooth = 10**9  # one round, no early stop
    a = block(700, 103)
    solo = mc_smooth_jax_batch([dict(a)], s)[0][0]
    wide = mc_smooth_jax_batch([dict(a)] + [dict(block(n, 200 + n)) for n in (120, 300, 512)], s)
    rel = abs(solo - wide[0][0]) / max(abs(solo), 1e-30)
    check(
        "widening the launch moves the score by at most a float32 ulp",
        rel <= 2**-22,
        f"relative difference {rel:.2e}, ulp is {2**-23:.2e}",
    )


def test_packing_splits_only_for_memory() -> None:
    s = settings()
    check(
        "eight small blocks plan as a single launch",
        len(_chunk_plan([(256, 1, 1)] * 8, False, False, s)) == 1,
    )
    s.mc_executor_jax_batch_width_smooth = 3
    plan = _chunk_plan([(4096, 1, 1)] * 8, True, False, s)
    flat = sorted(i for c in plan for i in c)
    check(
        "a width cap splits the plan and loses no block",
        len(plan) > 1 and flat == list(range(8)) and all(len(c) <= 3 for c in plan),
        f"{len(plan)} launches, sizes {[len(c) for c in plan]}",
    )


def test_largest_first() -> None:
    s = settings()
    s.mc_executor_jax_batch_width_smooth = 2
    plan = _chunk_plan([(256, 1, 1), (4096, 1, 1), (512, 1, 1), (2048, 1, 1)], False, False, s)
    check("the biggest block opens the first launch", plan[0][0] == 1, f"plan {plan}")


def test_every_block_comes_back() -> None:
    s = settings()
    sizes = [120, 300, 700, 1500, 260, 900, 180]
    blocks = [block(n, 300 + i) for i, n in enumerate(sizes)]
    got = run(blocks, s)
    check(
        "results come back in input order, one per block",
        [p.shape[0] for p in got] == sizes,
        f"got {[p.shape[0] for p in got]}",
    )


def main() -> int:
    print("batched smooth grouping independence checks\n")
    test_slot_does_not_matter()
    test_partner_does_not_matter()
    test_a_slower_partner_does_not_over_anneal_it()
    test_padding_does_not_leak()
    test_width_changes_only_the_last_bits()
    test_packing_splits_only_for_memory()
    test_largest_first()
    test_every_block_comes_back()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
