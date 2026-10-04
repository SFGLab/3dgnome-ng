"""The smooth driver can hand each milestone's positions to a caller.

    python harness/test_smooth_trace.py

`mc_smooth_numba` takes an optional `on_round` callback. The outer loop calls it once after
every milestone with the chain's positions at that point, so a caller can record the annealing
as a trajectory. With no callback nothing changes, which the parity gate holds.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.mc import numba as mc_numba  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def chain(n: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    pos = np.cumsum(rng.normal(size=(n, 3)), axis=0).astype(np.float32)
    fixed = np.zeros(n, dtype=np.bool_)
    fixed[[0, n - 1]] = True
    return pos, fixed, np.ones(n - 1, dtype=np.float32)


def settings() -> Settings:
    s = Settings()
    s.use_excluded_volume = False
    s.use_confinement = False
    s.use_motif_orientation = False
    s.mc_smooth_chains = 1
    s.mc_stop_steps_smooth = 2000
    return s


def main() -> int:
    print("smooth trace checks\n")
    frames: list[np.ndarray] = []
    pos, fixed, dtn = chain()
    mc_numba.seed_numba(1)
    mc_numba.mc_smooth_numba(
        pos, dtn, fixed, 0.5, settings(), on_round=lambda p: frames.append(p.copy())
    )
    check("the callback is called at least once per run", len(frames) >= 1, f"{len(frames)} rounds")
    check("each frame holds every bead", all(f.shape == pos.shape for f in frames))
    check("the last frame is the returned chain", np.allclose(frames[-1], pos, atol=1e-6))
    check(
        "frames differ, so they are snapshots not one array",
        len(frames) < 2 or not np.array_equal(frames[0], frames[-1]),
    )
    pos2, fixed2, dtn2 = chain()
    mc_numba.seed_numba(1)
    mc_numba.mc_smooth_numba(pos2, dtn2, fixed2, 0.5, settings())
    check("the callback changes nothing about the result", np.array_equal(pos, pos2))
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
