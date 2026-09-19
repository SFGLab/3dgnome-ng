"""The acceptance profile of the smooth stage over its anneal, per block.

    python playground/accept_profile.py CONFIG.ini DATA_DIR REGION OUT.csv

Reconstructs the region through the arcs stage, then re-runs each block's smooth stage on the
numba kernel from the stage's own seed, as `figures/trace_stages.py` does, and captures the
kernel's per round line, accepted moves out of the round's proposals. Writes one row per block
and round with the score, the improvement ratio and the accepted count, and prints the share
of rounds by acceptance band. The JAX kernel's cost per step is flat, so the share of rounds is
the share of wall, and the bands say what a scheme that pays only where acceptance is rare,
such as speculative prefetching, can give.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnome3d import log  # noqa: E402
from gnome3d.data import ContactData  # noqa: E402
from gnome3d.io import parse_chrs_arg  # noqa: E402
from gnome3d.mc import numba as mc_numba  # noqa: E402
from gnome3d.pipeline.coarse import build_state  # noqa: E402
from gnome3d.pipeline.coarse.stages import build_coarse_dag  # noqa: E402
from gnome3d.pipeline.ib.chain import ib_node_id  # noqa: E402
from gnome3d.pipeline.ib.smooth import SmoothStage, _start_positions  # noqa: E402
from gnome3d.pipeline.stage import StageKind  # noqa: E402
from gnome3d.reconstruct import pick_executor  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402
from gnome3d.util import add_movable_noise_inplace, seed_rng  # noqa: E402


class RoundCatcher(logging.Handler):
    """Collects the outer loop's per round record: step, score, ratio, accepted, proposals."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.rows: list[tuple[int, float, float, int, int]] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.msg.startswith("step %7s"):
            step, score, ratio, n_ok, n_prop, _ = record.args  # type: ignore[misc]
            self.rows.append((int(str(step).replace(",", "")), float(score), float(ratio), int(n_ok), int(n_prop)))


def main() -> None:
    config, data_dir, region, out = sys.argv[1:5]
    s = Settings()
    s.load_ini(config)
    for k in ("mc_executor_smooth", "mc_executor_estimate_dist", "mc_executor_arcs"):
        setattr(s, k, "threaded")
    chrom = region.split(":")[0]
    chrs, reg = parse_chrs_arg(region)
    s.data_dir = data_dir
    data = ContactData.from_files(s, chrs, reg)
    state = build_state(s, data, chrs, reg)
    dag, ib_sink = build_coarse_dag(state, 0)
    outputs = pick_executor(s).run(dag)
    ibs = sorted(
        (ib for ib in ib_sink if ib.chr_ == chrom),
        key=lambda ib: outputs[ib_node_id(ib.ib_id, StageKind.ARCS)].anchor_genomic[0][0],
    )
    logger = log.get("mc.numba")
    logger.setLevel(logging.DEBUG)
    lines = ["block,beads,round,step,score,ratio,accepted,proposals"]
    bands = [(0.0, 0.001, "<0.1%"), (0.001, 0.01, "0.1-1%"), (0.01, 0.1, "1-10%"), (0.1, 1.01, ">10%")]
    tally = {b[2]: 0 for b in bands}
    total = 0
    for k, ib in enumerate(ibs):
        sm = outputs[ib_node_id(ib.ib_id, StageKind.SMOOTH)]
        prob = SmoothStage().to_problem((sm,))
        pos = _start_positions(prob)
        seed = int(prob["seed"])
        seed_rng(seed)
        mc_numba.seed_numba(seed)
        add_movable_noise_inplace(pos, prob["fixed"], float(prob["step_size"]))
        catcher = RoundCatcher()
        logger.addHandler(catcher)
        mc_numba.mc_smooth_numba(
            pos, prob["dtn"], prob["fixed"], float(prob["step_size"]), prob["settings"],
            prob["char_orientations"], prob["anchor_neighbors"], prob["anchor_neighbor_weights"],
            prob["heat_dist"], prob["compartment"],
        )  # fmt: skip
        logger.removeHandler(catcher)
        n = len(pos)
        acc = np.array([r[3] / r[4] for r in catcher.rows])
        for i, (step, score, ratio, n_ok, n_prop) in enumerate(catcher.rows):
            lines.append(f"{k},{n},{i},{step},{score:.4f},{ratio:.5f},{n_ok},{n_prop}")
        for lo, hi, name in bands:
            tally[name] += int(((acc >= lo) & (acc < hi)).sum())
        total += len(acc)
        first = " ".join(f"{a:.3f}" for a in acc[:5])
        last = " ".join(f"{a:.4f}" for a in acc[-5:])
        print(f"block {k}: {n} beads, {len(acc)} rounds, acceptance first {first} ... last {last}, median {np.median(acc):.4f}", flush=True)
    Path(out).write_text("\n".join(lines) + "\n")
    print(f"\n{total} rounds over {len(ibs)} blocks; share of rounds by acceptance band:")
    for _, _, name in bands:
        print(f"  {name:>7s}: {tally[name] / max(total, 1):5.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
