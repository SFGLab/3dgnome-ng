"""Record every stage of one region's reconstruction, for the model creation video.

    python playground/figures/trace_stages.py CONFIG.ini DATA_DIR REGION OUT.npz [--frames 60]

Runs the region's pipeline in process on the numba kernels and keeps, per interaction block:
the anchors where the block layout put them, the Hilbert curve the joint solve starts from,
the anchors it solved, the densified chain on its straight lines, the coil start the smooth
stage begins from, the smooth stage's own trajectory (one snapshot per milestone, thinned to
`--frames`), its final chain, and the chain after the boundary stitch and the cross block
relaxation. The smooth stage is re-run per block from the same seed the stage used, with the
per milestone callback, so the trajectory is the production stage's and ends where it ended.

Everything is saved in one npz with the beads' genomic starts, ends and anchor flags, which the
renderer turns into frames.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gnome3d.data import ContactData  # noqa: E402
from gnome3d.io import parse_chrs_arg  # noqa: E402
from gnome3d.mc import numba as mc_numba  # noqa: E402
from gnome3d.pipeline.coarse import build_state  # noqa: E402
from gnome3d.pipeline.coarse.stages import build_coarse_dag  # noqa: E402
from gnome3d.pipeline.ib.arcs import hilbert_start  # noqa: E402
from gnome3d.pipeline.ib.chain import ib_node_id  # noqa: E402
from gnome3d.pipeline.ib.smooth import SmoothStage, _start_positions  # noqa: E402
from gnome3d.pipeline.relax import relax_blocks  # noqa: E402
from gnome3d.pipeline.stage import StageKind  # noqa: E402
from gnome3d.pipeline.stitch import stitch_blocks  # noqa: E402
from gnome3d.reconstruct import _beads, pick_executor  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402
from gnome3d.types import BeadOut  # noqa: E402
from gnome3d.util import add_movable_noise_inplace, seed_rng  # noqa: E402


def _flag(name: str, default: int) -> int:
    if name not in sys.argv:
        return default
    k = sys.argv.index(name)
    v = int(sys.argv[k + 1])
    del sys.argv[k : k + 2]
    return v


def thin(frames: list[np.ndarray], n: int) -> np.ndarray:
    if len(frames) <= n:
        return np.stack(frames)
    idx = np.unique(np.round(np.linspace(0, len(frames) - 1, n)).astype(int))
    return np.stack([frames[i] for i in idx])


def xyz(beads: list[BeadOut]) -> np.ndarray:
    return np.array([[b.x, b.y, b.z] for b in beads], dtype=np.float32)


def main() -> None:
    n_frames = _flag("--frames", 60)
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
    print(f"{len(ibs)} blocks on {chrom}", flush=True)

    # The Hilbert curve the joint solve starts every anchor of the chromosome on.
    arced = [outputs[ib_node_id(ib.ib_id, StageKind.ARCS)] for ib in ibs]
    mids_all = np.array([g[2] for a in arced for g in a.anchor_genomic], dtype=np.int64)
    seeds_all = np.concatenate([a.anchor_seed_pos for a in arced]).astype(np.float32)
    hilbert_all = hilbert_start(seeds_all, mids_all, s.polymer_law())
    cuts = np.cumsum([0] + [len(a.anchor_genomic) for a in arced])

    saved: dict[str, np.ndarray] = {}
    final_blocks: list[list[BeadOut]] = []
    for k, ib in enumerate(ibs):
        a = arced[k]
        d = outputs[ib_node_id(ib.ib_id, StageKind.DENSIFY)]
        sm = outputs[ib_node_id(ib.ib_id, StageKind.SMOOTH)]
        prob = SmoothStage().to_problem((sm,))
        # the stage's own run, with the callback
        pos = _start_positions(prob)
        seed = int(prob["seed"])
        seed_rng(seed)
        mc_numba.seed_numba(seed)
        pos_run = pos.copy()
        add_movable_noise_inplace(pos_run, prob["fixed"], float(prob["step_size"]))
        frames: list[np.ndarray] = [pos_run.copy()]
        mc_numba.mc_smooth_numba(
            pos_run,
            prob["dtn"],
            prob["fixed"],
            float(prob["step_size"]),
            prob["settings"],
            prob["char_orientations"],
            prob["anchor_neighbors"],
            prob["anchor_neighbor_weights"],
            prob["heat_dist"],
            prob["compartment"],
            on_round=lambda p: frames.append(p.astype(np.float32)),
        )
        same = np.allclose(pos_run, sm.final_pos, atol=1e-4)
        print(
            f"  block {k}: {len(a.anchor_genomic)} anchors, {d.pos.shape[0]} beads,"
            f" {len(frames) - 1} milestones, reproduces the stage: {same}",
            flush=True,
        )
        saved[f"b{k}_layout"] = a.anchor_seed_pos
        saved[f"b{k}_hilbert"] = hilbert_all[cuts[k] : cuts[k + 1]]
        saved[f"b{k}_anchors"] = a.anchor_pos
        saved[f"b{k}_line"] = d.pos
        saved[f"b{k}_coil"] = pos
        saved[f"b{k}_frames"] = thin(frames, n_frames)
        saved[f"b{k}_final"] = sm.final_pos
        saved[f"b{k}_start"] = np.array(d.bead_starts, dtype=np.int64)
        saved[f"b{k}_end"] = np.array(d.bead_ends, dtype=np.int64)
        saved[f"b{k}_fixed"] = d.fixed
        saved[f"b{k}_anchor_mid"] = np.array([g[2] for g in a.anchor_genomic], dtype=np.int64)
        final_blocks.append(_beads(sm))
    stitched = stitch_blocks(final_blocks, s)
    relaxed = relax_blocks(stitched, s)
    for k in range(len(ibs)):
        saved[f"b{k}_stitched"] = xyz(stitched[k])
        saved[f"b{k}_relaxed"] = xyz(relaxed[k])
    saved["n_blocks"] = np.array(len(ibs))
    np.savez_compressed(out, **saved)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
