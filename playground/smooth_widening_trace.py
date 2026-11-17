"""What the smooth stage does to a gap between its start and its end.

Runs one reconstruction in process with the smooth runners wrapped, so every block's start, its
start after the restart noise and its final chain are kept, and reports per arm how far pairs
inside one gap sit from the law at each of the three states. Also the bonds over their targets,
the mean bend, which the angle term scores as its cube, and the share of non neighbour pairs
under the excluded volume radius, which the hard wall may not deepen.

    python playground/smooth_widening_trace.py BASE.ini DATA_DIR REGION OUT_DIR [section.key=value ...]

The overrides are written into a copy of BASE.ini. The kept states go to OUT_DIR/trace.npz.
"""

from __future__ import annotations

import configparser
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.pipeline.ib import maxent, smooth
from gnome3d.pipeline.registry import register, runners_for
from gnome3d.pipeline.stage import StageKind
from gnome3d.util import add_movable_noise_inplace, seed_rng

BANDS = ((1_000, 5_000), (5_000, 25_000), (25_000, 50_000), (50_000, 100_000), (100_000, 200_000))
KEPT: list[dict[str, Any]] = []


def _keep(prob: dict[str, Any], final: np.ndarray) -> None:
    start = smooth._start_positions(prob)
    noised = start.copy()
    seed_rng(int(prob["seed"]))
    add_movable_noise_inplace(noised, prob["fixed"], float(prob["step_size"]))
    s = prob["settings"]
    r0 = float(s.exclusion_radius_smooth)
    if r0 <= 0.0:
        r0 = float(s.exclusion_auto_factor_smooth) * float(np.mean(prob["dtn"]))
    KEPT.append(
        {
            "start": np.asarray(start, dtype=np.float64),
            "noised": np.asarray(noised, dtype=np.float64),
            "final": np.asarray(final, dtype=np.float64),
            "fixed": np.asarray(prob["fixed"], dtype=bool),
            "dtn": np.asarray(prob["dtn"], dtype=np.float64),
            "starts": np.asarray(prob["bead_starts"], dtype=np.int64),
            "ends": np.asarray(prob["bead_ends"], dtype=np.int64),
            "r0": r0,
            "law": s.polymer_law(),
        }
    )


def _wrap() -> None:
    run = runners_for(StageKind.SMOOTH)
    serial, batch = run.serial, run.batch

    def serial_kept(prob: dict[str, Any]) -> Any:
        assert serial is not None
        res = serial(prob)
        _keep(prob, res[1])
        return res

    def batch_kept(probs: list[dict[str, Any]]) -> Any:
        assert batch is not None
        res = batch(probs)
        for p, r in zip(probs, res, strict=True):
            _keep(p, r[1])
        return res

    register(StageKind.SMOOTH, serial=serial_kept, batch=batch_kept)


def gap_ratios(pos: np.ndarray, blk: dict[str, Any]) -> dict[int, list[float]]:
    """Realised over law for bead pairs inside one gap, anchors excluded, by band."""
    fixed = blk["fixed"]
    mids = (blk["starts"] + blk["ends"]) / 2.0
    anchors = np.flatnonzero(fixed)
    out: dict[int, list[float]] = {k: [] for k in range(len(BANDS))}
    for a, b in zip(anchors[:-1], anchors[1:], strict=True):
        if b - a < 3:
            continue
        idx = np.arange(a + 1, b)
        m = mids[idx]
        law = maxent.law_distances(m, blk["law"])
        p = pos[idx]
        d = np.linalg.norm(p[:, None] - p[None, :], axis=2)
        sep = np.abs(m[:, None] - m[None, :])
        iu = np.triu_indices(len(idx), 1)
        for k, (lo, hi) in enumerate(BANDS):
            sel = (sep[iu] >= lo) & (sep[iu] < hi)
            out[k].extend((d[iu][sel] / law[iu][sel]).tolist())
    return out


def chain_stats(pos: np.ndarray, blk: dict[str, Any]) -> tuple[float, float, float]:
    """Median bond over target, mean bend as the angle term reads it, share of pairs closer than
    r0 among non neighbours, per thousand beads."""
    v = np.diff(pos, axis=0)
    n = np.linalg.norm(v, axis=1)
    bond = float(np.median(n / np.maximum(blk["dtn"][: len(n)], 1e-6)))
    c = np.einsum("ij,ij->i", v[:-1], v[1:]) / np.maximum(n[:-1] * n[1:], 1e-12)
    bend = float(np.mean(1.0 - (np.clip(c, -1, 1) + 1.0) * 0.5))
    close = 0
    from scipy.spatial import cKDTree

    pairs = cKDTree(pos).query_pairs(blk["r0"], output_type="ndarray")
    if len(pairs):
        close = int(np.sum(np.abs(pairs[:, 0] - pairs[:, 1]) > 1))
    return bond, bend, 1000.0 * close / len(pos)


def report() -> None:
    for state in ("start", "noised", "final"):
        pooled: dict[int, list[float]] = {k: [] for k in range(len(BANDS))}
        stats = []
        for blk in KEPT:
            for k, v in gap_ratios(blk[state], blk).items():
                pooled[k].extend(v)
            stats.append(chain_stats(blk[state], blk))
        bands = "  ".join(
            f"{lo // 1000}-{hi // 1000} kb {np.median(pooled[k]):.2f}"
            if pooled[k]
            else f"{lo // 1000}-{hi // 1000} kb -"
            for k, (lo, hi) in enumerate(BANDS)
        )
        st = np.array(stats)
        w = np.array([len(b["fixed"]) for b in KEPT], dtype=np.float64)
        bond, bend, close = (st * w[:, None]).sum(0) / w.sum()
        print(f"  {state:7s} gap over law  {bands}")
        print(f"          bond over target {bond:.2f}  mean bend {bend:.3f}  close pairs {close:.1f} per thousand")


def main() -> int:
    base, data_dir, region, out = sys.argv[1:5]
    over = sys.argv[5:]
    Path(out).mkdir(parents=True, exist_ok=True)
    c = configparser.ConfigParser()
    c.read(base)
    for o in over:
        key, val = o.split("=", 1)
        sec, k = key.split(".", 1)
        c[sec][k] = val
    ini = Path(out) / "trace.ini"
    with open(ini, "w") as fh:
        c.write(fh)

    _wrap()
    from gnome3d import cli

    sys.argv = ["gnome3d", "--config", str(ini), "--data-dir", data_dir, "--region", region,
                "-n", "1", "--out", str(Path(out) / "models")]
    cli.main()
    print(f"{out}: {len(KEPT)} blocks, {sum(len(b['fixed']) for b in KEPT)} beads, overrides {' '.join(over) or 'none'}")
    report()
    np.savez_compressed(
        Path(out) / "trace.npz",
        **{f"{k}_{i}": v for i, b in enumerate(KEPT) for k, v in b.items() if k != "law"},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
