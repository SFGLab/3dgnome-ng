"""The components per locus of `[data] components` in the node solve of a uniform chain,
`design/uniform-chain.md`.

The track holds a few numbers per bin of a chromosome arm, gathered from the Hi-C map of the cell
line by `playground/component_track.py`. The affinity of two bins of one arm is
`sum over c of sign_c * x_c(i) * x_c(j)`, an estimate of the log of their observed over expected
count. A pair of nodes without a loop, without a cell and beyond `background_range_bp` is held at
the background times `exp(-affinity / 3)`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from gnome3d import log

if TYPE_CHECKING:
    from gnome3d.polymer import PolymerLaw
    from gnome3d.settings import Settings
    from gnome3d.types import F64Array, I64Array

LOG = log.get("components")


class ArmComponents(NamedTuple):
    """The components of the bins of one chromosome arm, sorted by start."""

    start: I64Array  # (B,)
    end: I64Array  # (B,)
    x: F64Array  # (B, K)
    sign: F64Array  # (K,)


_TRACKS: dict[str, dict[str, list[ArmComponents]]] = {}


def read_components(path: str) -> dict[str, list[ArmComponents]]:
    """Reads a component track, `CHROM START END ARM x_1 ... x_k` with a line
    `# sign CHROM ARM s_1 ... s_k` per arm. Returns the arms of each chromosome.
    """
    if path in _TRACKS:
        return _TRACKS[path]
    if not os.path.isfile(path):
        raise ValueError(
            f"[data] components {path!r} is not a file. Give the track of"
            " playground/component_track.py or leave the key empty."
        )
    signs: dict[tuple[str, int], list[float]] = {}
    rows: dict[tuple[str, int], list[tuple[int, int, list[float]]]] = {}
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 5 and p[0] == "#" and p[1] == "sign":
                signs[(p[2], int(p[3]))] = [float(v) for v in p[4:]]
            elif len(p) >= 5 and not line.startswith("#"):
                rows.setdefault((p[0], int(p[3])), []).append(
                    (int(p[1]), int(p[2]), [float(v) for v in p[4:]])
                )
    out: dict[str, list[ArmComponents]] = {}
    for (chrom, arm), items in sorted(rows.items()):
        items.sort(key=lambda r: r[0])
        x: F64Array = np.array([r[2] for r in items], dtype=np.float64)
        sign: F64Array = np.array(signs.get((chrom, arm), [1.0] * x.shape[1]), dtype=np.float64)
        if sign.shape[0] != x.shape[1]:
            raise ValueError(
                f"[data] components {path!r} has {x.shape[1]} values a bin and {sign.shape[0]}"
                f" signs for arm {arm} of {chrom}. Write the track again with"
                " playground/component_track.py."
            )
        out.setdefault(chrom, []).append(
            ArmComponents(
                np.array([r[0] for r in items], dtype=np.int64),
                np.array([r[1] for r in items], dtype=np.int64),
                x,
                sign,
            )
        )
    _TRACKS[path] = out
    return out


def node_affinity(arms: list[ArmComponents], mids: I64Array) -> tuple[F64Array, I64Array]:
    """Returns the affinity of every pair of nodes and the arm of each node, -1 for a node whose
    locus has no components. The affinity of two nodes of different arms, or with a node without
    components, is 0.
    """
    m = int(mids.shape[0])
    arm_of: I64Array = np.full(m, -1, dtype=np.int64)
    affinity: F64Array = np.zeros((m, m), dtype=np.float64)
    for a, arm in enumerate(arms):
        idx = np.searchsorted(arm.start, mids, side="right") - 1
        inside = (idx >= 0) & (mids < arm.end[np.clip(idx, 0, len(arm.end) - 1)])
        sel = np.flatnonzero(inside)
        if sel.size == 0:
            continue
        arm_of[sel] = a
        x = arm.x[idx[sel]]
        affinity[np.ix_(sel, sel)] = (x * arm.sign) @ x.T
    return affinity, arm_of


def hold_by_components(
    mat: F64Array,
    mids: I64Array,
    arms: list[ArmComponents],
    law: PolymerLaw,
    s: Settings,
) -> F64Array:
    """Returns the target matrix with the pairs that the components of their two loci hold.

    A pair that holds -0.5, which is a pair without a loop, without a cell and beyond
    `background_range_bp`, with both nodes in one arm of the track, is held at the background
    times `exp(-affinity / 3)`, clipped to one bead and to `distance_heatmap_stretching`
    backgrounds. It is entered as minus its distance, which the kernels score with the background
    spring. Every other pair is left as it is.
    """
    affinity, arm_of = node_affinity(arms, mids)
    pos = mids.astype(np.float64)
    sep = np.abs(pos[:, None] - pos[None, :])
    free = (mat == -0.5) & (arm_of[:, None] >= 0) & (arm_of[:, None] == arm_of[None, :])
    np.fill_diagonal(free, False)
    bg = np.maximum(1.0, (np.maximum(sep, 1.0) / max(int(law.s0_bp), 1)) ** law.nu)
    stretch = float(s.heatmap_distance_stretching)
    target = np.clip(bg * np.exp(-affinity / 3.0), 1.0, stretch * bg)
    out: F64Array = np.where(free, -target, mat)
    held = int(free.sum()) // 2
    closer = int((free & (affinity > 0.0)).sum()) // 2
    spread = float(np.abs(affinity[free]).mean()) if held else 0.0
    log.status(
        LOG,
        "components: %s pairs of nodes held, %s closer than the background and %s further,"
        " mean absolute affinity %.2f",
        f"{held:,}",
        f"{closer:,}",
        f"{held - closer:,}",
        spread,
    )
    return out
