"""Trace of the orientation term in the smooth stage: bonds and bends beside the anchors it acts on.

    python playground/orientation_trace.py [--region chr1:11000000-13000000] [-n 4] [--cell GM12878]

Builds the region with the default chain and with the uniform chain, each with the term as it
is, with `[motif_orientation] form = stretch` and without the term, on the production config of
the cell line. Every call of the numba smooth kernel is recorded
with its chain before and after. Printed per arm, over all blocks and structures:

- the bonds within two beads of an anchor with a motif direction and a loop partner, as a multiple
  of their target: the median, the 95th percentile and the shares over 2 and over 3 targets,
- the same for the bonds three beads or more from every fixed bead,
- the share of bends sharper than 120 degrees at the beads of the two groups,
- the bonds of such an anchor to its two neighbours and the bonds from those neighbours outwards,
- the chord of such an anchor, the distance of its two neighbours over the sum of the two bond
  targets, and the rank correlation of the chord with the misfit of the anchor,
- the misfit of the term, the mean over loop pairs of ((1 - a.b) / 2)^2, before and after the stage,
  with the vector of an anchor over its two neighbours and over three beads a side,
- the bond energy and the orientation energy of the final chains in the units of the kernel,
- the share of the beads within five beads of an anchor the term acts on.

The arm with the term off is read at the anchors of the arm with the term on. The two arms build
the same blocks with the same nodes, and a block is matched by its fixed beads.

Reads `data/` of the directory it is started in.
"""

from __future__ import annotations

import argparse
import os
import threading
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

import gnome3d.mc.numba as mc_numba
from gnome3d import log
from gnome3d.data import ContactData
from gnome3d.io import parse_region
from gnome3d.production import settings_for_cell
from gnome3d.simulate import simulate
from gnome3d.mc.numba.common import ORIENTATION_STRETCH
from gnome3d.mc.numba.terms import calc_orientation_stretch_nb
from gnome3d.util import calc_orientation

parser = argparse.ArgumentParser()
parser.add_argument("--region", default="chr1:11000000-13000000")
parser.add_argument("-n", type=int, default=4)
parser.add_argument("--cell", default="GM12878")
parser.add_argument("--chains", default="default,uniform")
parser.add_argument("--terms", default="on,stretch,off", help="on, stretch, stretchN for N times the weight, off")
args = parser.parse_args()

records: list[dict[str, Any]] = []
lock = threading.Lock()
original = mc_numba.mc_smooth_numba


def recorded(pos, dtn, fixed, step, s, chars=None, nbrs=None, nbr_w=None, *rest, **kw):  # noqa: ANN001, ANN002, ANN003, ANN202
    before = np.array(pos, dtype=np.float64)
    score = original(pos, dtn, fixed, step, s, chars, nbrs, nbr_w, *rest, **kw)
    if chars is not None or bool(np.any(fixed)):
        with lock:
            records.append(
                {
                    "before": before,
                    "after": np.array(pos, dtype=np.float64),
                    "dtn": np.array(dtn, dtype=np.float64),
                    "fixed": np.array(fixed, dtype=np.bool_),
                    "chars": None if chars is None else np.array(chars),
                    "nbrs": nbrs,
                    "nbr_w": nbr_w,
                    "s": s,
                }
            )
    return score


mc_numba.mc_smooth_numba = recorded


def misfit(
    pos: np.ndarray, sites: np.ndarray, chars: np.ndarray, nbrs: Any, nbr_w: Any, stretch: int = 0
) -> tuple[float, float, int]:
    """Returns the weighted sum and the plain mean of ((1 - a.b) / 2)^2 over loop pairs, with the
    vector of an anchor over `stretch` beads a side, or over its two neighbours at 0.
    """
    n = pos.shape[0]
    if stretch > 0:
        orn = np.array(
            [calc_orientation_stretch_nb(pos, int(b), n, str(chars[b]) == "L", stretch) for b in sites]
        )
    else:
        orn = np.array([calc_orientation(pos, int(b), n, str(chars[b])) for b in sites])
    total, plain, count = 0.0, 0.0, 0
    for k, partners in nbrs.items():
        for j, w in zip(partners, nbr_w[k], strict=True):
            ang = (1.0 - float(orn[k] @ orn[j])) / 2.0
            total += ang * ang * float(w)
            plain += ang * ang
            count += 1
    return total, plain, count


def summarise(tag: str, recs: list[dict[str, Any]], weight: float) -> None:
    near: list[np.ndarray] = []
    far: list[np.ndarray] = []
    bend_near: list[np.ndarray] = []
    bend_far: list[np.ndarray] = []
    beads = reach = 0
    fit_before = fit_after = 0.0
    wide_before = wide_after = 0.0
    pairs = 0
    e_bond = e_orn = e_orn_before = 0.0
    longest = 0.0
    inner: list[float] = []
    outer: list[float] = []
    chords: list[float] = []
    own_misfit: list[float] = []
    for r in recs:
        pos, dtn, fixed, chars = r["after"], r["dtn"], r["fixed"], r["chars"]
        n = pos.shape[0]
        if n < 4:
            continue
        sites = np.flatnonzero(fixed)
        acted = np.zeros(n, dtype=np.bool_)
        if chars is not None and r["nbrs"]:
            for k, partners in r["nbrs"].items():
                if partners and k < len(sites):
                    acted[sites[k]] = True
        bond = np.linalg.norm(np.diff(pos, axis=0), axis=1) / np.maximum(dtn[: n - 1], 1e-9)
        longest = max(longest, float(bond.max()))
        idx = np.arange(n)
        to_acted = np.full(n, 10**9)
        for b in np.flatnonzero(acted):
            to_acted = np.minimum(to_acted, np.abs(idx - b))
        to_fixed = np.full(n, 10**9)
        for b in sites:
            to_fixed = np.minimum(to_fixed, np.abs(idx - b))
        # Bond k joins beads k and k + 1.
        b_near = np.minimum(to_acted[:-1], to_acted[1:]) <= 1
        b_far = np.minimum(to_fixed[:-1], to_fixed[1:]) >= 3
        near.append(bond[b_near])
        far.append(bond[b_far])
        u = np.diff(pos, axis=0)
        u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        cos = np.sum(u[:-1] * u[1:], axis=1)  # at beads 1 .. n - 2
        bend_near.append(cos[to_acted[1:-1] <= 2])
        bend_far.append(cos[to_fixed[1:-1] >= 3])
        beads += n
        reach += int((to_acted <= 5).sum())
        orn = (
            np.array([calc_orientation(pos, int(b), n, str(chars[b])) for b in sites])
            if chars is not None
            else None
        )
        for k, b in enumerate(sites):
            if not acted[b] or b < 2 or b > n - 3:
                continue
            inner += [float(bond[b - 1]), float(bond[b])]
            outer += [float(bond[b - 2]), float(bond[b + 1])]
            chords.append(float(np.linalg.norm(pos[b + 1] - pos[b - 1]) / (dtn[b - 1] + dtn[b])))
            assert orn is not None
            partners = r["nbrs"][k]
            own_misfit.append(float(np.mean([((1.0 - orn[k] @ orn[j]) / 2.0) ** 2 for j in partners])))
        s = r["s"]
        d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        rel = (d - dtn[: n - 1]) / np.maximum(dtn[: n - 1], 1e-9)
        k_bond = np.where(rel >= 0, float(s.spring_stretch), float(s.spring_squeeze))
        e_bond += float((k_bond * rel * rel).sum()) * float(s.smooth_dist_weight)
        if chars is not None and r["nbrs"]:
            t0, p0, c0 = misfit(r["before"], sites, chars, r["nbrs"], r["nbr_w"])
            t1, p1, _c1 = misfit(pos, sites, chars, r["nbrs"], r["nbr_w"])
            fit_before += p0
            fit_after += p1
            pairs += c0
            e_orn += weight * t1
            e_orn_before += weight * t0
            wide_before += misfit(r["before"], sites, chars, r["nbrs"], r["nbr_w"], ORIENTATION_STRETCH)[1]
            wide_after += misfit(pos, sites, chars, r["nbrs"], r["nbr_w"], ORIENTATION_STRETCH)[1]

    def row(name: str, parts: list[np.ndarray]) -> str:
        v = np.concatenate(parts) if parts else np.zeros(0)
        if v.size == 0:
            return f"{name}: none"
        return (
            f"{name}: {v.size} bonds, median {np.median(v):.2f}, 95th {np.percentile(v, 95):.2f},"
            f" over 2 targets {100 * (v > 2).mean():.1f} %, over 3 targets {100 * (v > 3).mean():.1f} %"
        )

    def bends(parts: list[np.ndarray]) -> str:
        v = np.concatenate(parts) if parts else np.zeros(0)
        return "none" if v.size == 0 else f"{100 * (v < -0.5).mean():.1f} % of {v.size}"

    print(f"TRACE {tag}: {len(recs)} smooth calls, {beads} beads, longest bond {longest:.1f} targets")
    print("TRACE   " + row("bonds within two beads of an anchor of the term", near))
    print("TRACE   " + row("bonds three beads or more from a fixed bead", far))
    print(f"TRACE   bends sharper than 120 degrees: beside the term {bends(bend_near)}, away {bends(bend_far)}")
    if chords:
        from scipy.stats import spearmanr

        vi, vo, vc = np.array(inner), np.array(outer), np.array(chords)
        rho = float(spearmanr(vc, np.array(own_misfit)).statistic)
        print(
            f"TRACE   bonds of an anchor of the term to its neighbours: median {np.median(vi):.2f},"
            f" over 2 targets {100 * (vi > 2).mean():.1f} %; from the neighbours outwards: median"
            f" {np.median(vo):.2f}, over 2 targets {100 * (vo > 2).mean():.1f} %"
        )
        print(
            f"TRACE   chord of {vc.size} anchors: median {np.median(vc):.2f}, 95th {np.percentile(vc, 95):.2f},"
            f" over 1.5 {100 * (vc > 1.5).mean():.1f} %, under 0.5 {100 * (vc < 0.5).mean():.1f} %;"
            f" rank correlation with the misfit of the anchor {rho:+.2f}"
        )
    if pairs:
        print(
            f"TRACE   misfit over {pairs} loop pair entries: {fit_before / pairs:.3f} before the stage,"
            f" {fit_after / pairs:.3f} after; over three beads a side {wide_before / pairs:.3f} before,"
            f" {wide_after / pairs:.3f} after"
        )
    print(
        f"TRACE   energy of the final chains: bonds {e_bond:.2f}, orientation at the weight of the"
        f" form neighbours {e_orn:.2f} ({e_orn_before:.2f} before the stage)"
    )
    if beads:
        print(f"TRACE   beads within five beads of an anchor of the term: {100 * reach / beads:.1f} %")


data_root = str(Path("data").resolve())
bed = parse_region(args.region)
assert bed is not None, args.region
for chain in args.chains.split(","):
    sites_of: dict[bytes, dict[str, Any]] = {}
    for term in args.terms.split(","):
        s = settings_for_cell(args.cell, data_root=data_root, uniform=chain == "uniform")
        if term == "off":
            s.use_ctcf_motif = False
        if term.startswith("stretch"):
            s.motif_form = "stretch"
            s.motif_stretch_weight = float(term[len("stretch") :] or 1.0)
        log.setup(s.output_level)
        records.clear()
        data = ContactData.from_files(s, [bed.chr], bed)
        simulate(s, data, [bed.chr], args.n, region=bed)
        # The cross block relaxation calls the kernel without the term. It is left out.
        if term != "off":
            stage = [r for r in records if r["chars"] is not None]
            sites_of = {r["fixed"].tobytes(): r for r in stage}
        else:
            stage = []
            for r in records:
                twin = sites_of.get(r["fixed"].tobytes())
                if twin is not None:
                    stage.append({**r, "chars": twin["chars"], "nbrs": twin["nbrs"], "nbr_w": twin["nbr_w"]})
        summarise(f"{chain} chain, term {term}, {args.cell} {args.region}", stage, float(s.motif_weight))
