"""Choose background_weight on real blocks, with the two band exponents as the gate.

Every arcless anchor pair now sits on the background for its separation, held by a spring of
this weight, in place of the old 1/d repulsion. The weight trades two things. Too weak and the
short range collapses under the loops as before, which is a flat curve under 100 kb and a steep
one above it. Too strong and the arcs cannot pull, which is a curve that follows the background
whatever the data says. Chain bonds, the full weight spring on consecutive arcless pairs, are
swept alongside since a background on every pair may make them redundant.

Blocks come from a finished model and the cell's own bedpe, the law from the cell's measured
exponent, and each arm is solved from a common start.

    python playground/background_weight_sweep.py <model.cif> <arcs.bedpe> <nu> [chr]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gnome3d.mc.numba.arcs_solver import solve_arcs
from gnome3d.pipeline.coarse.build import add_chain_bonds, arc_expected_matrix
from gnome3d.polymer import PolymerLaw, fit_arc_strength
from gnome3d.settings import Settings
from gnome3d.types import InteractionArc
from validation_battery import block_owner, load


def read_arcs(path: str, chrom: str) -> list[tuple[int, int, int]]:
    out = []
    for ln in open(path):
        f = ln.split()
        if len(f) < 6 or f[0] != chrom or f[3] != chrom:
            continue
        sc = int(f[6]) if len(f) > 6 and f[6].lstrip("-").isdigit() else 1
        out.append(((int(f[1]) + int(f[2])) // 2, (int(f[4]) + int(f[5])) // 2, sc))
    return out


def band_exponent(pos: np.ndarray, mids: np.ndarray, lo: float, hi: float) -> float:
    iu = np.triu_indices(len(pos), 1)
    d = np.linalg.norm(pos[iu[0]] - pos[iu[1]], axis=1)
    sep = np.abs(mids[iu[0]] - mids[iu[1]]).astype(float)
    k = (sep >= lo) & (sep <= hi) & (d > 0)
    return float(np.polyfit(np.log(sep[k]), np.log(d[k]), 1)[0]) if k.sum() > 40 else float("nan")


def main() -> None:
    model, bedpe, nu = sys.argv[1], sys.argv[2], float(sys.argv[3])
    chrom = sys.argv[4] if len(sys.argv) > 4 else "chr1"
    pos, start, bmid, anchor = load(Path(model))
    owner = block_owner(bmid, anchor, 1000)
    ai = np.where(anchor)[0]
    amid = bmid[ai]
    order = np.argsort(amid)
    asort = amid[order]

    def nearest(p: int) -> int:
        k = int(np.clip(np.searchsorted(asort, p), 0, len(asort) - 1))
        best, bd = -1, 1 << 62
        for t in (k - 1, k, k + 1):
            if 0 <= t < len(asort) and abs(int(asort[t]) - p) < bd:
                bd, best = abs(int(asort[t]) - p), int(ai[order[t]])
        return best if bd <= 20_000 else -1

    per_bead: dict[tuple[int, int], int] = {}
    for p1, p2, sc in read_arcs(bedpe, chrom):
        u, v = nearest(p1), nearest(p2)
        if u >= 0 and v >= 0 and u != v:
            key = (min(u, v), max(u, v))
            per_bead[key] = max(per_bead.get(key, 0), sc)
    all_arcs = [InteractionArc(0, 1, sc, genomic_start=int(bmid[u]), genomic_end=int(bmid[v])) for (u, v), sc in per_bead.items()]
    strength = fit_arc_strength(all_arcs)

    blocks = [np.where((owner == b) & anchor)[0] for b in np.unique(owner[ai])]
    blocks = sorted((b for b in blocks if len(b) >= 40), key=lambda b: -len(b))[:8]
    rng = np.random.default_rng(0)
    print(f"{len(blocks)} blocks, nu {nu}, arc strength fit {'ok' if strength.ok else strength.reason}")
    print(f"{'weight':>7} {'chain':>5} | {'20-100k':>8} {'100k-1M':>8} {'ovlp/k':>7} {'arc d/t':>8} {'bg d/t':>7} {'Rg':>6}")
    for chain in (True, False):
        for w in (0.01, 0.03, 0.1, 0.3, 1.0):
            rows = []
            for blk in blocks:
                idx = {int(g): k for k, g in enumerate(blk)}
                mids = [int(bmid[g]) for g in blk]
                arcs = [(idx[u], idx[v], sc) for (u, v), sc in per_bead.items() if u in idx and v in idx]
                s = Settings()
                s.polymer = PolymerLaw(nu=nu, s0_bp=1000, q_half=1.0, arcs=strength)
                s.use_arcs_chain_bonds = chain
                s.arcs_chain_bond_scale = 1.5
                s.background_weight = w
                s.use_confinement = True
                s.confinement_apply_to_arcs = True
                s.use_excluded_volume = True
                s.exclusion_apply_to_arcs = True
                s.exclusion_weight = 0.1
                mat = add_chain_bonds(arc_expected_matrix(s, mids, arcs), mids, s)
                n = len(mids)
                start_pos = np.ascontiguousarray(rng.normal(0.0, 1.5, (n, 3)).astype(np.float32))
                _, out = solve_arcs(start_pos, mat, s)
                p = np.asarray(out, np.float64)
                m = np.array(mids, dtype=float)
                iu = np.triu_indices(n, 1)
                d = np.linalg.norm(p[iu[0]] - p[iu[1]], axis=1)
                e = mat[iu]
                bond = float(np.median(np.linalg.norm(np.diff(p[np.argsort(m)], axis=0), axis=1)))
                far = np.abs(iu[0] - iu[1]) > 1
                rows.append([
                    band_exponent(p, m, 2e4, 1e5), band_exponent(p, m, 1e5, 1e6),
                    1000.0 * float((d[far] < 0.7 * bond).sum()) / n,
                    float(np.median(d[e > 1e-6] / e[e > 1e-6])) if (e > 1e-6).any() else np.nan,
                    float(np.median(d[e < 0] / -e[e < 0])) if (e < 0).any() else np.nan,
                    float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean())),
                ])
            r = np.nanmean(rows, axis=0)
            print(f"{w:>7.2f} {'on' if chain else 'off':>5} | {r[0]:>8.3f} {r[1]:>8.3f} {r[2]:>7.1f} {r[3]:>8.2f} {r[4]:>7.2f} {r[5]:>6.1f}", flush=True)
    print("\n  target for both bands is nu; arc d/t is realised over target for arc pairs, bg d/t the same for")
    print("  background pairs, both 1.0 when satisfied; ovlp/k is anchor pairs under 0.7 bonds per thousand.")


if __name__ == "__main__":
    main()
