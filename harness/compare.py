#!/usr/bin/env python3
"""
harness/compare.py - 3dgnome-ng correctness harness.

Compares the C++ reference scorer against the Python (src/) reimplementation.
Run from the repository root:

    python harness/compare.py              # run all tests
    python harness/compare.py distfns      # run one test group
    python harness/compare.py --build-only # just compile scorer
    python harness/compare.py --reference  # print C++ reference values only

Exits 0 when all implemented tests pass, 1 on any failure.
The harness skips tests whose Python counterpart is not yet implemented -
it never fails just because src/ is incomplete.
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCORER_SRC = ROOT / "harness" / "scorer.cpp"
SCORER_BIN = ROOT / "harness" / "scorer"
HARNESS_DIR = ROOT / "harness"

ATOL = 1e-6  # absolute tolerance for floating-point comparisons


# ---------------------------------------------------------------------------
# Build helpers

def build_scorer(force: bool = False) -> None:
    if not force and SCORER_BIN.exists():
        return
    mc = ROOT / "3dnome" / "MC"
    sources = (
        list(mc.glob("*.cpp")) +
        list((mc / "lib").glob("*.cpp")) +
        list((mc / "lib").glob("*.c"))
    )
    # Exclude main.cpp (it's in tools/, not MC/ directly, so not picked up above)
    cmd = [
              "g++", "-std=c++0x", "-Wno-write-strings", "-O2",
              f"-I{mc}",
              "-o", str(SCORER_BIN),
              str(SCORER_SRC),
          ] + [str(s) for s in sources] + ["-lm"]
    print(f"[build] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(f"[build] scorer.cpp compilation failed")
    print("[build] scorer compiled OK")


def run_scorer(*args: str, stdin_text: str = "") -> str:
    cmd = [str(SCORER_BIN)] + list(args)
    result = subprocess.run(cmd, input=stdin_text, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"[scorer] error: {result.stderr.strip()}")
    # LooperSolver constructor emits setup lines to stdout before the actual result.
    # The result is always the last non-empty line.
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    # For distfns mode the output is multi-line ("type value -> result" per query);
    # for single-value modes it's one line. Reconstruct correctly:
    #   - if any line contains " -> ", it's distfns multi-line output
    #   - otherwise take only the last line as the scalar result
    if any(" -> " in l for l in lines):
        return "\n".join(l for l in lines if " -> " in l)
    return lines[-1] if lines else ""


def run_scorer_filtered(*args: str, prefixes: tuple = (), stdin_text: str = "") -> str:
    """Like run_scorer but returns all lines whose first token is in `prefixes`."""
    cmd = [str(SCORER_BIN)] + list(args)
    result = subprocess.run(cmd, input=stdin_text, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"[scorer] error: {result.stderr.strip()}")
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    if prefixes:
        lines = [l for l in lines if l.split()[0] in prefixes]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fixture helpers

def write_tmp(content: str, suffix: str = ".txt") -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    f.write(content)
    f.close()
    return f.name


def positions_txt(positions) -> str:
    return "\n".join(f"{x} {y} {z}" for x, y, z in positions)


def matrix_txt(mat) -> str:
    return "\n".join(" ".join(f"{v:.15f}" for v in row) for row in mat)


def arcs_txt(arcs) -> str:
    return "\n".join(f"{i} {j} {d:.15f}" for i, j, d in arcs)


def dtn_txt(dtn) -> str:
    return "\n".join(f"{d:.15f}" for d in dtn)


# ---------------------------------------------------------------------------
# Reference C++ computations (as Python, matching scorer.cpp exactly)
# These are NOT the implementation-under-test - they are ground truth for
# generating expected values without running the binary.

def ref_genomic_dist(length_bp, base, scale, power):
    return base + scale * (length_bp / 1000.0) ** power


def ref_freq_to_dist_heatmap(freq, scale, power):
    return scale * (freq ** power)


def ref_freq_to_dist(freq, a, scale, shift, base_level):
    return base_level + scale / math.exp(a * (freq + shift))


def ref_angle_metric(v1, v2):
    """Matches common.cpp angle() exactly: 1 - (dot(normalized(v1), normalized(v2)) + 1) / 2"""
    l1 = math.sqrt(sum(x * x for x in v1))
    l2 = math.sqrt(sum(x * x for x in v2))
    if l1 < 1e-10 or l2 < 1e-10:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2)) / (l1 * l2)
    return 1.0 - (dot + 1.0) / 2.0


def ref_vlen(v):
    return math.sqrt(sum(x * x for x in v))


def ref_score_heatmap(positions, exp_dist, diag_size):
    n = len(positions)
    err = 0.0
    for moved in range(n):
        for i in range(n):
            if abs(i - moved) < diag_size:
                continue
            if exp_dist[i][moved] < 1e-6:
                continue
            d = ref_vlen([positions[i][k] - positions[moved][k] for k in range(3)])
            cerr = (d - exp_dist[i][moved]) / exp_dist[i][moved]
            err += cerr * cerr
    return err


def ref_score_arcs(positions, arcs, stretch_k, squeeze_k):
    sc = 0.0
    for i, j, exp_d in arcs:
        d = ref_vlen([positions[i][k] - positions[j][k] for k in range(3)])
        if exp_d < 0.0:
            sc += 1.0 / max(d, 1e-10)
            continue
        if exp_d < 1e-6:
            continue
        rel = (d - exp_d) / exp_d
        sc += rel * rel * (stretch_k if rel >= 0 else squeeze_k)
    return sc


def ref_score_smooth(positions, dist_to_next, stretch_k, squeeze_k, angular_k, w_dist, w_angle):
    n = len(positions)
    sca, scb = 0.0, 0.0
    v_prev = None
    for i in range(n - 1):
        v = [positions[i][k] - positions[i + 1][k] for k in range(3)]
        vlen = ref_vlen(v)
        dtn = dist_to_next[i] if i < len(dist_to_next) else 1.0
        if dtn < 1e-6:
            dtn = 1e-6
        diff = (vlen - dtn) / dtn
        sca += diff * diff * (stretch_k if diff >= 0 else squeeze_k)
        if v_prev is not None:
            ang = ref_angle_metric(v, v_prev)
            scb += ang * ang * ang * angular_k
        v_prev = v
    return sca * w_dist + scb * w_angle


def ref_metropolis_prob(jump_scale, jump_coef, score_curr, score_prev, T):
    if T <= 0.0:
        return 0.0
    return jump_scale * math.exp(-jump_coef * (score_curr / score_prev) / T)


# ---------------------------------------------------------------------------
# Test framework

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

_results = []


def check(name: str, cpp_val: float, py_val, atol: float = ATOL):
    """Compare C++ reference value to Python implementation value."""
    if py_val is None:
        print(f"  {SKIP}  {name}  (not implemented)")
        _results.append(("skip", name))
        return
    ok = abs(cpp_val - py_val) <= atol
    status = PASS if ok else FAIL
    if ok:
        print(f"  {status}  {name}  cpp={cpp_val:.8g}  py={py_val:.8g}")
    else:
        print(f"  {status}  {name}  cpp={cpp_val:.8g}  py={py_val:.8g}  diff={abs(cpp_val - py_val):.3e}")
    _results.append(("pass" if ok else "fail", name))


def check_close_enough(name: str, cpp_val: float, py_val, rtol: float = 1e-5):
    """Relative tolerance check (for values that can be large)."""
    if py_val is None:
        print(f"  {SKIP}  {name}  (not implemented)")
        _results.append(("skip", name))
        return
    denom = max(abs(cpp_val), abs(py_val), 1e-30)
    ok = abs(cpp_val - py_val) / denom <= rtol
    status = PASS if ok else FAIL
    if ok:
        print(f"  {status}  {name}  cpp={cpp_val:.8g}  py={py_val:.8g}")
    else:
        print(f"  {status}  {name}  cpp={cpp_val:.8g}  py={py_val:.8g}  rel={abs(cpp_val - py_val) / denom:.3e}")
    _results.append(("pass" if ok else "fail", name))


# ---------------------------------------------------------------------------
# Try to import Python implementation

def _try_import(fn_name: str):
    """Return function from src/ or None if not yet implemented."""
    try:
        import importlib
        sys.path.insert(0, str(ROOT / "src"))
        # Adjust module paths as src/ is developed
        mod = importlib.import_module("energy")
        fn = getattr(mod, fn_name, None)
        if fn is None:
            return None
        return fn
    except (ImportError, ModuleNotFoundError):
        return None


# ---------------------------------------------------------------------------
# Test groups

DEFAULT_DIST_PARAMS = dict(
    base=1.0, scale=0.5, power=0.75,
    freq_scale=25.0, freq_power=-0.6,
    freq_scale_inter=120.0, freq_power_inter=-1.0,
    count_a=0.2, count_scale=1.8, count_shift=8, count_base=0.2,
)


def test_distfns(reference_only=False):
    print("\n[distfns] Distance conversion functions")
    p = DEFAULT_DIST_PARAMS

    param_args = [
        str(p["base"]), str(p["scale"]), str(p["power"]),
        str(p["freq_scale"]), str(p["freq_power"]),
        str(p["freq_scale_inter"]), str(p["freq_power_inter"]),
        str(p["count_a"]), str(p["count_scale"]), str(p["count_shift"]),
        str(p["count_base"]),
    ]

    test_cases = [
        ("genomic", 1000),
        ("genomic", 10000),
        ("genomic", 100000),
        ("genomic", 1000000),
        ("freq", 0.5),
        ("freq", 1.0),
        ("freq", 2.0),
        ("freq_inter", 0.1),
        ("freq_inter", 1.0),
        ("count", 2),
        ("count", 10),
        ("count", 50),
    ]

    stdin_text = "\n".join(f"{t} {v}" for t, v in test_cases)
    cpp_out = run_scorer("distfns", *param_args, stdin_text=stdin_text)

    if reference_only:
        print(cpp_out)
        return

    py_genomic = _try_import("genomic_length_to_distance")
    py_freq = _try_import("freq_to_dist_heatmap")
    py_freq_i = _try_import("freq_to_dist_heatmap_inter")
    py_count = _try_import("freq_to_distance")

    for line in cpp_out.splitlines():
        parts = line.split()
        if parts[0] == "genomic":
            bp, cpp_val = int(float(parts[1])), float(parts[3])
            expected = ref_genomic_dist(bp, **{k: p[k] for k in ("base", "scale", "power")})
            assert abs(cpp_val - expected) < 1e-4, f"scorer/ref mismatch at genomic {bp}"
            py_val = py_genomic(bp, p["base"], p["scale"], p["power"]) if py_genomic else None
            check(f"genomic_dist({bp}bp)", cpp_val, py_val)
        elif parts[0] == "freq":
            f_, cpp_val = float(parts[1]), float(parts[3])
            py_val = py_freq(f_, p["freq_scale"], p["freq_power"]) if py_freq else None
            check(f"freq_to_dist_heatmap({f_})", cpp_val, py_val)
        elif parts[0] == "freq_inter":
            f_, cpp_val = float(parts[1]), float(parts[3])
            py_val = py_freq_i(f_, p["freq_scale_inter"], p["freq_power_inter"]) if py_freq_i else None
            check(f"freq_to_dist_inter({f_})", cpp_val, py_val)
        elif parts[0] == "count":
            n_, cpp_val = int(float(parts[1])), float(parts[3])
            py_val = py_count(n_, p["count_a"], p["count_scale"], p["count_shift"],
                              p["count_base"]) if py_count else None
            check(f"freq_to_distance(count={n_})", cpp_val, py_val)


SYNTHETIC_POSITIONS_5 = [
    (0.0, 0.0, 0.0),
    (3.0, 0.0, 0.0),
    (3.0, 4.0, 0.0),
    (0.0, 4.0, 2.0),
    (1.0, 2.0, 5.0),
]


def _build_exp_dist(positions, expected_fn):
    n = len(positions)
    return [[expected_fn(i, j) for j in range(n)] for i in range(n)]


def test_heatmap(reference_only=False):
    print("\n[heatmap] Heatmap energy score")
    pos = SYNTHETIC_POSITIONS_5
    n = len(pos)
    diag = 2
    stretch_k, squeeze_k = 1.0, 1.0

    # Build a simple expected distance matrix: actual distances * 0.9 (slight mismatch)
    def actual_dist(i, j):
        return math.sqrt(sum((pos[i][k] - pos[j][k]) ** 2 for k in range(3)))

    exp_dist = [[actual_dist(i, j) * 0.9 if i != j else 0.0 for j in range(n)] for i in range(n)]

    pos_f = write_tmp(positions_txt(pos))
    expd_f = write_tmp(matrix_txt(exp_dist))
    try:
        cpp_val = float(run_scorer("heatmap", str(diag), pos_f, expd_f))
    finally:
        os.unlink(pos_f);
        os.unlink(expd_f)

    ref_val = ref_score_heatmap(pos, exp_dist, diag)
    assert abs(cpp_val - ref_val) < 1e-4, f"scorer/ref mismatch: {cpp_val} vs {ref_val}"

    if reference_only:
        print(f"  heatmap score = {cpp_val:.10f}")
        return

    py_fn = _try_import("score_heatmap")
    import importlib, sys as _sys
    py_val = None
    if py_fn:
        import torch
        pos_t = torch.tensor(pos, dtype=torch.float64)
        expd_t = torch.tensor(exp_dist, dtype=torch.float64)
        py_val = py_fn(pos_t, expd_t, diag).item()

    check_close_enough("heatmap_score(5 beads, diag=2)", cpp_val, py_val)


def test_arcs(reference_only=False):
    print("\n[arcs] Arc spring energy score")
    pos = SYNTHETIC_POSITIONS_5
    arcs = [(0, 2, 5.0), (1, 4, 6.0), (2, 3, 3.0), (0, 4, -1.0)]  # last arc: repulsion
    stretch_k, squeeze_k = 1.0, 1.0

    pos_f = write_tmp(positions_txt(pos))
    arcs_f = write_tmp(arcs_txt(arcs))
    try:
        cpp_val = float(run_scorer("arcs", str(stretch_k), str(squeeze_k), pos_f, arcs_f))
    finally:
        os.unlink(pos_f);
        os.unlink(arcs_f)

    ref_val = ref_score_arcs(pos, arcs, stretch_k, squeeze_k)
    assert abs(cpp_val - ref_val) < 1e-4, f"scorer/ref mismatch: {cpp_val} vs {ref_val}"

    if reference_only:
        print(f"  arc score = {cpp_val:.10f}")
        return

    py_fn = _try_import("score_arcs")
    py_val = None
    if py_fn:
        import torch
        pos_t = torch.tensor(pos, dtype=torch.float64)
        arcs_t = [(i, j, d) for i, j, d in arcs]
        py_val = py_fn(pos_t, arcs_t, stretch_k, squeeze_k).item()

    check_close_enough("arc_score(5 beads, 4 arcs)", cpp_val, py_val)


def test_smooth(reference_only=False):
    print("\n[smooth] Chain smoothness energy score")
    pos = SYNTHETIC_POSITIONS_5
    n = len(pos)
    # dist_to_next: expected bond lengths (slightly off from actual)
    import math as _math
    dist_to_next = [
        _math.sqrt(sum((pos[i][k] - pos[i + 1][k]) ** 2 for k in range(3))) * 0.85
        for i in range(n - 1)
    ]
    stretch_k, squeeze_k, angular_k = 0.1, 0.1, 0.1
    w_dist, w_angle = 1.0, 1.0

    pos_f = write_tmp(positions_txt(pos))
    dtn_f = write_tmp(dtn_txt(dist_to_next))
    try:
        cpp_val = float(run_scorer(
            "smooth",
            str(stretch_k), str(squeeze_k), str(angular_k),
            str(w_dist), str(w_angle),
            pos_f, dtn_f
        ))
    finally:
        os.unlink(pos_f);
        os.unlink(dtn_f)

    ref_val = ref_score_smooth(pos, dist_to_next, stretch_k, squeeze_k, angular_k, w_dist, w_angle)
    assert abs(cpp_val - ref_val) < 1e-4, f"scorer/ref mismatch: {cpp_val} vs {ref_val}"

    if reference_only:
        print(f"  smooth score = {cpp_val:.10f}")
        return

    py_fn = _try_import("score_smooth")
    py_val = None
    if py_fn:
        import torch
        pos_t = torch.tensor(pos, dtype=torch.float64)
        dtn_t = torch.tensor(dist_to_next, dtype=torch.float64)
        py_val = py_fn(pos_t, dtn_t, stretch_k, squeeze_k, angular_k, w_dist, w_angle).item()

    check_close_enough("smooth_score(5 beads)", cpp_val, py_val)


def test_densify(reference_only=False):
    """
    Validates _densify_active_region: bead count, fixed positions, dtn sign,
    and subanchor linear interpolation.  Pure Python - no C++ scorer needed.
    """
    print("\n[densify] Subanchor densification (_densify_active_region)")

    if reference_only:
        print("  (pure Python validation - no C++ reference)")
        return

    try:
        import numpy as _np
        _root = str(ROOT)
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from src.solver import Solver
        from src.hierarchy import Cluster, LVL_ANCHOR
    except ImportError as exc:
        for name in ("densify.bead_count", "densify.n_fixed",
                     "densify.anchor_pos", "densify.dtn_nonneg", "densify.interp"):
            print(f"  {SKIP}  {name}  ({exc})")
            _results.append(("skip", name))
        return

    LD = 3  # loop_density
    anchor_starts = [0, 2000, 5000, 8000]
    anchor_ends = [1000, 3000, 6000, 9000]
    pos3d = [
        _np.array([0.0, 0.0, 0.0], dtype=_np.float32),
        _np.array([1.0, 0.0, 0.0], dtype=_np.float32),
        _np.array([2.0, 1.0, 0.0], dtype=_np.float32),
        _np.array([3.0, 1.0, 0.5], dtype=_np.float32),
    ]

    class _FakeSettings:
        loop_density = LD

        @staticmethod
        def genomic_length_to_distance(bp):
            return 1.0 + 0.5 * (max(bp, 0) / 1000.0) ** 0.75

    solver = Solver.__new__(Solver)
    solver.s = _FakeSettings()
    solver.clusters = []
    active_region = []
    for i, (s, e, p) in enumerate(zip(anchor_starts, anchor_ends, pos3d)):
        c = Cluster(start=s, end=e, level=LVL_ANCHOR)
        c.pos = p.copy()
        solver.clusters.append(c)
        active_region.append(i)

    pos, fixed, gpos, dtn, anchor_map = solver._densify_active_region(active_region)

    n_anc = len(anchor_starts)
    exp_n = n_anc + (n_anc - 1) * LD  # 4 + 3*3 = 13

    # 1. Total bead count
    check("densify.bead_count", float(exp_n), float(len(pos)), atol=0)

    # 2. Number of fixed beads == n_anchors
    check("densify.n_fixed", float(n_anc), float(int(fixed.sum())), atol=0)

    # 3. Anchor 3-D positions preserved in pos array
    ok_pos = all(_np.allclose(pos[bi], solver.clusters[ci].pos, atol=1e-5)
                 for bi, ci in anchor_map)
    check("densify.anchor_pos", 1.0, 1.0 if ok_pos else 0.0, atol=0)

    # 4. All dtn values are non-negative
    ok_dtn = bool((dtn >= 0).all())
    check("densify.dtn_nonneg", 1.0, 1.0 if ok_dtn else 0.0, atol=0)

    # 5. Subanchor positions are linearly interpolated between adjacent anchor beads
    ok_interp = True
    for seg in range(n_anc - 1):
        ca_i = seg * (LD + 1)
        cb_i = ca_i + (LD + 1)
        for j in range(LD):
            t = (j + 1.0) / (LD + 1)
            exp_pos = (1.0 - t) * pos[ca_i] + t * pos[cb_i]
            if not _np.allclose(pos[ca_i + 1 + j], exp_pos, atol=1e-5):
                ok_interp = False
    check("densify.interp", 1.0, 1.0 if ok_interp else 0.0, atol=0)


def orient_spec_txt(anchors, arcs):
    """
    anchors: list of (active_region_idx, orientation_char)
    arcs:    list of (anchor_list_i, anchor_list_j, weight)
    """
    lines = [str(len(anchors))]
    for ar_idx, ch in anchors:
        lines.append(f"{ar_idx} {ch}")
    lines.append(str(len(arcs)))
    for ai, aj, w in arcs:
        lines.append(f"{ai} {aj} {w:.15f}")
    return "\n".join(lines)


def ref_calc_orientation(positions, cind, n, char_orientation):
    """Matches C++ calcOrientation(cind)."""
    if cind == 0:
        orn = [positions[cind + 1][k] - positions[cind][k] for k in range(3)]
    elif cind == n - 1:
        orn = [positions[cind][k] - positions[cind - 1][k] for k in range(3)]
    else:
        orn = [positions[cind + 1][k] - positions[cind - 1][k] for k in range(3)]
    if char_orientation == 'L':
        orn = [-x for x in orn]
    norm = ref_vlen(orn)
    if norm > 1e-12:
        orn = [x / norm for x in orn]
    return orn


def ref_score_orientation_full(anchor_orn, neighbors, neighbor_weights, motif_weight,
                               motifs_symmetric=True):
    err = 0.0
    sign = 1.0 if motifs_symmetric else -1.0
    for i, nbrs in neighbors.items():
        ws = neighbor_weights[i]
        for k, j in enumerate(nbrs):
            v2 = [sign * x for x in anchor_orn[j]]
            ang = ref_angle_metric(anchor_orn[i], v2)
            err += ang * ang * ws[k]
    return err * motif_weight


def ref_score_orientation_local(anchor_orn, anchor_index, neighbors, motif_weight,
                                motifs_symmetric=True):
    err = 0.0
    sign = 1.0 if motifs_symmetric else -1.0
    for j in neighbors[anchor_index]:
        v2 = [sign * x for x in anchor_orn[j]]
        ang = ref_angle_metric(anchor_orn[anchor_index], v2)
        err += ang * ang
    return err * motif_weight


def test_orientation(reference_only=False):
    print("\n[orientation] CTCF orientation energy score")

    # 11-bead chain; anchors at active-region positions 0, 5, 10
    # (anchor list positions 0, 1, 2)
    pos = [
        (0.0, 0.0, 0.0),  # anchor 0  'R'
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (5.0, 0.0, 0.0),  # anchor 1  'L'
        (5.0, 1.0, 0.0),
        (5.0, 2.0, 0.0),
        (5.0, 3.0, 0.0),
        (5.0, 4.0, 0.0),
        (5.0, 5.0, 0.0),  # anchor 2  'R'
    ]
    n = len(pos)
    anchors_spec = [(0, 'R'), (5, 'L'), (10, 'R')]
    # arcs between anchor-list pairs with weights sqrt(arc_score)
    arcs_spec = [(0, 1, 1.5), (1, 2, 2.0)]
    motif_weight = 2.5
    motifs_sym = 1  # True

    pos_f = write_tmp(positions_txt(pos))
    spec_f = write_tmp(orient_spec_txt(anchors_spec, arcs_spec))
    try:
        raw = run_scorer_filtered(
            "orientation", str(motif_weight), str(motifs_sym), pos_f, spec_f,
            prefixes=("orientation", "global", "local"))
    finally:
        os.unlink(pos_f);
        os.unlink(spec_f)

    # Parse scorer output
    cpp_orn = {}
    cpp_global = None
    cpp_local = {}
    for line in raw.splitlines():
        parts = line.split()
        if parts[0] == "orientation":
            k = int(parts[1])
            cpp_orn[k] = (float(parts[2]), float(parts[3]), float(parts[4]))
        elif parts[0] == "global":
            cpp_global = float(parts[1])
        elif parts[0] == "local":
            k = int(parts[1])
            cpp_local[k] = float(parts[2])

    # Build reference values (Python)
    n_anchors = len(anchors_spec)
    ref_orn = [
        ref_calc_orientation(pos, anchors_spec[k][0], n, anchors_spec[k][1])
        for k in range(n_anchors)
    ]
    neighbors_py = {0: [1], 1: [0, 2], 2: [1]}
    weights_py = {0: [1.5], 1: [1.5, 2.0], 2: [2.0]}

    # Verify scorer vs reference (sanity check)
    for k in range(n_anchors):
        for dim in range(3):
            assert abs(cpp_orn[k][dim] - ref_orn[k][dim]) < 1e-5, \
                f"orientation vector mismatch anchor {k} dim {dim}"

    ref_global = ref_score_orientation_full(ref_orn, neighbors_py, weights_py,
                                            motif_weight, motifs_sym == 1)
    assert abs(cpp_global - ref_global) < 1e-4, \
        f"scorer/ref global mismatch: {cpp_global} vs {ref_global}"
    for k in range(n_anchors):
        ref_loc = ref_score_orientation_local(ref_orn, k, neighbors_py, motif_weight, motifs_sym == 1)
        assert abs(cpp_local[k] - ref_loc) < 1e-4, \
            f"scorer/ref local[{k}] mismatch: {cpp_local[k]} vs {ref_loc}"

    if reference_only:
        print(f"  global = {cpp_global:.10f}")
        for k in range(n_anchors):
            print(f"  local[{k}] = {cpp_local[k]:.10f}")
            print(f"  orientation[{k}] = {cpp_orn[k]}")
        return

    # Python implementation checks
    import numpy as _np
    py_calc_orn = _try_import("calc_orientation")
    py_score_orn = _try_import("score_orientation")
    py_local_orn = _try_import("local_score_orientation")

    pos_np = _np.array(pos, dtype=_np.float64)
    py_orn = None
    if py_calc_orn:
        py_orn = [
            py_calc_orn(pos_np, anchors_spec[k][0], n, anchors_spec[k][1])
            for k in range(n_anchors)
        ]
        for k in range(n_anchors):
            for dim in range(3):
                check(f"calc_orientation[{k}][{dim}]",
                      cpp_orn[k][dim], float(py_orn[k][dim]))

    # Global score
    py_global = None
    if py_score_orn and py_orn is not None:
        nbrs = {0: [1], 1: [0, 2], 2: [1]}
        wts = {0: [1.5], 1: [1.5, 2.0], 2: [2.0]}
        py_global = py_score_orn(py_orn, nbrs, wts, motif_weight, bool(motifs_sym))
    check_close_enough("score_orientation_global", cpp_global, py_global)

    # Local scores
    for k in range(n_anchors):
        py_loc = None
        if py_local_orn and py_orn is not None:
            nbrs = {0: [1], 1: [0, 2], 2: [1]}
            py_loc = py_local_orn(py_orn, k, nbrs, motif_weight, bool(motifs_sym))
        check_close_enough(f"score_orientation_local[{k}]", cpp_local[k], py_loc)


def test_metropolis(reference_only=False):
    print("\n[metropolis] Metropolis acceptance probability")

    cases = [
        (50.0, 20.0, 1.1, 1.0, 5.0),  # slight worsening, high T
        (50.0, 20.0, 2.0, 1.0, 1.0),  # large worsening, medium T
        (50.0, 20.0, 1.0, 1.0, 0.01),  # no change, near-zero T
        (50.0, 20.0, 0.9, 1.0, 5.0),  # improvement (never reaches stochastic branch)
    ]

    py_fn = _try_import("metropolis_prob") if not reference_only else None

    for js, jc, sc, sp, T in cases:
        cpp_val = float(run_scorer("metropolis", str(js), str(jc), str(sc), str(sp), str(T)))
        ref_val = ref_metropolis_prob(js, jc, sc, sp, T)
        assert abs(cpp_val - ref_val) < 1e-4, f"scorer/ref mismatch: {cpp_val} vs {ref_val}"

        if reference_only:
            print(f"  metropolis(js={js}, jc={jc}, sc={sc}, sp={sp}, T={T}) = {cpp_val:.8f}")
            continue

        py_val = py_fn(js, jc, sc, sp, T) if py_fn else None
        check(f"metropolis(sc={sc}, sp={sp}, T={T})", cpp_val, py_val)


def test_angle(reference_only=False):
    print("\n[angle] Custom angle metric  (NOT acos - see common.cpp line 40)")
    cases = [
        ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.0),  # parallel
        ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), 1.0),  # anti-parallel
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.5),  # perpendicular
    ]
    py_fn = _try_import("angle_metric") if not reference_only else None
    for v1, v2, expected in cases:
        ref_val = ref_angle_metric(v1, v2)
        assert abs(ref_val - expected) < 1e-6, f"ref_angle_metric({v1},{v2}) = {ref_val}, expected {expected}"
        if reference_only:
            print(f"  angle({v1}, {v2}) = {ref_val:.6f}  (expected {expected})")
        else:
            py_val = py_fn(v1, v2) if py_fn else None
            check(f"angle_metric{v1}", ref_val, py_val)


def test_contact_heatmaps(reference_only=False):
    """
    Validates _build_contact_heatmaps and anchor heatmap distance scaling.
    Pure Python - no C++ scorer needed.
    """
    print("\n[contact_heatmaps] Singleton contact heatmap building and anchor distance scaling")

    if reference_only:
        print("  (pure Python validation - no C++ reference)")
        return

    _test_names = [
        "contact_heatmaps.sub_shape", "contact_heatmaps.anc_shape",
        "contact_heatmaps.no_singletons_anc", "contact_heatmaps.no_singletons_sub",
        "contact_heatmaps.in_region", "contact_heatmaps.anc_symmetric",
        "contact_heatmaps.out_of_region", "contact_heatmaps.wrong_chr",
        "anchor_heatmap.scales_distances", "anchor_heatmap.disabled",
    ]

    try:
        import numpy as _np
        _root = str(ROOT)
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from src.solver import Solver as _Sv
        from src.io import InteractionArc as _IA
    except ImportError as exc:
        for name in _test_names:
            print(f"  {SKIP}  {name}  ({exc})")
            _results.append(("skip", name))
        return

    class _Cl:
        def __init__(self, start, end):
            self.start = start; self.end = end
            self.genomic_pos = (start + end) // 2; self.orientation = "N"
            self.pos = _np.zeros(3, dtype=_np.float32)
            self.arcs = []; self.children = []; self.parent = -1
            self.is_fixed = False; self.dist_to_next = 0.0

        @property
        def level(self): return 0

    class _St:
        loop_density = 3
        use_anchor_heatmap = True; anchor_heatmap_influence = 0.5
        use_subanchor_heatmap = True; subanchor_heatmap_influence = 0.1
        subanchor_heatmap_dist_weight = 0.01
        subanchor_estimate_replicates = 2; subanchor_estimate_steps = 2

        @staticmethod
        def genomic_length_to_distance(bp):
            return 1.0 + 0.5 * (max(bp, 0) / 1000.0) ** 0.75

    def _sv(ancs, sins):
        sv = _Sv.__new__(_Sv)
        sv.s = _St(); sv.clusters = ancs; sv.arcs = {"chr1": []}
        sv._singletons = sins; sv.chrs = []
        return sv

    ld, n_anc = 3, 3
    N = n_anc + (n_anc - 1) * ld  # 9

    # Bin shape
    sv = _sv([_Cl(0, 100), _Cl(500, 600), _Cl(1200, 1300)], [])
    h_anc, h_sub = sv._build_contact_heatmaps([0, 1, 2], "chr1")
    check("contact_heatmaps.sub_shape", float(N * N), float(h_sub.size), atol=0)
    check("contact_heatmaps.anc_shape", float(n_anc * n_anc), float(h_anc.size), atol=0)

    # No singletons → all zeros
    sv2 = _sv([_Cl(0, 100), _Cl(500, 600)], [])
    h_a2, h_s2 = sv2._build_contact_heatmaps([0, 1], "chr1")
    check("contact_heatmaps.no_singletons_anc", 0.0, float(h_a2.max()), atol=0)
    check("contact_heatmaps.no_singletons_sub", 0.0, float(h_s2.max()), atol=0)

    # In-region singleton raises heatmap and is symmetric
    sv3 = _sv([_Cl(0, 100), _Cl(500, 600)], [("chr1", 50, "chr1", 550, 1)])
    h_a3, _ = sv3._build_contact_heatmaps([0, 1], "chr1")
    check("contact_heatmaps.in_region", 1.0, 1.0 if h_a3.max() > 0.0 else 0.0, atol=0)
    check("contact_heatmaps.anc_symmetric", 1.0, 1.0 if _np.allclose(h_a3, h_a3.T) else 0.0, atol=0)

    # Out-of-region singleton ignored
    sv4 = _sv([_Cl(1000, 1100), _Cl(1500, 1600)], [("chr1", 50, "chr1", 550, 5)])
    h_a4, _ = sv4._build_contact_heatmaps([0, 1], "chr1")
    check("contact_heatmaps.out_of_region", 0.0, float(h_a4.max()), atol=0)

    # Wrong chromosome ignored
    sv5 = _sv([_Cl(0, 100), _Cl(500, 600)], [("chr2", 50, "chr2", 550, 10)])
    h_a5, _ = sv5._build_contact_heatmaps([0, 1], "chr1")
    check("contact_heatmaps.wrong_chr", 0.0, float(h_a5.max()), atol=0)

    # Anchor heatmap scales expected distances: s=(10/10)*0.5=0.5 → 5.0*(1-0.5)=2.5
    h_m = _np.zeros((2, 2)); h_m[0, 1] = h_m[1, 0] = 10.0
    sv6 = _Sv.__new__(_Sv); sv6.s = _St()
    sv6.clusters = [_Cl(0, 100), _Cl(500, 600)]
    arc = _IA(start=0, end=1, score=10)
    sv6.clusters[0].arcs = [0]; sv6.clusters[1].arcs = [0]
    sv6.arcs = {"chr1": [arc]}; sv6.s.freq_to_distance = lambda sc: 5.0
    mat = sv6._calc_anchor_expected_distances([0, 1], "chr1", h_m)
    check("anchor_heatmap.scales_distances", 2.5, float(mat[0, 1]))

    # Anchor heatmap disabled → no scaling
    st7 = _St(); st7.use_anchor_heatmap = False
    sv7 = _Sv.__new__(_Sv); sv7.s = st7
    sv7.clusters = [_Cl(0, 100), _Cl(500, 600)]
    arc7 = _IA(start=0, end=1, score=10)
    sv7.clusters[0].arcs = [0]; sv7.clusters[1].arcs = [0]
    sv7.arcs = {"chr1": [arc7]}; sv7.s.freq_to_distance = lambda sc: 5.0
    mat7 = sv7._calc_anchor_expected_distances([0, 1], "chr1", h_m)
    check("anchor_heatmap.disabled", 5.0, float(mat7[0, 1]))


def test_subanchor_heat(reference_only=False):
    """
    Validates _build_heat_dist_subanchor and mc_smooth heat integration.
    Pure Python - no C++ scorer needed.
    """
    print("\n[subanchor_heat] Subanchor heat distance targets and mc_smooth integration")

    if reference_only:
        print("  (pure Python validation - no C++ reference)")
        return

    _test_names = [
        "subanchor_heat.zero_heatmap_none", "subanchor_heat.high_contact_smaller",
        "subanchor_heat.nonneg_distances", "subanchor_heat.mc_smooth_heat_finite",
        "subanchor_heat.mc_smooth_orn_heat_finite", "subanchor_heat.local_sum_eq_init",
    ]

    try:
        import numpy as _np
        import math as _m
        _root = str(ROOT)
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from src.solver import Solver as _Sv
        from src.mc import mc_smooth, _init_heat_nb, _local_heat_nb, _as_f64
    except ImportError as exc:
        for name in _test_names:
            print(f"  {SKIP}  {name}  ({exc})")
            _results.append(("skip", name))
        return

    class _Cl:
        def __init__(self, start, end):
            self.start = start; self.end = end
            self.genomic_pos = (start + end) // 2; self.orientation = "N"
            self.pos = _np.zeros(3, dtype=_np.float32)
            self.arcs = []; self.children = []; self.parent = -1
            self.is_fixed = False; self.dist_to_next = 0.0

        @property
        def level(self): return 0

    class _St:
        loop_density = 3; use_anchor_heatmap = True; anchor_heatmap_influence = 0.5
        use_subanchor_heatmap = True; subanchor_heatmap_influence = 0.5
        subanchor_heatmap_dist_weight = 0.01
        subanchor_estimate_replicates = 2; subanchor_estimate_steps = 2
        max_temp_smooth = 5.0; dt_temp_smooth = 0.99
        jump_scale_smooth = 50.0; jump_coef_smooth = 20.0
        mc_stop_steps_smooth = 100; mc_stop_improvement_smooth = 0.995
        mc_stop_successes_smooth = 3
        spring_stretch = 0.1; spring_squeeze = 0.1; spring_angular = 0.1
        smooth_dist_weight = 1.0; smooth_angle_weight = 1.0
        noise_smooth = 5.0; use_ctcf_motif = False

        @staticmethod
        def genomic_length_to_distance(bp):
            return 1.0 + 0.5 * (max(bp, 0) / 1000.0) ** 0.75

    n = 9
    ancs = [_Cl(0, 100), _Cl(500, 600), _Cl(1200, 1300)]
    fixed = _np.zeros(n, dtype=bool)
    fixed[0] = fixed[4] = fixed[8] = True
    dtn = _np.full(n - 1, 2.0, dtype=_np.float32)

    def _sv():
        sv = _Sv.__new__(_Sv); sv.s = _St(); sv.clusters = ancs
        sv.arcs = {}; sv._singletons = []; sv.chrs = []
        return sv

    # 1. Zero heatmap → None
    pos1 = _np.random.RandomState(1).rand(n, 3).astype(_np.float32) * 10
    r1 = _sv()._build_heat_dist_subanchor(pos1, fixed, dtn, _np.zeros((n, n)), step_size=0.5)
    check("subanchor_heat.zero_heatmap_none", 1.0, 1.0 if r1 is None else 0.0, atol=0)

    # 2. High-contact pair gets smaller target distance than low-contact pair
    pos2 = _np.zeros((n, 3), dtype=_np.float32)
    for i in range(n):
        pos2[i, 0] = float(i) * 2.0
    h2 = _np.zeros((n, n)); h2[0, 8] = h2[8, 0] = 100.0; h2[0, 4] = h2[4, 0] = 1.0
    heat2 = _sv()._build_heat_dist_subanchor(pos2, fixed, dtn, h2, step_size=0.5)
    check("subanchor_heat.high_contact_smaller", 1.0,
          1.0 if (heat2 is not None and heat2[0, 8] <= heat2[0, 4] + 1e-6) else 0.0, atol=0)
    check("subanchor_heat.nonneg_distances", 1.0,
          1.0 if (heat2 is not None and float(heat2.min()) >= 0.0) else 0.0, atol=0)

    # 3. mc_smooth with heat produces finite non-negative score
    _np.random.seed(42)
    pos3 = _np.random.rand(n, 3).astype(_np.float32) * 5.0
    heat3 = _np.full((n, n), 0.1); _np.fill_diagonal(heat3, 0.0)
    st3 = _St(); st3.mc_stop_steps_smooth = 200; st3.mc_stop_successes_smooth = 2
    st3.mc_stop_improvement_smooth = 0.5
    s3 = mc_smooth(pos3, dtn, fixed, step_size=0.5, settings=st3, heat_dist=heat3)
    check("subanchor_heat.mc_smooth_heat_finite", 1.0,
          1.0 if _m.isfinite(s3) and s3 >= 0.0 else 0.0, atol=0)

    # 4. mc_smooth with heat + orientation produces finite score
    n7 = 7; fixed7 = _np.zeros(n7, dtype=bool)
    fixed7[0] = fixed7[3] = fixed7[6] = True
    dtn7 = _np.full(n7 - 1, 1.0, dtype=_np.float32)
    _np.random.seed(0); pos4 = _np.random.rand(n7, 3).astype(_np.float32) * 3.0
    heat4 = _np.full((n7, n7), 2.0); _np.fill_diagonal(heat4, 0.0)
    st4 = _St(); st4.use_ctcf_motif = True; st4.mc_stop_steps_smooth = 100
    st4.mc_stop_successes_smooth = 2; st4.mc_stop_improvement_smooth = 0.5
    s4 = mc_smooth(pos4, dtn7, fixed7, step_size=0.5, settings=st4,
                   char_orientations=_np.array(['R', 'N', 'N', 'L', 'N', 'N', 'R'], dtype='<U1'),
                   anchor_neighbors={0: [1], 1: [0, 2], 2: [1]},
                   anchor_neighbor_weights={0: [1.0], 1: [1.0, 1.0], 2: [1.0]},
                   heat_dist=heat4)
    check("subanchor_heat.mc_smooth_orn_heat_finite", 1.0,
          1.0 if _m.isfinite(s4) and s4 >= 0.0 else 0.0, atol=0)

    # 5. Numba: sum of _local_heat_nb over all beads == _init_heat_nb (double-count invariant)
    rng = _np.random.RandomState(7); pos5 = rng.rand(5, 3) * 5.0
    hd5 = _np.abs(rng.randn(5, 5)) + 0.5; _np.fill_diagonal(hd5, 0.0)
    hd5 = (hd5 + hd5.T) / 2
    pw5 = _as_f64(pos5); hd64 = _as_f64(hd5)
    total_local = sum(float(_local_heat_nb(pw5, hd64, p, 1.0)) for p in range(5))
    init_val = float(_init_heat_nb(pw5, hd64, 1.0))
    check("subanchor_heat.local_sum_eq_init", init_val, total_local)


# ---------------------------------------------------------------------------
# Summary

def print_summary():
    passes = sum(1 for r, _ in _results if r == "pass")
    fails = sum(1 for r, _ in _results if r == "fail")
    skips = sum(1 for r, _ in _results if r == "skip")
    print(f"\n{'=' * 60}")
    print(f"Results: {passes} passed, {fails} failed, {skips} skipped")
    if fails:
        print("FAILED tests:")
        for r, name in _results:
            if r == "fail":
                print(f"  - {name}")
    return fails == 0


ALL_TESTS = {
    "angle": test_angle,
    "distfns": test_distfns,
    "heatmap": test_heatmap,
    "arcs": test_arcs,
    "smooth": test_smooth,
    "densify": test_densify,
    "metropolis": test_metropolis,
    "orientation": test_orientation,
    "contact_heatmaps": test_contact_heatmaps,
    "subanchor_heat": test_subanchor_heat,
}


def main():
    parser = argparse.ArgumentParser(description="3dgnome-ng correctness harness")
    parser.add_argument("tests", nargs="*", metavar="TEST",
                        help=f"Test groups to run (default: all). Choices: {', '.join(ALL_TESTS)}")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="Force recompile scorer.cpp")
    parser.add_argument("--reference", action="store_true",
                        help="Print C++ reference values only; do not run Python impl")
    args = parser.parse_args()

    build_scorer(force=args.rebuild or args.build_only)
    if args.build_only:
        return

    selected = args.tests if args.tests else list(ALL_TESTS)

    for name in selected:
        if name in ALL_TESTS:
            ALL_TESTS[name](reference_only=args.reference)

    if not args.reference:
        ok = print_summary()
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
