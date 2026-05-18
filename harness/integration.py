#!/usr/bin/env python3
"""
harness/integration.py  -  integration test for 3dgnome-ng.

Runs the C++ 3dnome binary on a small chr1 region (~2 Mb, ~34 anchor beads)
to produce an ensemble of structures, then runs the Python reimplementation
on the same region and compares bead-position distributions.

Distributions compared per ensemble:
  - Radius of gyration (Rg) per structure
  - Pooled pairwise inter-bead distances (all i<j pairs, all structures)
  - Consecutive bond lengths along the chain

When both C++ and Python ensembles are available, a 2-sample KS test is used
to decide PASS/FAIL.  When Python is not yet implemented the test prints the
C++ reference statistics and exits 0 (no failure for unimplemented code).

C++ results are automatically cached to out/cpp_cache/ after each run so that
subsequent runs with --python-only skip the (slow) C++ step entirely.

Usage:
    python harness/integration.py              # full test (auto-skips Python)
    python harness/integration.py --cpp-only   # force C++ reference only
    python harness/integration.py --python-only  # skip C++, load cached results
    python harness/integration.py -n 5         # ensemble size (default 5)
    python harness/integration.py --keep       # keep temp output files
    python harness/integration.py --fast       # very fast but low-quality MC
    python harness/integration.py --output-dir ./out  # write CIF files to ./out/
    python harness/integration.py --cache-dir ./my_cache  # override cache location
    python harness/integration.py --region-override chr1:...  # override test region
    python harness/integration.py --with-orientation  # enable CTCF motif orientation
"""

import argparse
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
CPP_BIN = ROOT / "3dnome" / "3dnome"
DATA_DIR = ROOT / "data" / "GM12878"
REGION = "chr1:18288319-20307135"
REGION_LABEL = f"integration_test_region_{REGION.replace(':', '_').replace('-', '_')}"

# KS test threshold: p-value must exceed this to PASS
KS_P_THRESHOLD = 0.05
# Max KS statistic (distribution similarity) to PASS
KS_D_THRESHOLD = 0.3
# Cap pairwise distance pool before KS
KS_PWD_MAX_SAMPLES = 50_000
# Structural distance benchmark (from cudaMMC_benchmark_analysis.ipynb):
# median inter-model structural distance ratio (Python/C++) must be within this
# fraction of 1.0.  Mirrors the notebook's median ratio check.
STRUCT_DIST_RATIO_THRESHOLD = 0.10

PASS_STR = "\033[32mPASS\033[0m"
FAIL_STR = "\033[31mFAIL\033[0m"
SKIP_STR = "\033[33mSKIP\033[0m"

# ---------------------------------------------------------------------------
# Milestone capture

# Matches both C++ and Python milestone lines:
#   "    step   13000  score=11.6605  ratio=1.0000  ok=9/1000  [done]"
_MS_RE = re.compile(r'step\s+([\d,]+)\s+score=([^\s]+)\s+ratio=([^\s]+)\s+ok=(\d+)/(\d+)')
# Matches C++ IB header lines (raw subprocess output, no "[cpp]" prefix):
#   "  chr1 1/2"
_IB_RE = re.compile(r'^\s+\S+\s+(\d+/\d+)\s*$')
# Matches Python milestone label brackets: "[chr1 IB 1/2 run 1/1]" / "smooth"
_PY_LBL = re.compile(r'\[(?:\S+\s+)?IB\s+(\d+/\d+)\s+(run|smooth)')


class _TeeOut:
    """Write to real stdout AND capture to an internal buffer simultaneously."""

    def __init__(self):
        self._real = sys.stdout
        self._buf = io.StringIO()

    def write(self, s):
        self._real.write(s)
        self._buf.write(s)

    def flush(self):
        self._real.flush()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *_):
        sys.stdout = self._real

    def getvalue(self):
        return self._buf.getvalue()

    def __getattr__(self, name):
        return getattr(self._real, name)


def _parse_cpp_milestones(raw_lines: list) -> dict:
    """
    Parse raw C++ subprocess lines (no "[cpp]" prefix) into
    {(ib, phase): [(step, score, ok, total, done), ...]}.
    Arc phase milestones come first; after arc [done] the remaining
    milestones for that IB belong to the smooth phase.
    """
    result = defaultdict(list)
    cur_ib = None
    arc_done = False
    for line in raw_lines:
        m_ib = _IB_RE.match(line)
        if m_ib:
            cur_ib = m_ib.group(1)
            arc_done = False
            continue
        m = _MS_RE.search(line)
        if m and cur_ib is not None:
            phase = "smooth" if arc_done else "arc"
            step = int(m.group(1).replace(',', ''))
            score = float(m.group(2))
            ok = int(m.group(4))
            total = int(m.group(5))
            done = "[done]" in line
            result[(cur_ib, phase)].append((step, score, ok, total, done))
            if done and phase == "arc":
                arc_done = True
    return dict(result)


def _parse_py_milestones(text: str) -> dict:
    """
    Parse captured Python stdout into
    {(ib, phase): [(step, score, ok, total, done), ...]}.
    """
    result = defaultdict(list)
    for line in text.splitlines():
        lm = _PY_LBL.search(line)
        m = _MS_RE.search(line)
        if lm and m:
            ib = lm.group(1)
            phase = "arc" if lm.group(2) == "run" else "smooth"
            step = int(m.group(1).replace(',', ''))
            score = float(m.group(2))
            ok = int(m.group(4))
            total = int(m.group(5))
            done = "[done]" in line
            result[(ib, phase)].append((step, score, ok, total, done))
    return dict(result)


def _merge_milestones(all_data: list) -> dict:
    """
    Average milestone metrics across multiple structures.
    Input:  list of {(ib, phase): [(step, score, ok, total, done), ...]}
    Output: {(ib, phase): [(step, avg_score, avg_ok, total, any_done), ...]}
    """
    bucket = defaultdict(lambda: defaultdict(list))
    for data in all_data:
        for (ib, phase), rows in data.items():
            for step, score, ok, total, done in rows:
                bucket[(ib, phase)][step].append((score, ok, total, done))
    result = {}
    for (ib, phase), by_step in bucket.items():
        rows = []
        for step in sorted(by_step):
            vals = by_step[step]
            rows.append((
                step,
                sum(v[0] for v in vals) / len(vals),
                sum(v[1] for v in vals) / len(vals),
                vals[0][2],
                any(v[3] for v in vals),
            ))
        result[(ib, phase)] = rows
    return result


def print_step_comparison(cpp_ms: dict, py_ms: dict, n_structs: int) -> None:
    """Print a side-by-side per-milestone convergence table."""
    all_keys = sorted(set(cpp_ms) | set(py_ms),
                      key=lambda k: (k[0], 0 if k[1] == "arc" else 1))
    if not all_keys:
        return

    avg_note = f"avg over {n_structs} structure{'s' if n_structs > 1 else ''}"
    print(f"\n{'=' * 74}")
    print(f"  Step-by-step convergence  ({avg_note})")
    print(f"{'=' * 74}")

    for ib, phase in all_keys:
        cpp_rows = {r[0]: r for r in cpp_ms.get((ib, phase), [])}
        py_rows = {r[0]: r for r in py_ms.get((ib, phase), [])}
        steps = sorted(set(cpp_rows) | set(py_rows))
        if not steps:
            continue

        print(f"\n  IB {ib} - {phase} MC")
        print(f"  {'step':>8}  {'C++ score':>11}  {'Py score':>11}  "
              f"{'Δ%':>7}  {'C++ ok/N':>13}  {'Py ok/N':>13}")
        print("  " + "─" * 72)

        for step in steps:
            cr = cpp_rows.get(step)
            pr = py_rows.get(step)

            cs = f"{cr[1]:11.4f}" if cr else f"{'-':>11}"
            ps = f"{pr[1]:11.4f}" if pr else f"{'-':>11}"

            if cr and pr and pr[1] > 1e-9:
                pct = (cr[1] - pr[1]) / pr[1] * 100
                ds = f"{pct:+7.1f}%"
            else:
                ds = f"{'-':>8}"

            cok = f"{cr[2]:>6.0f}/{cr[3]}" if cr else f"{'-':>13}"
            pok = f"{pr[2]:>6.0f}/{pr[3]}" if pr else f"{'-':>13}"

            tag = "  [done]" if ((cr and cr[4]) or (pr and pr[4])) else ""
            print(f"  {step:>8,}  {cs}  {ps}  {ds}  {cok}  {pok}{tag}")


# ---------------------------------------------------------------------------
# Config generation

BASE_CONFIG = """\
[main]
output_level = 2
random_walk = no
loop_density = 5
use_2D = no
max_pet_length = 1000000
long_pet_power = 2.0
long_pet_scale = 1.0
steps_lvl1 = 1
steps_lvl2 = 1
steps_arcs = 1
steps_smooth = 1
noise_lvl1 = 0.5
noise_lvl2 = 0.5
noise_arcs = 0.01
noise_smooth = 5.0

[data]
data_dir = {data_dir}/
anchors = GM12878_anchors_3+_oriented.bed
clusters = GM12878_clusters_3+.bedpe
factors = CTCF
singletons = GM12878_singletons_lessthan3.bedpe
split_singleton_files_by_chr = no
singletons_inter =
segment_split = {data_dir}/ccds_all_hg38_merged100k_GM12878.breakpoints.bed
centromeres = {data_dir}/hg38_centromeres.bed

[distance]
genomic_dist_power = 0.75
genomic_dist_scale = 0.5
genomic_dist_base = 1.0
freq_dist_scale = 25.0
freq_dist_power = -0.6
freq_dist_scale_inter = 120.0
freq_dist_power_inter = -1.0
count_dist_a = 0.2
count_dist_scale = 1.8
count_dist_shift = 8
count_dist_base_level = 0.2

[template]
template_scale = 7.0
dist_heatmap_scale = 15.0

[motif_orientation]
use_motif_orientation = no
weight = 50.0

[anchor_heatmap]
use_anchor_heatmap = no
heatmap_influence = 0.5

[subanchor_heatmap]
use_subanchor_heatmap = no
estimate_distances_steps = 4
estimate_distances_replicates = 4
heatmap_influence = 0.1
heatmap_dist_weight = 0.01

[heatmaps]
inter_scaling = 1.0
distance_heatmap_stretching = 2.5

[springs]
stretch_constant = 0.1
squeeze_constant = 0.1
angular_constant = 0.1
stretch_constant_arcs = 1.0
squeeze_constant_arcs = 1.0

[simulation_heatmap]
max_temp_heatmap = 5.0
delta_temp_heatmap = {delta_heatmap}
jump_temp_scale_heatmap = 50.0
jump_temp_coef_heatmap = 20.0
stop_condition_improvement_threshold_heatmap = 0.99
stop_condition_successes_threshold_heatmap = {successes_heatmap}
stop_condition_steps_heatmap = {steps_heatmap}

[simulation_arcs]
max_temp = 5.0
jump_temp_scale = 50.0
jump_temp_coef = 20.0
delta_temp = {delta_arcs}
stop_condition_improvement_threshold = 0.975
stop_condition_successes_threshold = {successes_arcs}
stop_condition_steps = {steps_arcs}

[simulation_arcs_smooth]
dist_weight = 1.0
angle_weight = 1.0
max_temp = 5.0
jump_temp_scale = 50.0
jump_temp_coef = 20.0
delta_temp = {delta_smooth}
stop_condition_improvement_threshold = 0.99
stop_condition_successes_threshold = {successes_smooth}
stop_condition_steps = {steps_smooth}
"""


def write_config(path: Path, fast: bool, use_orientation: bool = False) -> None:
    if fast:
        # Very fast: ~10 s per structure, low quality
        cfg = BASE_CONFIG.format(
            data_dir=DATA_DIR,
            delta_heatmap=0.995, successes_heatmap=2, steps_heatmap=1000,
            delta_arcs=0.995, successes_arcs=5, steps_arcs=1000,
            delta_smooth=0.995, successes_smooth=5, steps_smooth=1000,
        )
    else:
        # Balanced: ~2 min per structure, reasonable quality
        cfg = BASE_CONFIG.format(
            data_dir=DATA_DIR,
            delta_heatmap=0.999, successes_heatmap=5, steps_heatmap=5000,
            delta_arcs=0.999, successes_arcs=20, steps_arcs=5000,
            delta_smooth=0.999, successes_smooth=20, steps_smooth=5000,
        )
    if use_orientation:
        cfg = cfg.replace(
            "use_motif_orientation = no",
            "use_motif_orientation = yes",
        )
    path.write_text(cfg)


# ---------------------------------------------------------------------------
# HCM file parser

def parse_hcm(hcm_path: Path):
    """
    Parse a .hcm model file and return sorted leaf-bead positions.

    Returns:
        list of (midpoint_bp, x, y, z) tuples sorted by genomic midpoint.

    The .hcm format (toFilePreviousFormat):
        line 1:  n_clusters  n_arcs  root_index  n_factors
        line 2:  factor names (whitespace-separated)
        lines 3+: midpoint start end x y z n_children [child_idx ...]
        arc lines: start end score factor gen_start gen_end 0 0
    """
    with open(hcm_path) as f:
        header = f.readline().split()
        n_clusters = int(header[0])
        _n_arcs = int(header[1])
        _root = int(header[2])
        n_factors = int(header[3])

        # factor names line
        _ = f.readline()

        clusters = []
        for _ in range(n_clusters):
            parts = f.readline().split()
            mid = int(parts[0])
            # start = int(parts[1])  # not needed
            # end   = int(parts[2])
            x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            n_ch = int(parts[6])
            clusters.append((mid, x, y, z, n_ch))

    # Leaf beads: clusters with no children, sorted by genomic midpoint
    leaves = [(mid, x, y, z) for mid, x, y, z, n_ch in clusters if n_ch == 0]
    leaves.sort(key=lambda b: b[0])
    return leaves


# ---------------------------------------------------------------------------
# Distribution statistics

def radius_of_gyration(positions):
    """Rg = sqrt(mean(||r_i - r_cm||^2))."""
    n = len(positions)
    cx = sum(p[0] for p in positions) / n
    cy = sum(p[1] for p in positions) / n
    cz = sum(p[2] for p in positions) / n
    var = sum((p[0] - cx) ** 2 + (p[1] - cy) ** 2 + (p[2] - cz) ** 2 for p in positions) / n
    return math.sqrt(var)


def pairwise_distances(positions):
    """All unique pairwise distances (i < j)."""
    dists = []
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dz = positions[i][2] - positions[j][2]
            dists.append(math.sqrt(dx * dx + dy * dy + dz * dz))
    return dists


def consecutive_distances(positions):
    """Bond lengths between consecutive beads."""
    dists = []
    for i in range(len(positions) - 1):
        dx = positions[i][0] - positions[i + 1][0]
        dy = positions[i][1] - positions[i + 1][1]
        dz = positions[i][2] - positions[i + 1][2]
        dists.append(math.sqrt(dx * dx + dy * dy + dz * dz))
    return dists


def mean_std(values):
    if not values:
        return float("nan"), float("nan")
    m = sum(values) / len(values)
    v = sum((x - m) ** 2 for x in values) / len(values)
    return m, math.sqrt(v)


def ks_2samp(a, b):
    """
    2-sample KS test. Returns (statistic, p_value).
    p_value is approximated via the KS distribution formula.
    Falls back to scipy if available (more accurate for small samples).
    """
    try:
        from scipy.stats import ks_2samp as scipy_ks
        return scipy_ks(a, b)
    except ImportError:
        pass

    # Manual KS: compare ECDFs
    a_sorted = sorted(a)
    b_sorted = sorted(b)
    na, nb = len(a_sorted), len(b_sorted)
    all_vals = sorted(set(a_sorted + b_sorted))
    d = 0.0
    ia = ib = 0
    for v in all_vals:
        while ia < na and a_sorted[ia] <= v:
            ia += 1
        while ib < nb and b_sorted[ib] <= v:
            ib += 1
        d = max(d, abs(ia / na - ib / nb))

    # Asymptotic p-value approximation (Kolmogorov distribution)
    en = math.sqrt(na * nb / (na + nb))
    t = (en + 0.12 + 0.11 / en) * d
    # P(D > t) ≈ 2 * sum_{k=1}^{inf} (-1)^{k-1} * exp(-2k^2t^2)
    p = 0.0
    for k in range(1, 100):
        term = ((-1) ** (k - 1)) * math.exp(-2 * k * k * t * t)
        p += term
        if abs(term) < 1e-8:
            break
    p = max(0.0, min(1.0, 2 * p))
    return d, p


def _subsample(values: list, n: int, seed: int = 42) -> list:
    """Return a random subsample of at most n elements (deterministic)."""
    import random as _rnd
    if len(values) <= n:
        return values
    rng = _rnd.Random(seed)
    return rng.sample(values, n)


# ---------------------------------------------------------------------------
# Run C++ binary

def run_cpp_ensemble(outdir: Path, config: Path, n: int, max_level: int,
                     region: str, region_label: str) -> list:
    """
    Run 3dnome on the test region, produce n structures.
    Returns list of bead-position lists (one per structure).
    """
    cmd = [
        str(CPP_BIN),
        "-a", "create",
        "-s", str(config),
        "-n", region_label,
        "-c", region,
        "-o", str(outdir) + "/",
        "-m", str(n),
        "-v", str(max_level),
    ]
    print(f"[cpp] running: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    raw_lines = []
    for line in proc.stdout:
        print(f"[cpp] {line}", end="", flush=True)
        raw_lines.append(line.rstrip("\n"))
    proc.wait()
    if proc.returncode != 0:
        sys.exit(f"[cpp] 3dnome exited with code {proc.returncode}")

    # Collect output .hcm files.
    # C++ names: loops_{label}.hcm for n=1, loops_{label}_{i}.hcm for n>1.
    structures = []
    for i in range(n):
        hcm = (outdir / f"loops_{region_label}.hcm") if n == 1 else \
            (outdir / f"loops_{region_label}_{i}.hcm")
        if not hcm.exists():
            sys.exit(f"[cpp] expected output not found: {hcm}")
        beads = parse_hcm(hcm)
        if not beads:
            sys.exit(f"[cpp] no leaf beads parsed from {hcm}")
        structures.append(beads)

    return structures, raw_lines


# ---------------------------------------------------------------------------
# Try Python reimplementation

def try_python_ensemble(config: Path, n: int, region: str) -> list | None:
    """
    Call src.simulate.run_region if available.
    Expected signature:
        run_region(config_path: str, region: str, n_structures: int)
            -> list of list[(midpoint_bp, x, y, z)]
    Returns None if not yet implemented.
    """
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from simulate import run_region
    except (ImportError, ModuleNotFoundError):
        return None

    try:
        result = run_region(str(config), region, n)
    except NotImplementedError:
        return None

    return result


# ---------------------------------------------------------------------------
# Print ensemble statistics

def print_stats(label: str, structures: list) -> dict:
    """Compute and print summary statistics for an ensemble."""
    rg_vals = [radius_of_gyration([(x, y, z) for _, x, y, z in s]) for s in structures]
    bond_vals = []
    pwd_vals = []
    for s in structures:
        pts = [(x, y, z) for _, x, y, z in s]
        bond_vals.extend(consecutive_distances(pts))
        pwd_vals.extend(pairwise_distances(pts))

    rg_m, rg_s = mean_std(rg_vals)
    bond_m, bond_s = mean_std(bond_vals)
    pwd_m, pwd_s = mean_std(pwd_vals)

    n_beads = len(structures[0])
    print(f"\n  [{label}]  {len(structures)} structures × {n_beads} beads")
    print(f"    Rg:              {rg_m:8.3f} ± {rg_s:.3f}  (radius of gyration)")
    print(f"    bond length:     {bond_m:8.3f} ± {bond_s:.3f}  (consecutive beads)")
    print(f"    mean pairwise d: {pwd_m:8.3f} ± {pwd_s:.3f}  (all bead pairs)")

    return {"rg": rg_vals, "pwd": pwd_vals, "bond": bond_vals}


def structural_distance_matrix(structures: list) -> list:
    """
    Compute off-diagonal inter-model structural distances for an ensemble.

    For each pair (i, j) of structures the structural distance is:
        1 - Pearson(triu_pairwise_dists_i, triu_pairwise_dists_j)

    This mirrors the similarity matrix comparison in cudaMMC_benchmark_analysis.ipynb
    (cells 42-51) and the structural_distances_lvl2.heat files produced by
    `3dnome -a ensemble`.  Values near 0 = highly similar; near 2 = anti-correlated.

    Returns all n*(n-1) off-diagonal values (i != j) as a flat list.
    """
    try:
        import numpy as np
    except ImportError:
        return []

    n = len(structures)
    # Build (n, k) matrix where k = M*(M-1)/2 pairwise distances per structure
    rows = []
    for s in structures:
        pts = np.array([(x, y, z) for _, x, y, z in s], dtype=np.float64)
        diff = pts[:, None, :] - pts[None, :, :]  # (M, M, 3)
        d = np.sqrt((diff * diff).sum(axis=2))  # (M, M)
        idx = np.triu_indices(len(pts), k=1)
        rows.append(d[idx])
    mat = np.array(rows)  # (n, k)

    # Pearson correlation between every pair of rows
    m = mat.mean(axis=1, keepdims=True)
    s = mat.std(axis=1, keepdims=True)
    s[s < 1e-12] = 1.0
    z = (mat - m) / s  # z-scored rows
    k = mat.shape[1]
    corr = (z @ z.T) / k  # (n, n)

    dist = 1.0 - corr  # structural distance matrix
    return [float(dist[i, j]) for i in range(n) for j in range(n) if i != j]


# ---------------------------------------------------------------------------
# Main

def save_cif_ensemble(structs: list, label: str, outdir: Path, region_label: str) -> None:
    """Write each structure in structs as a CIF file under outdir."""
    sys.path.insert(0, str(ROOT))
    from src.io import write_cif
    outdir.mkdir(parents=True, exist_ok=True)
    for i, beads in enumerate(structs, start=1):
        path = outdir / f"{region_label}_{label}_s{i}.cif"
        entry_id = f"{region_label}_{label}_s{i}"
        write_cif(str(path), beads, entry_id=entry_id)
    print(f"[integration] {len(structs)} {label} CIF files written to {outdir}/")


def _cache_path(cache_dir: Path, region_label: str, n: int) -> Path:
    return cache_dir / f"{region_label}_n{n}.json"


def _save_cache(path: Path, region: str, n: int, fast: bool,
                structs: list, raw_lines: list,
                use_orientation: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "region": region,
            "n_structures": n,
            "mode": "fast" if fast else "balanced",
            "use_orientation": use_orientation,
            "structures": [[list(b) for b in s] for s in structs],
            "raw_lines": raw_lines,
        }, f)
    print(f"[integration] C++ results cached to {path}")


def _load_cache(path: Path) -> tuple:
    with open(path) as f:
        cached = json.load(f)
    structs = [[(b[0], b[1], b[2], b[3]) for b in s] for s in cached["structures"]]
    return structs, cached["raw_lines"]


def main():
    parser = argparse.ArgumentParser(description="3dgnome-ng integration test")
    parser.add_argument("-n", "--n-structures", type=int, default=5,
                        help="ensemble size (default 5)")
    parser.add_argument("--cpp-only", action="store_true",
                        help="run C++ reference only, skip Python comparison")
    parser.add_argument("--python-only", action="store_true",
                        help="skip C++ run; load cached C++ results (requires a prior full run)")
    parser.add_argument("--cache-dir", metavar="PATH", default=None,
                        help="directory for C++ result cache (default: out/cpp_cache)")
    parser.add_argument("--keep", action="store_true",
                        help="keep temp output directory after test")
    parser.add_argument("--fast", action="store_true",
                        help="use very fast (low quality) MC settings (~5s/structure)")
    parser.add_argument("--output-dir", metavar="PATH",
                        help="write output CIF files to this directory (created if needed)")
    parser.add_argument("--region-override", metavar="REGION",
                        help="override test region (must be in data dir and match config)")
    parser.add_argument("--with-orientation", action="store_true",
                        help="enable CTCF motif orientation energy (use_motif_orientation=yes); "
                             "uses a separate cache slot from the no-orientation run")
    args = parser.parse_args()

    if args.python_only and args.cpp_only:
        sys.exit("[error] --python-only and --cpp-only are mutually exclusive")

    if not args.python_only and not CPP_BIN.exists():
        sys.exit(f"[error] binary not found: {CPP_BIN}\n  run: make 3dnome")

    if not DATA_DIR.exists():
        sys.exit(f"[error] data directory not found: {DATA_DIR}")

    if args.region_override:
        region = args.region_override
        region_label = f"integration_test_region_{region.replace(':', '_').replace('-', '_')}"
        print(f"[integration] overriding test region with: {region}")
    else:
        region = REGION
        region_label = REGION_LABEL

    use_orn = getattr(args, "with_orientation", False)

    # Parametrize region_label with mode and orientation so cache files, CIF
    # outputs, and any other label-derived paths are automatically distinct.
    _mode_suffix = "fast" if args.fast else "balanced"
    _orn_suffix = "_orientation" if use_orn else ""
    region_label = f"{region_label}_{_mode_suffix}{_orn_suffix}"

    cache_dir = Path(args.cache_dir) if args.cache_dir else ROOT / "out" / "cpp_cache"
    cache_file = _cache_path(cache_dir, region_label, args.n_structures)

    print(f"[integration] region: {region}")
    print(f"[integration] ensemble size: {args.n_structures}")
    print(f"[integration] mode: {'fast' if args.fast else 'balanced'}")
    if use_orn:
        print(f"[integration] CTCF motif orientation: ENABLED")

    tmpdir = Path(tempfile.mkdtemp(prefix="gnome3d_integ_"))
    config = tmpdir / "config.ini"
    write_config(config, fast=args.fast, use_orientation=use_orn)

    # max_level=2 -> heatmap MC + arc MC + smooth MC (Level 4 runs inside arc reconstruction).
    # Both C++ and Python produce subanchor beads: n_anchors + (n_anchors-1)*loop_density.
    MAX_LEVEL = 2

    try:
        # -- C++ ensemble --------------------------------------------------
        if args.python_only:
            if not cache_file.exists():
                sys.exit(
                    f"[error] no cached C++ results found: {cache_file}\n"
                    f"  run without --python-only first to generate the cache"
                )
            cpp_structs, cpp_raw = _load_cache(cache_file)
            print(f"[integration] loaded {len(cpp_structs)} cached C++ structures from {cache_file}")
        else:
            cpp_outdir = tmpdir / "cpp"
            cpp_outdir.mkdir()
            cpp_structs, cpp_raw = run_cpp_ensemble(
                cpp_outdir, config, args.n_structures, MAX_LEVEL, region, region_label)
            _save_cache(cache_file, region, args.n_structures, args.fast,
                        cpp_structs, cpp_raw, use_orientation=use_orn)

        if args.output_dir:
            save_cif_ensemble(cpp_structs, "cpp", Path(args.output_dir), region_label)
        cpp_ms_raw = _parse_cpp_milestones(cpp_raw)

        # -- Python ensemble -----------------------------------------------
        py_structs = None
        py_ms_raw = {}
        if not args.cpp_only:
            with _TeeOut() as tee:
                py_structs = try_python_ensemble(config, args.n_structures, region)
            py_ms_raw = _parse_py_milestones(tee.getvalue())

        if py_structs is None:
            print(f"\n  [{SKIP_STR}] Python src/simulate.run_region not implemented - "
                  "skipping comparison")
            print("\n[integration] C++ reference run complete.")
            return

        if args.output_dir:
            save_cif_ensemble(py_structs, "python", Path(args.output_dir), region_label)

        # -- Compare distributions -----------------------------------------
        print("\n  [comparison]")
        results = []

        # Bead count must match
        n_cpp = len(cpp_structs[0])
        n_py = len(py_structs[0])
        if n_cpp != n_py:
            print(f"  {FAIL_STR}  bead count mismatch: C++={n_cpp}  Python={n_py}")
            results.append(False)
        else:
            print(f"  {PASS_STR}  bead count matches: {n_cpp} (anchors + subanchors)")
            results.append(True)

        cpp_stats = print_stats("C++", cpp_structs)
        py_stats = print_stats("Python", py_structs)

        # KS test on Rg distribution
        d_rg, p_rg = ks_2samp(cpp_stats["rg"], py_stats["rg"])
        ok_rg = p_rg >= KS_P_THRESHOLD and d_rg <= KS_D_THRESHOLD
        status = PASS_STR if ok_rg else FAIL_STR
        print(f"  {status}  Rg distribution  KS d={d_rg:.3f}  p={p_rg:.3f}")
        results.append(ok_rg)

        # KS test on pooled pairwise distances (subsampled - raw pool is ~18M
        # non-independent values that inflate KS power far beyond physical meaning)
        cpp_pwd = _subsample(cpp_stats["pwd"], KS_PWD_MAX_SAMPLES)
        py_pwd = _subsample(py_stats["pwd"], KS_PWD_MAX_SAMPLES)
        d_pw, p_pw = ks_2samp(cpp_pwd, py_pwd)
        ok_pw = p_pw >= KS_P_THRESHOLD and d_pw <= KS_D_THRESHOLD
        status = PASS_STR if ok_pw else FAIL_STR
        print(f"  {status}  pairwise dist KS  d={d_pw:.3f}  p={p_pw:.3f}"
              f"  (subsampled {len(cpp_pwd):,}/{len(cpp_stats['pwd']):,})")
        results.append(ok_pw)

        # KS test on bond lengths
        d_bd, p_bd = ks_2samp(cpp_stats["bond"], py_stats["bond"])
        ok_bd = p_bd >= KS_P_THRESHOLD and d_bd <= KS_D_THRESHOLD
        status = PASS_STR if ok_bd else FAIL_STR
        print(f"  {status}  bond lengths KS   d={d_bd:.3f}  p={p_bd:.3f}")
        results.append(ok_bd)

        # Structural distance benchmark (from cudaMMC_benchmark_analysis.ipynb)
        # Mirrors the notebook's median ratio check (cells 50-51): compute the
        # inter-model structural distance matrix for each ensemble and compare
        # medians.  KS is intentionally not used - the n*(n-1) off-diagonal values
        # are all correlated (each structure appears in n-1 pairs), so the test
        # would be massively overpowered.
        cpp_sd = structural_distance_matrix(cpp_structs)
        py_sd = structural_distance_matrix(py_structs)
        if cpp_sd and py_sd:
            cpp_med = sorted(cpp_sd)[len(cpp_sd) // 2]
            py_med = sorted(py_sd)[len(py_sd) // 2]
            ratio = py_med / cpp_med if cpp_med > 1e-9 else float("nan")
            ok_sd = math.isfinite(ratio) and abs(ratio - 1.0) <= STRUCT_DIST_RATIO_THRESHOLD
            status = PASS_STR if ok_sd else FAIL_STR
            print(f"  {status}  struct diversity  median ratio={ratio:.3f}"
                  f"  (C++ med={cpp_med:.4f}  Py med={py_med:.4f}"
                  f"  threshold=±{STRUCT_DIST_RATIO_THRESHOLD:.0%})")
            results.append(ok_sd)

        all_ok = all(results)
        overall = PASS_STR if all_ok else FAIL_STR
        print(f"\n[integration] {overall}")

        # -- Per-step convergence comparison --------------------------------
        cpp_ms = _merge_milestones([cpp_ms_raw])
        py_ms = _merge_milestones([py_ms_raw])
        print_step_comparison(cpp_ms, py_ms, args.n_structures)

        if not all_ok:
            sys.exit(1)

    finally:
        if args.keep:
            print(f"\n[integration] output kept at: {tmpdir}")
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
