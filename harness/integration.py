#!/usr/bin/env python3
"""
harness/integration.py  —  integration test for 3dgnome-torch.

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

Usage:
    python harness/integration.py              # full test (auto-skips Python)
    python harness/integration.py --cpp-only   # force C++ reference only
    python harness/integration.py -n 5         # ensemble size (default 5)
    python harness/integration.py --keep       # keep temp output files
    python harness/integration.py --fast       # very fast but low-quality MC
    python harness/integration.py --output-dir ./out  # write CIF files to ./out/
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
CPP_BIN = ROOT / "3dnome" / "3dnome"
DATA_DIR = ROOT / "data" / "GM12878"
REGION = "chr5:5418819-7758469"  # 12 non-overlapping anchors, 10 arcs, 2.34 Mb — no NaN in smooth MC
REGION_LABEL = "chr5_5418819_7758469"

# KS test threshold: p-value must exceed this to PASS
KS_P_THRESHOLD = 0.05
# Max KS statistic (distribution similarity) to PASS
KS_D_THRESHOLD = 0.3

PASS_STR = "\033[32mPASS\033[0m"
FAIL_STR = "\033[31mFAIL\033[0m"
SKIP_STR = "\033[33mSKIP\033[0m"


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
delta_temp = 0.9999
stop_condition_improvement_threshold = 0.99
stop_condition_successes_threshold = 50
stop_condition_steps = 50000
"""

def write_config(path: Path, fast: bool) -> None:
    if fast:
        # Very fast: ~5 seconds per structure, low quality
        cfg = BASE_CONFIG.format(
            data_dir=DATA_DIR,
            delta_heatmap=0.995, successes_heatmap=2, steps_heatmap=1000,
            delta_arcs=0.995,    successes_arcs=5,    steps_arcs=1000,
        )
    else:
        # Balanced: ~60 seconds per structure, reasonable quality
        cfg = BASE_CONFIG.format(
            data_dir=DATA_DIR,
            delta_heatmap=0.999, successes_heatmap=5, steps_heatmap=5000,
            delta_arcs=0.999,    successes_arcs=20,   steps_arcs=5000,
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
        _n_arcs    = int(header[1])
        _root      = int(header[2])
        n_factors  = int(header[3])

        # factor names line
        _ = f.readline()

        clusters = []
        for _ in range(n_clusters):
            parts = f.readline().split()
            mid   = int(parts[0])
            # start = int(parts[1])  # not needed
            # end   = int(parts[2])
            x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            n_ch  = int(parts[6])
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
    var = sum((p[0]-cx)**2 + (p[1]-cy)**2 + (p[2]-cz)**2 for p in positions) / n
    return math.sqrt(var)


def pairwise_distances(positions):
    """All unique pairwise distances (i < j)."""
    dists = []
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dz = positions[i][2] - positions[j][2]
            dists.append(math.sqrt(dx*dx + dy*dy + dz*dz))
    return dists


def consecutive_distances(positions):
    """Bond lengths between consecutive beads."""
    dists = []
    for i in range(len(positions) - 1):
        dx = positions[i][0] - positions[i+1][0]
        dy = positions[i][1] - positions[i+1][1]
        dz = positions[i][2] - positions[i+1][2]
        dists.append(math.sqrt(dx*dx + dy*dy + dz*dz))
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


# ---------------------------------------------------------------------------
# Run C++ binary

def run_cpp_ensemble(outdir: Path, config: Path, n: int, max_level: int) -> list:
    """
    Run 3dnome on the test region, produce n structures.
    Returns list of bead-position lists (one per structure).
    """
    cmd = [
        str(CPP_BIN),
        "-a", "create",
        "-s", str(config),
        "-n", REGION_LABEL,
        "-c", REGION,
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
    for line in proc.stdout:
        print(f"[cpp] {line}", end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        sys.exit(f"[cpp] 3dnome exited with code {proc.returncode}")

    # Collect output .hcm files.
    # C++ names: loops_{label}.hcm for n=1, loops_{label}_{i}.hcm for n>1.
    structures = []
    for i in range(n):
        hcm = (outdir / f"loops_{REGION_LABEL}.hcm") if n == 1 else \
              (outdir / f"loops_{REGION_LABEL}_{i}.hcm")
        if not hcm.exists():
            sys.exit(f"[cpp] expected output not found: {hcm}")
        beads = parse_hcm(hcm)
        if not beads:
            sys.exit(f"[cpp] no leaf beads parsed from {hcm}")
        structures.append(beads)

    return structures


# ---------------------------------------------------------------------------
# Try Python reimplementation

def try_python_ensemble(config: Path, n: int) -> list | None:
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
        result = run_region(str(config), REGION, n)
    except NotImplementedError:
        return None

    return result


# ---------------------------------------------------------------------------
# Print ensemble statistics

def print_stats(label: str, structures: list) -> dict:
    """Compute and print summary statistics for an ensemble."""
    rg_vals   = [radius_of_gyration([(x, y, z) for _, x, y, z in s]) for s in structures]
    bond_vals = []
    pwd_vals  = []
    for s in structures:
        pts = [(x, y, z) for _, x, y, z in s]
        bond_vals.extend(consecutive_distances(pts))
        pwd_vals.extend(pairwise_distances(pts))

    rg_m,   rg_s   = mean_std(rg_vals)
    bond_m, bond_s = mean_std(bond_vals)
    pwd_m,  pwd_s  = mean_std(pwd_vals)

    n_beads = len(structures[0])
    print(f"\n  [{label}]  {len(structures)} structures × {n_beads} beads")
    print(f"    Rg:              {rg_m:8.3f} ± {rg_s:.3f}  (radius of gyration)")
    print(f"    bond length:     {bond_m:8.3f} ± {bond_s:.3f}  (consecutive beads)")
    print(f"    mean pairwise d: {pwd_m:8.3f} ± {pwd_s:.3f}  (all bead pairs)")

    return {"rg": rg_vals, "pwd": pwd_vals, "bond": bond_vals}


# ---------------------------------------------------------------------------
# Main

def save_cif_ensemble(structs: list, label: str, outdir: Path) -> None:
    """Write each structure in structs as a CIF file under outdir."""
    sys.path.insert(0, str(ROOT))
    from src.io import write_cif
    outdir.mkdir(parents=True, exist_ok=True)
    for i, beads in enumerate(structs, start=1):
        path = outdir / f"{REGION_LABEL}_{label}_s{i}.cif"
        entry_id = f"{REGION_LABEL}_{label}_s{i}"
        write_cif(str(path), beads, entry_id=entry_id)
    print(f"[integration] {len(structs)} {label} CIF files written to {outdir}/")


def main():
    parser = argparse.ArgumentParser(description="3dgnome-torch integration test")
    parser.add_argument("-n", "--n-structures", type=int, default=5,
                        help="ensemble size (default 5)")
    parser.add_argument("--cpp-only", action="store_true",
                        help="run C++ reference only, skip Python comparison")
    parser.add_argument("--keep", action="store_true",
                        help="keep temp output directory after test")
    parser.add_argument("--fast", action="store_true",
                        help="use very fast (low quality) MC settings (~5s/structure)")
    parser.add_argument("--output-dir", metavar="PATH",
                        help="write output CIF files to this directory (created if needed)")
    args = parser.parse_args()

    if not CPP_BIN.exists():
        sys.exit(f"[error] binary not found: {CPP_BIN}\n  run: make 3dnome")

    if not DATA_DIR.exists():
        sys.exit(f"[error] data directory not found: {DATA_DIR}")

    print(f"[integration] region: {REGION}")
    print(f"[integration] ensemble size: {args.n_structures}")
    print(f"[integration] mode: {'fast' if args.fast else 'balanced'}")

    tmpdir = Path(tempfile.mkdtemp(prefix="gnome3d_integ_"))
    config = tmpdir / "config.ini"
    write_config(config, fast=args.fast)

    # max_level=2 → heatmap + arc reconstruction (anchor-level leaves, no subanchors)
    # this is the level that both C++ and Python produce in phase 1+2
    MAX_LEVEL = 2

    try:
        # -- C++ ensemble --------------------------------------------------
        cpp_outdir = tmpdir / "cpp"
        cpp_outdir.mkdir()
        cpp_structs = run_cpp_ensemble(cpp_outdir, config, args.n_structures, MAX_LEVEL)
        cpp_stats = print_stats("C++", cpp_structs)
        if args.output_dir:
            save_cif_ensemble(cpp_structs, "cpp", Path(args.output_dir))

        # -- Python ensemble -----------------------------------------------
        py_structs = None
        if not args.cpp_only:
            py_structs = try_python_ensemble(config, args.n_structures)

        if py_structs is None:
            print(f"\n  [{SKIP_STR}] Python src/simulate.run_region not implemented — "
                  "skipping comparison")
            print("\n[integration] C++ reference run complete.")
            return

        py_stats = print_stats("Python", py_structs)
        if args.output_dir:
            save_cif_ensemble(py_structs, "python", Path(args.output_dir))

        # -- Compare distributions -----------------------------------------
        print("\n  [comparison]")
        results = []

        # Bead count must match
        n_cpp = len(cpp_structs[0])
        n_py  = len(py_structs[0])
        if n_cpp != n_py:
            print(f"  {FAIL_STR}  bead count mismatch: C++={n_cpp}  Python={n_py}")
            results.append(False)
        else:
            print(f"  {PASS_STR}  bead count matches: {n_cpp}")
            results.append(True)

        # KS test on Rg distribution
        d_rg, p_rg = ks_2samp(cpp_stats["rg"], py_stats["rg"])
        ok_rg = p_rg >= KS_P_THRESHOLD and d_rg <= KS_D_THRESHOLD
        status = PASS_STR if ok_rg else FAIL_STR
        print(f"  {status}  Rg distribution  KS d={d_rg:.3f}  p={p_rg:.3f}")
        results.append(ok_rg)

        # KS test on pooled pairwise distances
        d_pw, p_pw = ks_2samp(cpp_stats["pwd"], py_stats["pwd"])
        ok_pw = p_pw >= KS_P_THRESHOLD and d_pw <= KS_D_THRESHOLD
        status = PASS_STR if ok_pw else FAIL_STR
        print(f"  {status}  pairwise dist KS  d={d_pw:.3f}  p={p_pw:.3f}")
        results.append(ok_pw)

        # KS test on bond lengths
        d_bd, p_bd = ks_2samp(cpp_stats["bond"], py_stats["bond"])
        ok_bd = p_bd >= KS_P_THRESHOLD and d_bd <= KS_D_THRESHOLD
        status = PASS_STR if ok_bd else FAIL_STR
        print(f"  {status}  bond lengths KS   d={d_bd:.3f}  p={p_bd:.3f}")
        results.append(ok_bd)

        all_ok = all(results)
        overall = PASS_STR if all_ok else FAIL_STR
        print(f"\n[integration] {overall}")
        if not all_ok:
            sys.exit(1)

    finally:
        if args.keep:
            print(f"\n[integration] output kept at: {tmpdir}")
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
