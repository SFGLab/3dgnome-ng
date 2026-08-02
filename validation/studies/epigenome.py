"""Ablate the epigenome energy terms against real Hi-C.

Reconstructs one region several times, each with a different subset of the
compartment and accessibility terms enabled, and reports for each:

  * compartment eigenvector correlation with that cell line's own Hi-C, plus
    Cohen's kappa on the per-bin compartment calls.  This is MultiMM's second
    validation.
  * radius of gyration, bond-length spread and overlap fraction, the polymer
    sanity numbers that must not regress while the compartment score improves.

The terms are purely attractive, so a run that improves compartment agreement by
collapsing the structure has not improved anything.  Reporting both together is
the point of this study.

Needs tracks built first:

    python -m validation fetch  --manifest validation/manifests/<CELL>_hic.json --out data/_hic
    python -m validation fetch  --manifest validation/manifests/<CELL>_accessibility.json \\
                                --out data/_epigenome
    python -m validation tracks --cell <CELL>
    python -m validation epigenome --cell <CELL> --region chr1:20000000-40000000

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gnome3d.types import BedRegion, F64Array, I64Array
from validation.core import config as cfgmod
from validation.core import ensemble as ens_mod
from validation.metrics import hic as contacts
from validation.metrics import structure as smetrics
from validation.studies import Context, Study, register

# Each arm names the flags it turns on, on top of the canonical config.
#
# CANONICAL already enables excluded volume and confinement, so the baseline is
# not a bare polymer and there is no point in an EV-only arm. That matters here:
# the affinity terms are attractive and need that repulsion to push back against.
# Stride between independent ensembles, matching reconstruct.MEMBER_SEED_STRIDE's
# intent: far enough apart that two repeats share no member seeds.
_SEED_STRIDE = 50_000_003

ARMS: dict[str, dict[str, object]] = {
    "off": {},
    "compartments": {"use_compartments": True},
    "bridging": {"use_bridging": True},
    "fibre": {"use_fibre_compaction": True},
    "all": {
        "use_compartments": True,
        "use_bridging": True,
        "use_fibre_compaction": True,
    },
}


def _track_on_bins(comp_path: str, chrom: str, bin_starts: I64Array) -> F64Array:
    """The input compartment track sampled onto the Hi-C bin grid.

    Used to sort bins for the saddle statistic. The same track is used for every
    arm so the quantile definition is fixed, which makes a change in enrichment a
    change in the structure rather than in the binning.
    """
    vals: dict[int, float] = {}
    with open(comp_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4 or parts[0] != chrom:
                continue
            try:
                vals[int(parts[1])] = float(parts[3])
            except ValueError:
                continue
    if not vals:
        return np.zeros(len(bin_starts), dtype=np.float64)
    # A coarser track (H1ESC is 250kb against a 100kb grid) needs the covering
    # interval, not an exact start match, or every bin reads as unassigned.
    keys = np.array(sorted(vals))
    out = np.zeros(len(bin_starts), dtype=np.float64)
    for i, s in enumerate(bin_starts):
        j = int(np.searchsorted(keys, s, side="right")) - 1
        if j >= 0:
            out[i] = vals[int(keys[j])]
    return out


def _signal_on_bins(path: str, chrom: str, bin_starts: I64Array, binsize: int) -> F64Array:
    """A signal bedGraph averaged onto the Hi-C bin grid.

    Averaging rather than sampling, because the accessibility track is finer than
    the contact grid (5 kb against 10 kb or more) and taking one sub-interval per
    bin would discard half the signal and add noise the metric would read as
    structure. Bins with no covering interval stay at zero.
    """
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    lo, hi = int(bin_starts[0]), int(bin_starts[-1]) + binsize
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4 or parts[0] != chrom:
                continue
            try:
                s0, v = int(parts[1]), float(parts[3])
            except ValueError:
                continue
            if s0 < lo or s0 >= hi:
                continue
            b = (s0 - lo) // binsize
            sums[b] = sums.get(b, 0.0) + v
            counts[b] = counts.get(b, 0) + 1
    out = np.zeros(len(bin_starts), dtype=np.float64)
    for b, tot in sums.items():
        if 0 <= b < len(out):
            out[b] = tot / counts[b]
    return out


def _track_paths(cell: str, data_root: str) -> tuple[str, str]:
    """Absolute paths.  `Settings.data_path` joins a relative name onto `data_dir`,
    which for these tracks is already `<data_root>/<cell>`, so a repo-relative path
    would resolve to `data/<cell>/data/<cell>/...` and silently load nothing."""
    d = (Path(data_root) / cell).resolve()
    return (
        str(d / f"{cell}_compartments.bedGraph"),
        str(d / f"{cell}_atac.bedGraph"),
    )


def _ib_ids(settings: object, chrs: list[str], region: object, bin_starts: I64Array) -> I64Array:
    """Which interaction block each Hi-C bin falls in, or -1 for none.

    Built from the coarse cluster tree, so it reflects the blocks the
    reconstruction actually used rather than a re-derived guess. Level 3 is the
    interaction block level.
    """
    from gnome3d.data import ContactData
    from gnome3d.pipeline import coarse as cb

    data = ContactData.from_files(settings, chrs, region)  # pyright: ignore[reportArgumentType]
    st = cb.build_state(settings, data, chrs, region)  # pyright: ignore[reportArgumentType]
    ibs = [c for c in st.clusters if int(c.level) == 3]

    out = np.full(len(bin_starts), -1, dtype=np.int64)
    for k, c in enumerate(ibs):
        hit = (bin_starts >= int(c.start)) & (bin_starts <= int(c.end))
        out[hit] = k
    return out


def _run_arm(
    ctx: Context,
    args: argparse.Namespace,
    flags: dict[str, object],
    chrs: list[str],
    region: BedRegion | None,
    c_obs: F64Array,
    bin_starts: I64Array,
    sort_track: F64Array,
    seed_offset: int = 0,
    block_id: I64Array | None = None,
    acc_track: F64Array | None = None,
    contact_radius: float | None = None,
) -> dict[str, float]:
    """Reconstruct one arm and score it. Returns the metric row.

    Raises rather than returning a partial row: an arm that silently reports the
    baseline because its track failed to load is worse than a visible error.
    """
    from gnome3d.data import ContactData

    s = cfgmod.settings_for_cell(ctx.cell, ctx.data_root, ctx.quality)
    s = cfgmod.apply_flags(s, flags)
    data = ContactData.from_files(s, chrs, region)
    if flags.get("use_compartments") and not data.compartments:
        raise RuntimeError(f"no compartment intervals loaded from {flags.get('data_compartments')}")
    if flags.get("use_bridging") and not data.accessibility:
        raise RuntimeError(f"no accessibility bins loaded from {flags.get('data_accessibility')}")

    ens = ens_mod.run_ensemble(s, data, chrs, region, ctx.n, seed_offset=seed_offset)
    cl, ml = ens_mod.to_arrays_list(ens)

    # The contact radius must be the SAME for every arm. Deriving it per arm from
    # that arm's own median bond length means a term which shortens bonds also
    # shrinks the radius, so its contact map is built at a different effective
    # resolution and every contact metric shifts for that reason alone. Fibre
    # compaction shortens bonds by about a fifth, which is enough to move accE and
    # overlap on its own. The caller passes the baseline's radius; `None` keeps the
    # self-derived value for standalone use.
    own_radius = float(np.median(smetrics.bond_lengths(cl[0])))
    radius = own_radius if contact_radius is None else float(contact_radius)
    c_sim = np.zeros_like(c_obs)
    for coords, mids in zip(cl, ml, strict=True):
        c_sim += contacts.simulated_contacts(coords, mids, bin_starts, args.binsize, radius)
    cc = contacts.compartment_correlation(c_sim, c_obs)
    sad = contacts.compartment_saddle(c_sim, sort_track)

    bl = smetrics.bond_lengths(cl[0])
    # Density of the simulated map. compartment_saddle divides out distance decay, so a
    # map where nearly every bin pair is in contact returns a strength of exactly 1.0 or
    # nan rather than a measurement. Reporting the density makes that visible instead of
    # leaving a saturated run looking like a real null.
    off_diag = c_sim.size - c_sim.shape[0]
    # Does the structure organize by interaction block rather than by data? Only
    # meaningful against the same ratio on the experimental map, which the caller
    # computes once per region.
    ib_ratio = (
        contacts.block_enrichment(c_sim, block_id)["ratio"]
        if block_id is not None
        else float("nan")
    )
    # Saddle sorted by accessibility rather than compartment. Bridging clusters
    # accessible beads, and nothing else measured that directly.
    acc_sad = (
        contacts.compartment_saddle(c_sim, contacts.signed_track(acc_track))["strength"]
        if acc_track is not None
        else float("nan")
    )
    return {
        "ib_ratio": ib_ratio,
        "acc_saddle": acc_sad,
        "saddle": sad["strength"],
        "eig": cc["eig_pearson_abs"],
        "kappa": cc["agreement_kappa"],
        "rg": float(np.mean([smetrics.radius_of_gyration(c) for c in cl])),
        "cv": float(bl.std() / bl.mean()) if bl.mean() > 0 else float("nan"),
        "overlap": float(smetrics.overlap_fraction(cl[0], radius)[0]),
        "density": float((c_sim > 0).sum() - np.count_nonzero(np.diag(c_sim))) / max(off_diag, 1),
        "n_beads": float(len(cl[0])),
        "radius": radius,
        "own_radius": own_radius,
    }


def _arm_flags(
    name: str, args: argparse.Namespace, comp_path: str, acc_path: str
) -> dict[str, object]:
    flags = dict(ARMS[name])
    if flags.get("use_compartments"):
        flags["compartment_weight"] = args.compartment_weight
    if flags.get("use_bridging"):
        flags["bridging_weight"] = args.bridging_weight
    if flags.get("use_fibre_compaction"):
        flags["fibre_compaction"] = args.fibre
    # The tracks are always pointed at; only the flags decide whether a term reads them.
    flags["data_compartments"] = comp_path
    flags["data_accessibility"] = acc_path
    # Applied to every arm including off, so the baseline reads the same track the
    # treated arms do and a difference is the term rather than the normalisation.
    flags["accessibility_mode"] = args.accessibility_mode
    flags["accessibility_percentile"] = args.accessibility_percentile
    return flags


class Epigenome(Study):
    name = "epigenome"
    help = "ablate the compartment and accessibility terms against real Hi-C"

    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--region", default="chr1:20000000-40000000")
        p.add_argument(
            "--binsize", type=int, default=100_000, help="Hi-C resolution for the eigenvector"
        )
        p.add_argument("--arms", default=",".join(ARMS), help="comma list of arm names")
        p.add_argument("--compartment-weight", type=float, default=2.0)
        p.add_argument("--bridging-weight", type=float, default=1.0)
        p.add_argument("--fibre", type=float, default=0.2)
        p.add_argument(
            "--accessibility-mode",
            default="log",
            choices=["log", "binary"],
            help="how the raw ATAC track maps to [0,1]; binary is HiP-HoP's open/closed state",
        )
        p.add_argument("--accessibility-percentile", type=float, default=80.0)
        p.add_argument(
            "--baseline-repeats",
            type=int,
            default=1,
            help="run the off arm this many times to measure the noise floor; "
            "an effect smaller than that spread is not a result",
        )

    def run(self, ctx: Context, args: argparse.Namespace) -> None:
        from gnome3d.io import parse_chrs_arg

        comp_path, acc_path = _track_paths(ctx.cell, ctx.data_root)
        # Only the tracks some selected arm actually reads are required. Demanding
        # both refuses a compartments-only run on a machine that has no
        # accessibility bigWig, which is a precondition the run does not have.
        wanted = [ARMS[a.strip()] for a in args.arms.split(",") if a.strip() in ARMS]
        need_comp = any(f.get("use_compartments") for f in wanted)
        need_acc = any(f.get("use_bridging") or f.get("use_fibre_compaction") for f in wanted)
        required = ([(comp_path, "compartments")] if need_comp else []) + (
            [(acc_path, "accessibility")] if need_acc else []
        )
        # The saddle sorting track is the compartment call, so it is needed for the
        # report even when no arm switches the compartment term on.
        if not need_comp:
            required.append((comp_path, "compartments (for the saddle sorting track)"))
        for path, what in required:
            if not Path(path).exists():
                print(f"[epigenome] missing {what} track: {path}")
                print(f"[epigenome] build it: python -m validation tracks --cell {ctx.cell}")
                return
        if not ctx.hic:
            print("[epigenome] --hic is required; it is the target the arms are scored against")
            return

        chrs, region = parse_chrs_arg(args.region)
        c_obs, bin_starts = contacts.observed_hic(ctx.hic, args.region, args.binsize, balance=True)
        sort_track = _track_on_bins(comp_path, chrs[0], bin_starts)
        obs_saddle = contacts.compartment_saddle(c_obs, sort_track)
        # Interaction-block labels, so every arm reports how block-organized its
        # structure is next to how block-organized the experiment is. A saddle gain
        # that arrives with a rising block ratio is the fragmentation artifact.
        blocks = _ib_ids(
            cfgmod.settings_for_cell(ctx.cell, ctx.data_root, ctx.quality),
            chrs,
            region,
            bin_starts,
        )
        obs_ib = contacts.block_enrichment(c_obs, blocks)["ratio"]
        # Accessibility on the same grid, for the accessibility-sorted saddle.
        acc_track = _signal_on_bins(acc_path, chrs[0], bin_starts, args.binsize)
        obs_acc = contacts.compartment_saddle(c_obs, contacts.signed_track(acc_track))["strength"]

        print(f"epigenome ablation  {ctx.cell}  {args.region}  n={ctx.n}")
        print(f"  compartments: {comp_path}")
        print(f"  accessibility: {acc_path}")
        print(f"  scored against {Path(ctx.hic).name} @ {args.binsize // 1000}kb")
        print(
            f"  experimental saddle = {obs_saddle['strength']:.3f} "
            f"over {int(obs_saddle['n_bins'])} bins  (1.0 = no compartmentalization)\n"
            f"  experimental within-block enrichment = {obs_ib:.3f}\n"
            f"  experimental accessibility saddle    = {obs_acc:.3f}\n"
        )
        header = (
            f"  {'arm':<14}{'saddle':>9}{'accE':>9}{'ibE':>9}{'eig |r|':>9}{'kappa':>8}"
            f"{'Rg':>9}{'bondCV':>9}{'overlap':>9}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))

        # Measuring the detection floor is not optional.  Each repeat must use a
        # different ensemble seed: the pipeline is deterministic, so repeating the
        # same call gives byte-identical structures and a floor of exactly zero,
        # which measures nothing.  (An earlier apparent run-to-run spread of 0.020
        # came from the JAX kernels seeding off `hash()` of a string, which is
        # salted per process - that is a bug, not sampling noise.)
        floor: float | None = None
        if args.baseline_repeats > 1:
            off_flags = _arm_flags("off", args, comp_path, acc_path)
            reps = [
                _run_arm(
                    ctx,
                    args,
                    off_flags,
                    chrs,
                    region,
                    c_obs,
                    bin_starts,
                    sort_track,
                    seed_offset=i * _SEED_STRIDE,
                    block_id=blocks,
                    acc_track=acc_track,
                )
                for i in range(args.baseline_repeats)
            ]
            eigs = [r["eig"] for r in reps]
            sads = [r["saddle"] for r in reps]
            accs = [r["acc_saddle"] for r in reps]
            if len(eigs) > 1:
                floor = float(np.std(eigs, ddof=1))
                sad_floor = float(np.std(sads, ddof=1))
                acc_floor = (
                    float(np.std(accs, ddof=1)) if np.all(np.isfinite(accs)) else float("nan")
                )
                print(
                    f"  {'off x' + str(len(eigs)):<14}"
                    f"{np.mean(sads):>9.3f}{np.mean(eigs):>9.3f}{'':>8}{'':>9}{'':>9}{'':>9}"
                    f"   floor: saddle sd={sad_floor:.3f}  accE sd={acc_floor:.3f}  eig sd={floor:.3f}"
                )

        base: dict[str, float] = {}
        # Every arm is scored at the same contact radius, taken from the baseline,
        # so a term that changes bond length cannot move the contact metrics by
        # changing the map's effective resolution.
        shared_radius: float | None = None
        # `off` must run first, since it supplies that radius. Reordering rather
        # than trusting the caller: an arms list starting with a treated arm would
        # otherwise score that arm at its own radius and the rest at the baseline's,
        # which is the inconsistency this is meant to remove.
        names = [n.strip() for n in args.arms.split(",") if n.strip()]
        if "off" in names and names[0] != "off":
            names = ["off"] + [n for n in names if n != "off"]
            print("  (running `off` first: it supplies the shared contact radius)\n")
        for name in names:
            name = name.strip()
            if name not in ARMS:
                print(f"  {name:<14}  unknown arm")
                continue
            flags = _arm_flags(name, args, comp_path, acc_path)
            try:
                row = _run_arm(
                    ctx,
                    args,
                    flags,
                    chrs,
                    region,
                    c_obs,
                    bin_starts,
                    sort_track,
                    block_id=blocks,
                    acc_track=acc_track,
                    contact_radius=shared_radius,
                )
                if name == "off":
                    base = row
                    shared_radius = row["own_radius"]
                mark = ""
                if base and name != "off":
                    d_sad = row["saddle"] - base["saddle"]
                    d_eig = row["eig"] - base["eig"]
                    d_rg = (row["rg"] - base["rg"]) / base["rg"] if base["rg"] else 0.0
                    mark = f"   saddle {d_sad:+.3f}  eig {d_eig:+.3f}  Rg {d_rg * 100:+.0f}%"
                    if floor:
                        sigma = abs(d_eig) / floor if floor > 0 else 0.0
                        mark += f"  ({sigma:.1f} sd eig)" + ("" if sigma >= 2.0 else " = noise")
                print(
                    f"  {name:<14}{row['saddle']:>9.3f}{row['acc_saddle']:>9.3f}"
                    f"{row['ib_ratio']:>9.3f}"
                    f"{row['eig']:>9.3f}{row['kappa']:>8.3f}"
                    f"{row['rg']:>9.2f}{row['cv']:>9.3f}{row['overlap']:>9.3f}{mark}"
                )
            except Exception as e:  # noqa: BLE001
                print(f"  {name:<14}  ERROR: {type(e).__name__}: {e}")

        print(
            "\n  saddle is the A-A plus B-B contact enrichment over A-B, with bins sorted by the\n"
            "  input compartment track. 1.0 means no compartmentalization. It is the direct\n"
            "  measure of what the compartment term acts on, and unlike the eigenvector\n"
            "  correlation it does not depend on the region's own baseline architecture.\n"
            "  eig |r| is the absolute correlation of the structure's compartment eigenvector\n"
            "  with the experimental one; kappa is Cohen's kappa on the per-bin compartment\n"
            "  calls, which is 0 at chance. The raw same-compartment fraction is not chance\n"
            "  corrected and sits near 0.6 on an unbalanced region even with no signal, so it\n"
            "  is not reported. A gain only counts if Rg and bondCV hold: these terms are\n"
            "  attractive and can raise the score by collapsing the structure.\n"
            "  Pass --baseline-repeats to print the noise floor; an effect under 2 sd of it\n"
            "  is not a result."
        )


register(Epigenome())
