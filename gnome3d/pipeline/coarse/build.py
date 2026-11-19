"""`CoarseState` and the functions that build heatmaps, distance targets and positions from it.

`clusters` is shared and the coarse stages write `.pos` into it in place. The positioning
functions and `interpolate_children_linear` draw from the global RNG and must run in a fixed
order.
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

from gnome3d import log

if TYPE_CHECKING:
    from gnome3d.polymer import PolymerLaw
from gnome3d.anchor_contacts import AnchorContactMap, dense_counts
from gnome3d.data import ContactData
from gnome3d.hierarchy import Cluster, Level, build_cluster_tree, set_level
from gnome3d.io import create_singleton_heatmap
from gnome3d.pipeline.coarse.heatmap import (
    create_distance_heatmap,
    get_diagonal_size,
    normalize_heatmap,
    normalize_heatmap_diagonal_total,
    normalize_heatmap_inter,
)
from gnome3d.settings import Settings
from gnome3d.tracks import bin_compartments
from gnome3d.types import *
from gnome3d.util import random_vector_np, seed_rng

LOG = log.get("coarse")

# The base seed of the coarse stages. `seed_offset` shifts it per ensemble member.
COARSE_SEED: int = 0xC0FFEE


def seed_global_rng(seed_offset: int = 0) -> int:
    """Seeds the global RNGs, Python `random` and numba, for the coarse stages and returns the seed.

    The coarse stages seed once here and draw from the one stream in order.
    """
    from gnome3d.mc import numba as mc_numba

    coarse_seed = (COARSE_SEED + seed_offset) & 0x7FFFFFFF
    seed_rng(coarse_seed)
    mc_numba.seed_numba(coarse_seed)
    return coarse_seed


@dataclass(frozen=True)
class CoarseState:
    """The cluster tree and the contact data that position it.

    The fields are fixed. The clusters mutate in place as the coarse stages write `.pos`.
    """

    s: Settings
    clusters: list[Cluster]
    chr_root: ChrRootMap
    chr_first_cluster: ChrFirstClusterMap
    chrs: list[str]
    anchors: AnchorMap
    arcs: ArcMap
    singletons: list[SingletonContact]
    long_arcs: RawArcMap
    selected_region: BedRegion | None = None
    compartments: CompartmentMap = field(default_factory=empty_compartment_map)
    activity: SignalMap = field(default_factory=empty_signal_map)
    # Empty builds the anchor contact map from the singletons.
    anchor_contacts: AnchorContactMap = field(default_factory=dict)


def attach_polymer_law(settings: Settings, data: ContactData) -> PolymerLaw:
    """Returns the polymer law of the run and logs what was measured or why not.

    The exponent is `polymer_exponent` when pinned, else the contact fit, else the fallback. One
    bead is the subanchor spacing of the run. A repeated attach with the same exponent logs at
    debug.
    """
    from gnome3d.polymer import FALLBACK_NU, PolymerLaw

    fit = data.contact_fit
    args: tuple[object, ...]
    if settings.polymer_exponent > 0.0:
        nu = float(settings.polymer_exponent)
        msg = "polymer law: exponent pinned at %.3f by the config"
        args = (nu,)
    elif fit is not None and fit.ok:
        nu = fit.nu
        msg = "polymer law: exponent %.3f measured on %s intra pairs, %d to %d kb, slope %+.3f"
        args = (nu, f"{fit.n_pairs:,}", fit.lo // 1000, fit.hi // 1000, fit.slope)
    else:
        nu = FALLBACK_NU
        msg = "polymer law: the singletons cannot supply an exponent (%s), using the fallback %.3f"
        args = (fit.reason if fit is not None else "no fit", nu)
    repeat = settings.polymer is not None and abs(settings.polymer.nu - nu) < 1e-12
    if repeat:
        LOG.debug(msg, *args)
    else:
        log.status(LOG, msg, *args)
    arcs = data.arc_fit
    if arcs is not None and not arcs.ok and not repeat:
        log.status(LOG, "polymer law: loop strength follows the PET count alone (%s)", arcs.reason)
    return PolymerLaw(
        nu=nu,
        s0_bp=int(settings.target_bp_per_subanchor),
        q_half=float(settings.contact_half_saturation),
        arcs=arcs,
        arcs_by_factor=dict(data.arc_fits),
        strength_by_factor=settings.factor_strengths(),
    )


def build_state(
    settings: Settings,
    data: ContactData,
    chrs_list: list[str],
    region: BedRegion | None = None,
) -> CoarseState:
    """Builds the cluster tree from loaded contact data."""
    with log.step(LOG, "build cluster hierarchy"):
        clusters, chr_root, chr_first_cluster = build_cluster_tree(
            data.anchors,
            data.arcs,
            data.breakpoints,
            chrs_list,
        )
        LOG.info("total clusters: %d", len(clusters))

    settings.polymer = attach_polymer_law(settings, data)

    return CoarseState(
        s=settings,
        clusters=clusters,
        chr_root=chr_root,
        chr_first_cluster=chr_first_cluster,
        chrs=chrs_list,
        anchors=data.anchors,
        arcs=data.arcs,
        singletons=data.singletons,
        long_arcs=data.long_arcs,
        selected_region=region,
        compartments=data.compartments,
        activity=data.activity,
        anchor_contacts=data.anchor_contacts,
    )


def compartment_for_clusters(
    state: CoarseState, indices: list[ClusterIndex], chr_: str
) -> I8Array | None:
    """Returns the compartment call per cluster, or None when no track covers the chromosome."""
    ivs = state.compartments.get(chr_, [])
    if not ivs or not indices:
        return None
    clusters = state.clusters
    starts = [clusters[ci].start for ci in indices]
    ends = [clusters[ci].end for ci in indices]
    cls, _score = bin_compartments(ivs, starts, ends)
    return cls


def coarse_track_arrays(
    state: CoarseState, active_region: list[ClusterIndex], chr_of: list[str]
) -> I8Array | None:
    """Returns the compartment per cluster of an active region of several chromosomes.

    `active_region` and `chr_of` are parallel. Returns None when the compartment term is off or
    no chromosome has a track.
    """
    s = state.s
    if not active_region or not s.use_compartments:
        return None

    n = len(active_region)
    clusters = state.clusters
    by_chr: dict[str, list[int]] = {}
    for i, c in enumerate(chr_of):
        by_chr.setdefault(c, []).append(i)

    comp: I8Array = np.zeros(n, dtype=np.int8)
    got_comp = False

    for chr_, rows in by_chr.items():
        idx = [active_region[i] for i in rows]
        starts = [clusters[ci].start for ci in idx]
        ends = [clusters[ci].end for ci in idx]
        ivs = state.compartments.get(chr_, [])
        if ivs:
            cls, _score = bin_compartments(ivs, starts, ends)
            comp[rows] = cls
            got_comp = True

    return comp if got_comp else None


def interpolate_children_linear(state: CoarseState, parent_indices: list[int]) -> None:
    """Sets child cluster positions by linear interpolation between parents.

    A single parent adds `random_vector_np(100.0)` per child and recurses, which draws from the
    global RNG. The call order must not change.
    """
    clusters = state.clusters
    n = len(parent_indices)
    if n == 0:
        return

    if n == 1:
        par = clusters[parent_indices[0]]
        for child_idx in par.children:
            clusters[child_idx].pos = par.pos + random_vector_np(100.0)
            if clusters[child_idx].children:
                interpolate_children_linear(state, [child_idx])
        return

    for i, par_idx in enumerate(parent_indices):
        par = clusters[par_idx]
        n_children = len(par.children)
        if n_children == 0:
            continue

        p_start = par.pos
        p_end = clusters[parent_indices[min(i + 1, n - 1)]].pos

        for j, child_idx in enumerate(par.children):
            t = (j + 0.5) / n_children
            clusters[child_idx].pos = ((1 - t) * p_start + t * p_end).astype(np.float32)
            if clusters[child_idx].children:
                interpolate_children_linear(state, [child_idx])


def compute_segment_bins(
    state: CoarseState, current_level: ChrLevel
) -> tuple[dict[str, list[int]], dict[str, int], int, list[float]]:
    """Computes the heatmap bin boundaries of the segment clusters.

    Returns `(bins, start_ind, total_size, bin_lengths_mb)`. `bin_lengths_mb` is the genomic span
    of each global bin in Mb. The first and the last bin of a chromosome use the cluster start
    and end, not the 0 and 1e9 sentinels.
    """
    clusters = state.clusters
    bins: dict[str, list[int]] = {}
    start_ind: dict[str, int] = {}
    curr_idx = 0
    bin_lengths_mb: list[float] = []

    for chr_ in state.chrs:
        segs = current_level.get(chr_, [])
        breaks = [0]
        for i in range(len(segs) - 1):
            pos = (clusters[segs[i]].end + clusters[segs[i + 1]].start) // 2
            breaks.append(pos)
        breaks.append(int(1e9))
        bins[chr_] = breaks
        start_ind[chr_] = curr_idx

        n = len(segs)
        for i in range(n):
            if n == 1:
                bp = clusters[segs[0]].end - clusters[segs[0]].start
            elif i == 0:
                bp = breaks[1] - clusters[segs[0]].start
            elif i == n - 1:
                bp = clusters[segs[-1]].end - breaks[-2]
            else:
                bp = breaks[i + 1] - breaks[i]
            bin_lengths_mb.append(max(bp, 1) / 1e6)

        curr_idx += len(breaks) - 1

    return bins, start_ind, curr_idx, bin_lengths_mb


def add_long_pet_to_segment_heatmap(
    state: CoarseState,
    h: list[list[float]],
    bins: dict[str, list[int]],
    start_ind: dict[str, int],
    total_size: int,
) -> None:
    """Bins the arcs longer than `max_pet_length` into the segment heatmap.

    Each arc adds `long_pet_scale * score ** long_pet_power` twice to the cell `[st][end]`. `normalize_heatmap` then symmetrises.
    """
    if not state.long_arcs:
        return
    import bisect

    scale = state.s.long_pet_scale
    power = state.s.long_pet_power
    n_added = 0
    for chr_ in state.chrs:
        chrlong_arcs = state.long_arcs.get(chr_, [])
        chr_bins = bins.get(chr_)
        if not chrlong_arcs or chr_bins is None:
            continue
        si_base = start_ind.get(chr_, 0)
        for arc in chrlong_arcs:
            st = si_base + bisect.bisect_right(chr_bins, arc.start) - 1
            end = si_base + bisect.bisect_right(chr_bins, arc.end) - 1
            if st == end:
                continue
            if st < 0 or end < 0 or st >= total_size or end >= total_size:
                continue
            val = scale * (arc.score**power)
            h[st][end] += val
            h[st][end] += val
            n_added += 1
    if n_added > 0:
        LOG.info("long-PET folded into segment heatmap: %d arcs", n_added)


def arc_expected_matrix(
    s: Settings, mids: list[int], arcs: list[tuple[int, int, int] | tuple[int, int, int, int]]
) -> F64Array:
    """Returns the arc target matrix of one active region.

    An arc pair holds the positive target of its PET count and span. An arcless pair closer than
    `background_range_bp` holds minus the background of its separation when `background_weight`
    is on, which the kernels score as a spring symmetric in log distance. Any other arcless pair
    holds -0.5, which the kernels score as the truncated repulsion. A background is at least one
    bead. Sign and magnitude separate the two arcless kinds. The diagonal is 0.

    Parameters
    ----------
    mids
        Genomic midpoint in bp per anchor, in active region order.
    arcs
        (i, j, score) or (i, j, score, factor) per arc, in active region indices.
    """
    n = len(mids)
    mat: F64Array = np.full((n, n), -0.5, dtype=np.float64)
    if float(s.background_weight) > 0.0:
        law = s.polymer_law()
        m = np.asarray(mids, dtype=np.float64)
        sep = np.abs(m[:, None] - m[None, :])
        rng = float(s.background_range_bp)
        for i, j in zip(*np.nonzero((sep > 0.0) & (sep < rng)), strict=True):
            mat[i, j] = -law.background(int(sep[i, j]))
    np.fill_diagonal(mat, 0.0)
    for arc in arcs:
        i, j, score = arc[0], arc[1], arc[2]
        factor = arc[3] if len(arc) > 3 else 0
        exp_d = 1.0 if s.loop_contact else s.arc_expected_distance(score, mids[j] - mids[i], factor)
        # A pair with arcs of two factors takes the closer of the targets.
        if mat[i, j] > 0.0:
            exp_d = min(exp_d, float(mat[i, j]))
        mat[i, j] = exp_d
        mat[j, i] = exp_d
    return mat


def add_contact_background(
    mat: F64Array,
    mids: list[int],
    anchor_heatmap: F64Array | None,
    s: Settings,
    lens: F64Array | None = None,
) -> F64Array:
    """Holds an arcless anchor pair beyond the short range at the contact distance of the law
    when its contact cell puts it under the background.

    Returns a new matrix, or the input when the term is off, the spring weight is zero or there
    is no contact map. The pair is entered as minus its distance, at least one bead and the
    kernels score it with the background spring. A pair at or under its expected contact keeps the
    repulsion marker. Arc pairs and pairs inside the range are unchanged.

    Parameters
    ----------
    mat
        The target matrix of `arc_expected_matrix`.
    mids
        Genomic midpoint of each anchor.
    anchor_heatmap
        Contact counts between anchor pairs, or None.
    s
        The settings of the run.
    lens
        The length of each anchor in bp, at least one, as the map was normalised by. Required
        with `contact_background_z` above zero.
    """
    if (
        not bool(s.use_contact_background)
        or float(s.background_weight) <= 0.0
        or anchor_heatmap is None
    ):
        return mat
    n = len(mids)
    heat = np.asarray(anchor_heatmap, dtype=np.float64)
    if n < 2 or float(heat.max()) <= 0.0:
        return mat
    pos = np.asarray(mids, dtype=np.float64)
    if float(s.contact_background_z) > 0.0:
        if lens is None:
            raise ValueError("contact_background_z needs the anchor lengths")
        return _contact_z_background(mat, pos, heat, np.asarray(lens, dtype=np.float64), s)
    dist, _avg = create_distance_heatmap(s, heat, n, separations_bp=pos)
    dist = np.asarray(dist, dtype=np.float64)
    law = s.polymer_law()
    sep = np.abs(pos[:, None] - pos[None, :])
    bg = np.maximum(1.0, (sep / max(int(law.s0_bp), 1)) ** law.nu)  # law.background as an array
    eligible = (mat == -0.5) & (sep > float(s.background_range_bp)) & (dist > 0.0) & (dist < bg)
    if not eligible.any():
        return mat
    out = np.array(mat, dtype=np.float64, copy=True)
    out[eligible] = -np.maximum(1.0, dist[eligible])
    return out


_Z_ROWS = 512


def _contact_z_background(
    mat: F64Array, pos: F64Array, heat: F64Array, lens: F64Array, s: Settings
) -> F64Array:
    """`add_contact_background` under `contact_background_z`.

    Every arcless pair whose expectation is above z squared and whose count is more than z Poisson
    deviations from it is held at the background times observed over expected to the minus third.
    The count is the map times the two anchor lengths over 1e6. The expectation is the area of the
    pair times the count per unit area over every pair in its log separation bin, eight a decade.
    The ratio has half a count on each side and the target is clipped to one bead and to
    `heatmap_distance_stretching` backgrounds. The pass runs in blocks of rows.
    """
    n = len(pos)
    law = s.polymer_law()
    z_cut = float(s.contact_background_z)
    s0 = max(int(law.s0_bp), 1)
    stretch = float(s.heatmap_distance_stretching)
    n_key = 80

    def rows(r0: int) -> tuple[I64Array, I64Array, F64Array, F64Array, I64Array]:
        r1 = min(n, r0 + _Z_ROWS)
        ii, jj = np.nonzero(np.arange(n)[None, :] > np.arange(r0, r1)[:, None])
        ii = ii + r0
        sep = np.abs(pos[jj] - pos[ii])
        area = lens[ii] * lens[jj] / 1e6
        key = np.clip(np.floor(np.log10(np.maximum(sep, 1.0)) * 8.0), 0, n_key - 1)
        return ii, jj, sep, area, key.astype(np.int64)

    num = np.zeros(n_key)
    den = np.zeros(n_key)
    for r0 in range(0, n, _Z_ROWS):
        ii, jj, _sep, area, key = rows(r0)
        num += np.bincount(key, heat[ii, jj] * area, minlength=n_key)
        den += np.bincount(key, area, minlength=n_key)
    density = np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)

    out = np.array(mat, dtype=np.float64, copy=True)
    held_in = held_out = 0
    for r0 in range(0, n, _Z_ROWS):
        ii, jj, sep, area, key = rows(r0)
        obs = heat[ii, jj] * area
        exp = density[key] * area
        z = (obs - exp) / np.sqrt(np.maximum(exp, 1e-300))
        # A pair is tested only where its expectation is above z squared. There a count of zero is z
        # deviations under it and the test can fire both ways.
        sel = (mat[ii, jj] < 0.0) & (exp > z_cut * z_cut) & (sep > 0.0) & (np.abs(z) > z_cut)
        if not sel.any():
            continue
        bg = np.maximum(1.0, (np.maximum(sep[sel], 1.0) / s0) ** law.nu)
        ratio = (obs[sel] + 0.5) / (exp[sel] + 0.5)
        target = np.clip(bg * ratio ** (-1.0 / 3.0), 1.0, stretch * bg)
        out[ii[sel], jj[sel]] = -target
        out[jj[sel], ii[sel]] = -target
        held_in += int((z[sel] > 0).sum())
        held_out += int((z[sel] < 0).sum())
    log.status(
        LOG,
        "contact background: %s pairs held in, %s held out, of %s at z %.1f",
        f"{held_in:,}",
        f"{held_out:,}",
        f"{n * (n - 1) // 2:,}",
        z_cut,
    )
    return out


def dropout_keep(
    s: Settings,
    mids: list[int],
    arcs: list[tuple[int, int, int] | tuple[int, int, int, int]],
    seed: int,
) -> list[bool]:
    """Returns which arcs one conformation keeps under `loop_dropout`, each with probability
    `q / (q + loop_dropout_scale)`, `q` the strength of the law. The draw comes from a generator
    seeded by the conformation.
    """
    law = s.polymer_law()
    scale = max(float(s.loop_dropout_scale), 1e-9)
    rng = np.random.default_rng((int(seed) * 0x9E3779B1 + 0x7F4A7C15) & 0xFFFFFFFF)
    keep: list[bool] = []
    for arc in arcs:
        i, j, score = arc[0], arc[1], arc[2]
        factor = arc[3] if len(arc) > 3 else 0
        q = max(law.arc_strength(score, abs(mids[i] - mids[j]), factor), 0.0)
        keep.append(bool(rng.random() < q / (q + scale)))
    return keep


def active_arcs(
    state: CoarseState, active_region: list[int], chr_: str, seed: int | None = None
) -> tuple[list[int], list[tuple[int, int, int] | tuple[int, int, int, int]]]:
    """Returns the genomic midpoints of the anchors and the arcs among them in active region
    indices, each arc once.

    With `loop_dropout` on and a `seed`, the arcs that the conformation drops are left out.
    """
    clusters = state.clusters
    cluster_to_active = {ci: ai for ai, ci in enumerate(active_region)}
    chr_arcs = state.arcs.get(chr_, [])
    arcs: list[tuple[int, int, int] | tuple[int, int, int, int]] = []
    for ai, ci in enumerate(active_region):
        for arc_local in clusters[ci].arcs:
            if arc_local >= len(chr_arcs):
                continue
            arc = chr_arcs[arc_local]
            # A pair that two factors join also has a summary arc of score 0. It is skipped.
            # The arcs of the pair carry it.
            if arc.factor < 0:
                continue
            other = arc.end if arc.start == ci else arc.start
            if other < ci or other not in cluster_to_active:
                continue
            arcs.append((ai, cluster_to_active[other], int(arc.score), int(arc.factor)))
    mids = [int(clusters[ci].genomic_pos) for ci in active_region]
    if state.s.loop_dropout and seed is not None and arcs:
        keep = dropout_keep(state.s, mids, arcs, seed)
        arcs = [a for a, k in zip(arcs, keep, strict=True) if k]
    return mids, arcs


def arc_weight_matrix(
    s: Settings, mids: list[int], arcs: list[tuple[int, int, int] | tuple[int, int, int, int]]
) -> F32Array | None:
    """Returns the spring weight of each arc pair, its strength to `arc_weight_exponent` and
    one elsewhere. Returns None at exponent zero.
    """
    alpha = float(s.arc_weight_exponent)
    if alpha <= 0.0:
        return None
    law = s.polymer_law()
    n = len(mids)
    w: F32Array = np.ones((n, n), dtype=np.float32)
    for arc in arcs:
        i, j, score = arc[0], arc[1], arc[2]
        factor = arc[3] if len(arc) > 3 else 0
        q = law.arc_strength(score, abs(mids[i] - mids[j]), factor)
        w[i, j] = w[j, i] = max(q, 1e-6) ** alpha
    return w


def calc_anchor_arc_weights(
    state: CoarseState, active_region: list[int], chr_: str, seed: int | None = None
) -> F32Array | None:
    """Returns the spring weight matrix of the arcs of `calc_anchor_expected_distances`."""
    mids, arcs = active_arcs(state, active_region, chr_, seed)
    return arc_weight_matrix(state.s, mids, arcs)


def calc_anchor_activity(
    state: CoarseState, active_region: list[int], chr_: str
) -> F32Array | None:
    """Returns the activity of each anchor for the factory term, in active region order.

    The activity is the largest value of the track intervals that overlap the anchor, zero without
    one, scaled to a mean of one over the active anchors. Returns None when the term is off or the
    chromosome has no track.
    """
    s = state.s
    track = state.activity.get(chr_)
    if float(s.factory_weight) <= 0.0 or not track:
        return None
    starts = np.array([iv.start for iv in track], dtype=np.int64)
    ends = np.array([iv.end for iv in track], dtype=np.int64)
    vals = np.array([iv.value for iv in track], dtype=np.float64)
    end_max = np.maximum.accumulate(ends)
    act = np.zeros(len(active_region), dtype=np.float32)
    for k, ci in enumerate(active_region):
        a, b = int(state.clusters[ci].start), int(state.clusters[ci].end)
        lo = int(np.searchsorted(end_max, a))
        hi = int(np.searchsorted(starts, b, side="right"))
        if hi > lo:
            m = (ends[lo:hi] >= a) & (starts[lo:hi] <= b)
            if m.any():
                act[k] = float(vals[lo:hi][m].max())
    on = act > 0.0
    if on.any():
        act[on] /= float(act[on].mean())
    return act


def calc_anchor_expected_distances(
    state: CoarseState,
    active_region: list[int],
    chr_: str,
    anchor_heatmap: F64Array | None = None,
    seed: int | None = None,
) -> F64Array:
    """Builds the expected distance matrix of an anchor level active region.

    `seed` selects the conformation for `loop_dropout`. Without one every arc is kept. With
    `anchor_heatmap` and `use_anchor_heatmap` the expected distances of pairs with many contacts
    are scaled down.
    """
    s = state.s
    n = len(active_region)
    mids, arcs = active_arcs(state, active_region, chr_, seed)
    mat = arc_expected_matrix(s, mids, arcs)

    if anchor_heatmap is not None and s.use_anchor_heatmap:
        max_val = float(anchor_heatmap.max())
        influence = float(s.anchor_heatmap_influence)
        if max_val > 1e-6:
            for i in range(n):
                for j in range(i + 1, n):
                    if mat[i, j] <= 0.0:
                        continue
                    s_val = (anchor_heatmap[i, j] / max_val) * influence
                    if s_val > 1.0:
                        s_val = 1.0
                    mat[i, j] *= 1.0 - s_val
                    mat[j, i] = mat[i, j]

    lens = np.maximum(
        np.array(
            [state.clusters[a].end - state.clusters[a].start for a in active_region],
            dtype=np.float64,
        ),
        1.0,
    )
    mat = add_contact_background(mat, mids, anchor_heatmap, s, lens)
    return repulsion_range(mat, mids, s)


def repulsion_range(mat: F64Array, mids: list[int], s: Settings) -> F64Array:
    """Clears every repulsion marker when `arcs_repulsion` is off and returns the input
    otherwise. See `design/gap-fill.md`.
    """
    del mids
    if bool(s.arcs_repulsion):
        return mat
    out = mat.copy()
    out[out == -0.5] = 0.0
    return out


def subanchor_counts_per_arc(state: CoarseState, active_region: list[int]) -> list[int]:
    """Returns the number of subanchors between each consecutive anchor pair.

    Without `use_dynamic_loop_density` the count is `loop_density`. With it,
    `n = round(span_bp / target_bp_per_subanchor) - 1`, clamped to the minimum and the maximum.
    That the chain segments are about the target long. Densification and
    `build_contact_heatmaps` both use this.
    """
    s = state.s
    clusters = state.clusters
    n_arcs = max(len(active_region) - 1, 0)
    if not s.use_dynamic_loop_density:
        return [s.loop_density] * n_arcs

    target = max(s.target_bp_per_subanchor, 1)
    mn = max(s.min_subanchors_per_arc, 0)
    mx = max(s.max_subanchors_per_arc, mn)
    counts: list[int] = []
    for i in range(n_arcs):
        ca = clusters[active_region[i]]
        cb = clusters[active_region[i + 1]]
        span = abs(cb.start - ca.end)
        n = round(span / target) - 1
        counts.append(max(mn, min(mx, n)))
    return counts


def build_contact_heatmaps(
    state: CoarseState,
    active_region: list[int],
    chr_: str,
    with_subanchor: bool = True,
) -> tuple[F64Array, F32Array | None]:
    """Builds the anchor and the subanchor singleton contact heatmaps.

    Returns `(anchor_heatmap, subanchor_heatmap_raw)`. The anchor heatmap is (n_anchors,
    n_anchors) float64, the contact density between anchor pairs. The subanchor heatmap is (N, N)
    float32 at bead resolution. With `with_subanchor` false the anchor heatmap is binned at anchor
    resolution, byte identical and the subanchor matrix is None.
    """
    clusters = state.clusters
    n_anchors = len(active_region)
    counts = subanchor_counts_per_arc(state, active_region)  # length n_anchors - 1
    N = n_anchors + sum(counts)

    # `anchor_offsets[k]` is the bin of anchor k. Subanchor j of arc i is at bin `offsets[i] + 1 +
    # j`.
    anchor_offsets: list[int] = [0]
    for c in counts:
        anchor_offsets.append(anchor_offsets[-1] + 1 + c)

    anchor_lens: list[int] = []
    gap_lens: list[int] = []

    region_start = clusters[active_region[0]].start
    region_end = clusters[active_region[-1]].end

    # `breaks[k]` is the left edge of bin k. An arc with no subanchor gets one break at the midpoint
    # of its gap.
    breaks: list[int] = [region_start]
    anchor_lens.append(clusters[active_region[0]].end - clusters[active_region[0]].start)

    for i in range(n_anchors - 1):
        ca_end = clusters[active_region[i]].end
        cb_start = clusters[active_region[i + 1]].start
        gap = max(cb_start - ca_end, 0)
        anchor_len = clusters[active_region[i + 1]].end - clusters[active_region[i + 1]].start
        gap_lens.append(gap)
        anchor_lens.append(anchor_len)
        c = counts[i]
        if c >= 1:
            breaks.append(ca_end)
            for j in range(1, c):
                breaks.append(ca_end + int(gap * j / c))
            breaks.append(cb_start)
        else:
            breaks.append((ca_end + cb_start) // 2)

    breaks.append(region_end)

    # `breaks` is not monotonic when consecutive anchors overlap. This stays a `bisect` loop.
    # `np.searchsorted` differs there. Contacts are filtered by chromosome.
    import bisect

    anchor_off = np.asarray(anchor_offsets[:n_anchors], dtype=np.intp)
    al = np.maximum(np.asarray(anchor_lens, dtype=np.float64), 1.0)

    # A full depth anchor contact table, when configured, is the anchor map. The singletons then
    # feed the subanchor map alone.
    table = state.anchor_contacts.get(chr_)
    h_table: F64Array | None = None
    if table is not None:
        st = np.array([clusters[a].start for a in active_region], dtype=np.int64)
        en = np.array([clusters[a].end for a in active_region], dtype=np.int64)
        h_table = dense_counts(table, st, en) / (np.outer(al, al) / 1e6)
        if not with_subanchor:
            return h_table, None

    if not with_subanchor:
        # A contact with an end in a subanchor bin is not an anchor pair and is dropped.
        bin_anchor = np.full(N, -1, dtype=np.int64)
        bin_anchor[anchor_off] = np.arange(n_anchors)
        h_direct: F64Array = np.zeros((n_anchors, n_anchors), dtype=np.float64)
        for c1, p1, c2, p2, sc in state.singletons:
            if c1 != chr_ or c2 != chr_:
                continue
            if p1 < region_start or p1 > region_end or p2 < region_start or p2 > region_end:
                continue
            si = bisect.bisect_right(breaks, p1) - 1
            ei = bisect.bisect_right(breaks, p2) - 1
            if si < 0 or ei < 0 or si >= N or ei >= N or si == ei:
                continue
            ai = int(bin_anchor[si])
            aj = int(bin_anchor[ei])
            if ai < 0 or aj < 0:
                continue
            h_direct[ai, aj] += sc
            h_direct[aj, ai] += sc
        h_direct /= np.outer(al, al) / 1e6
        np.fill_diagonal(h_direct, 0.0)
        return h_direct, None

    h_sub: F64Array = np.zeros((N, N), dtype=np.float64)
    for c1, p1, c2, p2, sc in state.singletons:
        if c1 != chr_ or c2 != chr_:
            continue
        if p1 < region_start or p1 > region_end or p2 < region_start or p2 > region_end:
            continue
        si = bisect.bisect_right(breaks, p1) - 1
        ei = bisect.bisect_right(breaks, p2) - 1
        if si < 0 or ei < 0 or si >= N or ei >= N or si == ei:
            continue
        h_sub[si, ei] += sc
        h_sub[ei, si] += sc

    # The anchor heatmap comes from the raw subanchor values, before normalisation, over the anchor
    # area in Mb squared.
    h_anchor: F64Array = h_sub[np.ix_(anchor_off, anchor_off)] / (np.outer(al, al) / 1e6)
    np.fill_diagonal(h_anchor, 0.0)
    if h_table is not None:
        h_anchor = h_table

    # -1 at anchor bins
    bin_arc_idx: list[int] = [-1] * N
    for i, c in enumerate(counts):
        base = anchor_offsets[i] + 1
        for j in range(c):
            bin_arc_idx[base + j] = i

    # Divided by the mean count, then by the bin areas in kb squared.
    avg_count = float(h_sub.mean())
    if avg_count > 1e-6:
        h_sub /= avg_count

        # An anchor bin has the length of the anchor. A subanchor bin has the gap length over the
        # count.
        bin_sizes: F64Array = np.empty(N, dtype=np.float64)
        bai = np.asarray(bin_arc_idx, dtype=np.intp)
        gl = np.asarray(gap_lens, dtype=np.float64)
        cnt = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
        per_arc = np.maximum(gl / cnt, 1.0) / 1000.0
        sub_mask = bai >= 0
        bin_sizes[sub_mask] = per_arc[bai[sub_mask]]
        bin_sizes[anchor_off] = al / 1000.0

        denom = np.outer(bin_sizes, bin_sizes)
        off = denom > 0.0
        np.fill_diagonal(off, False)  # the diagonal is unchanged
        h_sub[off] = h_sub[off] / denom[off]

    return h_anchor, h_sub.astype(np.float32)


# The functions below draw from the global RNG in a fixed order and write `clusters[i].pos`.


def reconstruct_heatmap(state: CoarseState) -> None:
    """Positions the segment beads.

    Takes one of three branches: a random walk, one segment at the origin or the chromosome and
    segment heatmap MC.
    """
    current_level = set_level(
        Level.SEGMENT - Level.CHROMOSOME, state.chr_root, state.clusters, state.chrs
    )

    if state.s.random_walk:
        random_walk_segment_level(state, current_level)
        return

    total_segs = sum(len(v) for v in current_level.values())
    single_seg = len(state.chrs) == 1 and total_segs <= 1

    if single_seg:
        place_single_segment(state, current_level)
        return

    if len(state.chrs) > 1:
        reconstruct_chromosome_level(state)

    LOG.info("segment level")
    reconstruct_segment_level(state, current_level)


def place_single_segment(state: CoarseState, current_level: ChrLevel) -> None:
    """Places one segment at the origin and interpolates its children."""
    LOG.info("single segment -> place at origin")
    chr_ = state.chrs[0]
    if current_level[chr_]:
        state.clusters[current_level[chr_][0]].pos = np.zeros(3, dtype=np.float32)
        interpolate_children_linear(state, current_level[chr_])


def reconstruct_chromosome_level(state: CoarseState) -> None:
    """Positions the chromosome roots by heatmap MC on the inter chromosomal singleton heatmap.

    Cell (i, j) is the sum of singleton scores between chromosomes i and j. Without inter
    chromosomal singletons the roots are scattered around the origin.
    """
    from gnome3d.mc import numba as mc_numba

    s = state.s
    clusters = state.clusters
    n_chr = len(state.chrs)
    if n_chr <= 1:
        return

    chr_to_idx = {chr_: i for i, chr_ in enumerate(state.chrs)}
    h: list[list[float]] = [[0.0] * n_chr for _ in range(n_chr)]
    n_inter = 0
    for c1, _p1, c2, _p2, sc in state.singletons:
        i = chr_to_idx.get(c1, -1)
        j = chr_to_idx.get(c2, -1)
        if i < 0 or j < 0 or i == j:
            continue
        h[i][j] += sc
        h[j][i] += sc
        n_inter += 1

    LOG.info("chromosome level  (%d chr, %d inter-chr singletons)", n_chr, n_inter)

    if n_inter == 0:
        LOG.info("no inter-chromosomal singletons; scattering chr roots randomly")
        for chr_ in state.chrs:
            root_idx = state.chr_root.get(chr_)
            if root_idx is not None:
                clusters[root_idx].pos = random_vector_np(100.0, in_2d=s.use_2d)
        return

    hd = normalize_heatmap_diagonal_total(h, n_chr, 1.0)
    # Chromosome pairs have no genomic separation. Every cell shares one expectation and the
    # background is taken at the mean chromosome span.
    spans = [
        float(clusters[state.chr_root[c]].end - clusters[state.chr_root[c]].start)
        for c in state.chrs
        if c in state.chr_root
    ]
    heatmap_dist, avg_dist = create_distance_heatmap(
        s, hd, n_chr, scale_bp=float(np.mean(spans)) if spans else None
    )
    heatmap_dist = np.array(heatmap_dist, dtype=np.float64)
    heatmap_dist_diag = get_diagonal_size(hd, n_chr)

    step_size = avg_dist * s.noise_lvl1

    # Scattered around the origin.
    initial_pos: F32Array = np.zeros((n_chr, 3), dtype=np.float32)
    for i in range(n_chr):
        initial_pos[i] = random_vector_np(step_size, in_2d=s.use_2d)

    pos: F32Array = initial_pos.copy()
    best_pos: F32Array = pos.copy()
    best_score = -1.0

    for run in range(s.steps_lvl1):
        with log.step(LOG, f"chr-level run {run + 1}/{s.steps_lvl1}", "(%d chr)", n_chr):
            for i in range(n_chr):
                pos[i] = initial_pos[i] + random_vector_np(step_size, in_2d=s.use_2d)

            score = mc_numba.mc_heatmap_numba(pos, heatmap_dist, heatmap_dist_diag, step_size, s)
            if score < best_score or best_score < 0:
                best_score = score
                best_pos = pos.copy()

            LOG.info("-> score=%.6f  best=%.6f", score, best_score)

    # The segment level reads these as its origin.
    for i, chr_ in enumerate(state.chrs):
        root_idx = state.chr_root.get(chr_)
        if root_idx is not None:
            clusters[root_idx].pos = best_pos[i].copy()


def random_walk_segment_level(state: CoarseState, current_level: ChrLevel) -> None:
    """Positions the segment beads on a chained random walk.

    Each chromosome starts at the origin, takes one step of the distance of 1 Mb and then one
    step of 50 units per segment.
    """
    clusters = state.clusters
    in_2d = state.s.use_2d
    size = float(state.s.genomic_length_to_distance(1_000_000))
    with log.step(LOG, "random walk for segment level", "(size=%.2f, 2D=%s)", size, in_2d):
        for chr_ in state.chrs:
            segs = current_level.get(chr_, [])
            if not segs:
                continue
            LOG.info("%s: %d segment(s)", chr_, len(segs))
            rw_pos: F32Array = np.zeros(3, dtype=np.float32)
            rw_pos = rw_pos + random_vector_np(size, in_2d=in_2d)
            for seg_idx in segs:
                rw_pos = rw_pos + random_vector_np(50.0, in_2d=in_2d)
                clusters[seg_idx].pos = rw_pos.copy()
            interpolate_children_linear(state, segs)


def reconstruct_segment_level(state: CoarseState, current_level: ChrLevel) -> None:
    """Positions the segment beads by singleton heatmap MC."""
    from gnome3d.mc import numba as mc_numba

    s = state.s
    clusters = state.clusters
    bins, start_ind, total_size, bin_lengths_mb = compute_segment_bins(state, current_level)

    LOG.info("create segment heatmap")
    h_raw = create_singleton_heatmap(
        state.singletons, bins, start_ind, total_size, bin_lengths_mb=bin_lengths_mb
    )

    add_long_pet_to_segment_heatmap(state, h_raw, bins, start_ind, total_size)

    h_norm = normalize_heatmap(h_raw, total_size)
    h_norm = normalize_heatmap_diagonal_total(h_norm, total_size, 1.0)
    if len(state.chrs) > 1:
        h_norm = normalize_heatmap_inter(h_norm, total_size, current_level, s.heatmap_inter_scaling)

    ends = np.cumsum(np.asarray(bin_lengths_mb, dtype=np.float64)) * 1e6
    separations = ends - 0.5 * np.asarray(bin_lengths_mb, dtype=np.float64) * 1e6
    heatmap_dist, avg_dist = create_distance_heatmap(
        s, h_norm, total_size, separations_bp=separations
    )
    heatmap_dist = np.array(heatmap_dist, dtype=np.float64)
    heatmap_dist_diag = get_diagonal_size(h_norm, total_size)

    for chr_ in state.chrs:
        segs = current_level.get(chr_, [])
        if not segs:
            continue
        par = clusters[segs[0]].parent
        origin = clusters[par].pos.copy() if par >= 0 else np.zeros(3, dtype=np.float32)
        for seg_idx in segs:
            clusters[seg_idx].pos = origin.copy()

    active_region: list[int] = []
    chr_of: list[str] = []
    for chr_ in state.chrs:
        segs_here = current_level.get(chr_, [])
        active_region.extend(segs_here)
        chr_of.extend([chr_] * len(segs_here))

    if len(active_region) <= 1:
        return

    step_size = avg_dist * s.noise_lvl2

    seg_comp = coarse_track_arrays(state, active_region, chr_of)

    pos: F32Array = np.array([clusters[i].pos for i in active_region], dtype=np.float32)
    n = len(active_region)
    best_score = -1.0
    best_pos: F32Array = pos.copy()

    for run in range(s.steps_lvl2):
        with log.step(LOG, f"heatmap run {run + 1}/{s.steps_lvl2}", "(%d beads)", n):
            for i in range(n):
                pos[i] = clusters[active_region[i]].pos + random_vector_np(step_size)

            score = mc_numba.mc_heatmap_numba(
                pos,
                heatmap_dist,
                heatmap_dist_diag,
                step_size,
                s,
                seg_comp,
            )
            if score < best_score or best_score < 0:
                best_score = score
                best_pos = pos.copy()
            LOG.info("-> score=%.6f  best=%.6f", score, best_score)

    for i, idx in enumerate(active_region):
        clusters[idx].pos = best_pos[i].copy()

    for chr_ in state.chrs:
        segs = current_level.get(chr_, [])
        if segs:
            interpolate_children_linear(state, segs)


def position_interaction_blocks(state: CoarseState, segs: list[int], chr_: str) -> None:
    """Positions the IB clusters between the segment positions. With `use_ib_mc` an MC pass on the IB centroids follows."""
    clusters = state.clusters
    if len(segs) > 1:
        interpolate_children_linear(state, segs)
    else:
        seg = clusters[segs[0]]
        pos: F32Array = np.zeros(3, dtype=np.float32)
        for ib_idx in seg.children:
            pos = pos + random_vector_np(100.0)
            clusters[ib_idx].pos = pos.copy()

    if state.s.use_ib_mc:
        ib_mc_refine(state, segs, chr_)


def block_heatmap_distances(
    s: Settings,
    singletons: list[SingletonContact],
    chr_: str,
    blocks: list[tuple[int, int, int]],
) -> F64Array:
    """Returns the distance targets between the blocks of one chain, from the contacts of the run.

    Contacts are binned by block, with bin edges halfway between neighbouring blocks, then
    normalised and converted with the law as the segment heatmap is. A cell with no contact holds
    0 and the diagonal band -1 and the kernel skips both.

    Parameters
    ----------
    s
        The settings of the run.
    singletons
        Contacts on this chromosome.
    chr_
        The chromosome.
    blocks
        One (start, end, midpoint) per block, in genomic order.
    """
    n = len(blocks)
    if n <= 1:
        return np.zeros((n, n), dtype=np.float64)
    breaks = [0]
    for (_, end_a, _), (start_b, _, _) in zip(blocks[:-1], blocks[1:], strict=True):
        breaks.append((end_a + start_b) // 2)
    breaks.append(int(1e9))
    lengths_mb: list[float] = []
    for i, (start, end, _) in enumerate(blocks):
        if i == 0:
            bp = breaks[1] - start
        elif i == n - 1:
            bp = end - breaks[-2]
        else:
            bp = breaks[i + 1] - breaks[i]
        lengths_mb.append(max(bp, 1) / 1e6)
    h_raw = create_singleton_heatmap(
        singletons, {chr_: breaks}, {chr_: 0}, n, bin_lengths_mb=lengths_mb
    )
    if float(np.asarray(h_raw).sum()) <= 0.0:
        return np.zeros((n, n), dtype=np.float64)
    h_norm = normalize_heatmap(h_raw, n)
    h_norm = normalize_heatmap_diagonal_total(h_norm, n, 1.0)
    mids = np.array([m for _, _, m in blocks], dtype=np.float64)
    dist, _avg = create_distance_heatmap(s, h_norm, n, separations_bp=mids)
    return np.array(dist, dtype=np.float64)


def ib_mc_refine(state: CoarseState, segs: list[int], chr_: str) -> None:
    """Refines the IB centroids with an MC pass of chain bonds, excluded volume and confinement,
    under `use_ib_mc`.

    `ib_refine_scope` sets the chain. "segment" refines the blocks of each segment alone and
    skips a segment with one block or fewer. "chromosome" refines every block of the chromosome as
    one chain, which puts every block pair inside the excluded volume term.
    """
    from gnome3d.mc import numba as mc_numba

    s = state.s
    clusters = state.clusters
    if str(getattr(s, "ib_refine_scope", "segment")).strip().lower() == "chromosome":
        groups = [
            sorted(
                (ib for seg_idx in segs for ib in clusters[seg_idx].children),
                key=lambda i: clusters[i].genomic_pos,
            )
        ]
    else:
        groups = [list(clusters[seg_idx].children) for seg_idx in segs]

    heat_w = float(s.heatmap_weight_ib)
    chr_contacts: list[SingletonContact] = (
        [c for c in state.singletons if c[0] == chr_ and c[2] == chr_] if heat_w > 0.0 else []
    )

    for ibs in groups:
        if len(ibs) <= 1:
            continue
        pos: F32Array = np.array([clusters[ib].pos for ib in ibs], dtype=np.float32)
        heat: F64Array | None = None
        if heat_w > 0.0:
            heat = block_heatmap_distances(
                s,
                chr_contacts,
                chr_,
                [(clusters[ib].start, clusters[ib].end, clusters[ib].genomic_pos) for ib in ibs],
            )
        dtn: F32Array = np.zeros(len(ibs) - 1, dtype=np.float32)
        for i in range(len(ibs) - 1):
            gap = abs(clusters[ibs[i + 1]].genomic_pos - clusters[ibs[i]].genomic_pos)
            dtn[i] = float(s.genomic_length_to_distance(gap))

        avg_dtn = float(dtn.mean()) if dtn.size > 0 else 1.0
        step_size = avg_dtn * s.noise_ib

        log_ev_r0 = (
            s.exclusion_radius_ib
            if s.exclusion_radius_ib > 0.0
            else s.exclusion_auto_factor_ib * avg_dtn
        )
        log_conf_R = (
            s.confinement_radius_ib
            if s.confinement_radius_ib > 0.0
            else s.confinement_packing_factor_ib * avg_dtn * (len(ibs) ** (1.0 / 3.0))
        )
        conf_tag = (
            f", tether_R={log_conf_R:.2f}"
            if (s.use_confinement and s.confinement_apply_to_ib)
            else ""
        )

        with log.step(
            LOG,
            f"IB-MC {chr_} ({len(segs)} segments)",
            "%d IBs, avg_bond=%.2f, ev_r0=%.3f, step=%.3f%s",
            len(ibs),
            avg_dtn,
            log_ev_r0,
            step_size,
            conf_tag,
        ):
            gyr_before = float(np.linalg.norm(pos - pos.mean(axis=0), axis=1).mean())
            mc_numba.mc_ib_numba(
                pos,
                dtn,
                step_size,
                s,
                compartment_for_clusters(state, ibs, chr_),
                heat_dist=heat,
            )
            gyr_after = float(np.linalg.norm(pos - pos.mean(axis=0), axis=1).mean())
            LOG.info("gyr %.2f -> %.2f", gyr_before, gyr_after)

        for i, ib in enumerate(ibs):
            clusters[ib].pos = pos[i].copy()


__all__ = [
    "active_arcs",
    "repulsion_range",
    "CoarseState",
    "COARSE_SEED",
    "seed_global_rng",
    "build_state",
    "interpolate_children_linear",
    "compute_segment_bins",
    "add_long_pet_to_segment_heatmap",
    "calc_anchor_expected_distances",
    "subanchor_counts_per_arc",
    "build_contact_heatmaps",
    "create_distance_heatmap",
    "reconstruct_heatmap",
    "place_single_segment",
    "reconstruct_chromosome_level",
    "random_walk_segment_level",
    "reconstruct_segment_level",
    "position_interaction_blocks",
    "ib_mc_refine",
]
