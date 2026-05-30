"""
GPU region-batched solver.

`JaxSolver` fills the GPU by annealing many independent interaction blocks (IBs)
in ONE vmapped smooth-MC kernel instead of one at a time.  The base `Solver`
processes IBs serially (the GPU sits ~99% idle per IB at K=1); region batching
turns hundreds of tiny latency-bound kernels into a handful of wide ones —
measured ~8-11x on the dominant smooth phase, scaling further with batch width.

It overrides exactly one seam — `_dispatch_ib_work`, the per-chromosome IB
dispatch — and reuses everything else from `Solver` (hierarchy, arc MC, contact
heatmaps, subanchor heat-dist prep, write-back).  See
`gnome3d.mc_jax.mc_smooth_jax_batch` for the batched kernel entry.

Work is split into three phases per chromosome:
  * Phase 1 (`_prepare_ibs_threaded`): arc MC + contact heatmaps + densify +
    orientation, per IB.  Pure CPU (heat-dist deferred) and arc MC is numba/
    nogil, so it threads across cores — wall = slowest IB, not the sum.
  * Phase 1.5 (`_batched_heat_dist`): every IB's subanchor heat-dist estimate
    (the dry-smooth trials) runs in batched GPU kernels, grouped by size bucket.
  * Phase 2 (`_batched_final_smooth`): every IB's final smooth, likewise.

Selected via `gnome3d.util.make_solver` when `settings.jax_region_batch` is set
(opt-in).
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from . import log
from .mc_jax import _bucket_for, mc_smooth_jax_batch  # pyright: ignore[reportPrivateUsage]
from .solver import Solver
from .types import BeadOut, BoolArray, F32Array, F64Array
from .util import random_vector_np

LOG = log.get("solver")


class JaxSolver(Solver):
    """Solver that region-batches the IB smooth phase on the GPU."""

    def _estimate_avg_dist(
        self,
        pos: F32Array,
        fixed: BoolArray,
        dtn: F32Array,
        step_size: float,
        n_reps: int,
        n_steps: int,
    ) -> tuple[F64Array, float]:
        """Batched subanchor-distance estimate: run all `n_reps*n_steps` dry
        smooth trials as ONE vmapped kernel instead of the base's sequential
        double loop.  Falls back to the reference path when `subanchor_batch_trials`
        is off or JAX is unavailable.

        Intentional divergence from the reference: batched trials share a
        best-of-K convergence stop rather than each running to its own.
        Validated on avg_dist.
        """
        s = self.s
        if not bool(getattr(s, "subanchor_batch_trials", False)):
            return super()._estimate_avg_dist(pos, fixed, dtn, step_size, n_reps, n_steps)
        from . import mc_jax

        if not mc_jax.is_available():
            return super()._estimate_avg_dist(pos, fixed, dtn, step_size, n_reps, n_steps)

        n = len(pos)
        t_b = time.perf_counter()
        n_trials = n_reps * n_steps
        starts = np.empty((n_trials, n, 3), dtype=np.float32)
        b = 0
        for _rep in range(n_reps):
            for _step in range(n_steps):
                pt = pos.copy()
                for i in range(n):
                    if not fixed[i]:
                        pt[i] += random_vector_np(step_size)
                starts[b] = pt
                b += 1
        # One kernel: K = n_trials independent anneals from the distinct starts.
        with log.scope("est batched"):
            scores_flat, finals_flat = mc_jax.mc_smooth_jax(
                pos, dtn, fixed, step_size, s, pos_batch=starts, return_all=True
            )
        scores = np.asarray(scores_flat).reshape(n_reps, n_steps)
        finals = np.asarray(finals_flat).reshape(n_reps, n_steps, n, 3)
        avg_dist: F64Array = np.zeros((n, n), dtype=np.float64)
        for rep in range(n_reps):
            bt = int(np.argmin(scores[rep]))
            rep_best_pos = finals[rep, bt]
            diff = rep_best_pos[:, np.newaxis, :] - rep_best_pos[np.newaxis, :, :]
            avg_dist += np.sqrt((diff * diff).sum(axis=2))
            LOG.info("rep %d/%d: best_score=%.4f", rep + 1, n_reps, float(scores[rep, bt]))
        avg_dist /= n_reps
        t_mc_total = time.perf_counter() - t_b
        LOG.info("batched %d trials in one kernel (%.2fs)", n_trials, t_mc_total)
        return avg_dist, t_mc_total

    def _dispatch_ib_work(self, chr_: str, work: list[tuple[int, int, str, list[int]]]) -> None:
        # Phase 1: prepare every IB (arc MC + contact heatmaps + densify +
        # orientation), DEFERRING the heat-dist estimate.  This phase is now
        # pure CPU (the GPU work is all in the batched phases below), and arc MC
        # is numba/nogil, so we thread it across cores — wall = slowest IB, not
        # the sum.  IBs own disjoint clusters, so the threads don't race.
        probs = self._prepare_ibs_threaded(chr_, work)
        if not probs:
            return

        # Phase 1.5: estimate the subanchor heat-dist for ALL IBs in batched
        # kernels (the dry-smooth trials), then build each target matrix.
        self._batched_heat_dist(probs)

        # Phase 2: anneal all IBs' final smooths in batched kernels, grouped so
        # each batch is shape-uniform (same energy terms + size bucket).
        beads = self._batched_final_smooth(probs)
        self.dense_active_regions.setdefault(chr_, []).extend(beads)

    def _prepare_ibs_threaded(
        self, chr_: str, work: list[tuple[int, int, str, list[int]]]
    ) -> list[dict[str, Any]]:
        """Run Phase-1 prep for every IB, threaded across cores.  Prep is CPU
        only (heat-dist deferred) and arc MC is numba/nogil, so threads give
        real parallelism; `log.parallel` keeps the interleaved output tagged."""

        def prep_one(item: tuple[int, int, str, list[int]]) -> dict[str, Any] | None:
            _ib_i, ib_idx, ib_label, active_region = item
            with log.step(LOG, ib_label, "(%d anchors)", len(active_region)):
                prob = self._prepare_ib(ib_idx, active_region, chr_, defer_heat=True)
            if prob is not None:
                prob["ib_label"] = ib_label
            return prob

        n_workers = min(len(work), os.cpu_count() or 1)
        if n_workers > 1 and len(work) > 1:
            with log.parallel(), ThreadPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(prep_one, work))
        else:
            results = [prep_one(item) for item in work]
        return [p for p in results if p is not None]

    def _batched_heat_dist(self, probs: list[dict[str, Any]]) -> None:
        """Phase 1.5: estimate every IB's subanchor avg-pairwise-distance in
        batched dry-smooth kernels (chain+EV+conf; no heat, no orientation),
        then build each IB's heat-dist target matrix and attach it to its
        problem.  This lifts the per-IB heat-dist estimate cross-IB — the half
        of the smooth cost the serial path leaves on the table.

        IBs whose subanchor heatmap is empty (mean<1e-6) are skipped entirely —
        their estimate would be discarded anyway (matches the serial early-out).
        """
        s = self.s
        n_reps = int(s.subanchor_estimate_replicates)
        n_steps = int(s.subanchor_estimate_steps)
        per_ib = n_reps * n_steps
        bucket = bool(s.jax_bucket_shapes)

        # Skip IBs whose heat is empty (mean<1e-6, discarded anyway) OR too sparse
        # to move the structure (`subanchor_heat_min_reduction` early-out).  Those
        # keep heat_dist=None, so Phase 2 smooths them without heat and we never
        # pay for their dry-smooth trials here.
        need = [
            p
            for p in probs
            if p.get("subanchor_heat_raw") is not None
            and float(p["subanchor_heat_raw"].mean()) >= 1e-6
            and not self._heat_signal_negligible(p["subanchor_heat_raw"], p["n"])
        ]
        if not need:
            return

        groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for prob in need:
            groups[_bucket_for(prob["n"]) if bucket else prob["n"]].append(prob)

        for bkt, group in groups.items():
            with log.step(
                LOG,
                f"batched heat-dist: {len(group)} IBs x {per_ib} trials",
                "(bucket %d)",
                bkt,
            ):
                # Flat batch of dry-smooth trials; `spans[gi]` is IB gi's start
                # offset (it occupies `per_ib` consecutive entries).
                batch: list[dict[str, Any]] = []
                spans: list[int] = []
                for prob in group:
                    spans.append(len(batch))
                    pos = prob["pos"]
                    fixed = prob["fixed"]
                    n = prob["n"]
                    step = prob["step_size"]
                    for _ in range(per_ib):
                        start = pos.copy()
                        for i in range(n):
                            if not fixed[i]:
                                start[i] += random_vector_np(step)
                        batch.append(
                            {
                                "pos": start,
                                "dtn": prob["dtn"],
                                "fixed": fixed,
                                "step_size": step,
                                "heat_dist": None,
                                "char_orientations": None,
                                "anchor_neighbors": None,
                                "anchor_neighbor_weights": None,
                            }
                        )

                t0 = time.perf_counter()
                results = mc_smooth_jax_batch(batch, s)
                t_batch = time.perf_counter() - t0
                LOG.info(
                    "%d trials for %d IBs in one kernel (%.2fs)", len(batch), len(group), t_batch
                )

                # Per IB: best-of-n_steps per replicate -> accumulate avg_dist.
                for gi, prob in enumerate(group):
                    n = prob["n"]
                    base = spans[gi]
                    avg_dist: F64Array = np.zeros((n, n), dtype=np.float64)
                    for rep in range(n_reps):
                        rep_slice = results[base + rep * n_steps : base + (rep + 1) * n_steps]
                        scores = [r[0] for r in rep_slice]
                        best_pos = rep_slice[int(np.argmin(scores))][1]
                        diff = best_pos[:, None, :] - best_pos[None, :, :]
                        avg_dist += np.sqrt((diff * diff).sum(axis=2))
                    avg_dist /= n_reps
                    prob["heat_dist"] = self._heat_dist_from_avg(
                        avg_dist, prob["subanchor_heat_raw"], t_batch
                    )

    def _batched_final_smooth(self, probs: list[dict[str, Any]]) -> list[BeadOut]:
        """Run the per-IB final smooth as batched GPU kernels and apply results.

        IBs are grouped by (has-heat, has-orientation, size bucket) so every
        batch has uniform kernel shapes.  Each IB is noised+restarted
        `steps_smooth` times (mirroring the serial path); the best restart per
        IB is kept.
        """
        s = self.s
        bucket = bool(s.jax_bucket_shapes)
        n_restarts = max(1, int(s.steps_smooth))

        groups: dict[tuple[bool, bool, int], list[dict[str, Any]]] = defaultdict(list)
        for prob in probs:
            key = (
                prob["heat_dist"] is not None,
                prob["char_orientations"] is not None,
                _bucket_for(prob["n"]) if bucket else prob["n"],
            )
            groups[key].append(prob)

        out_beads: list[BeadOut] = []
        for (_has_heat, _has_orn, bkt), group in groups.items():
            with log.step(LOG, f"batched smooth: {len(group)} IBs", "(bucket %d)", bkt):
                # Build the flat batch: each IB contributes `n_restarts` noised
                # starts; `owner[j]` maps batch entry j back to its IB.
                batch: list[dict[str, Any]] = []
                owner: list[int] = []
                for gi, prob in enumerate(group):
                    fixed = prob["fixed"]
                    step = prob["step_size"]
                    n = prob["n"]
                    for _ in range(n_restarts):
                        start = prob["pos"].copy()
                        for i in range(n):
                            if not fixed[i]:
                                start[i] += random_vector_np(step)
                        batch.append(
                            {
                                "pos": start,
                                "dtn": prob["dtn"],
                                "fixed": fixed,
                                "step_size": step,
                                "heat_dist": prob["heat_dist"],
                                "char_orientations": prob["char_orientations"],
                                "anchor_neighbors": prob["anchor_neighbors"],
                                "anchor_neighbor_weights": prob["anchor_neighbor_weights"],
                            }
                        )
                        owner.append(gi)

                results = mc_smooth_jax_batch(batch, s)

                # Reduce to the best restart per IB, then write back + emit beads.
                best: dict[int, tuple[float, Any]] = {}
                for (score, final_pos), gi in zip(results, owner, strict=True):
                    if gi not in best or score < best[gi][0]:
                        best[gi] = (score, final_pos)
                for gi, prob in enumerate(group):
                    _score, final_pos = best[gi]
                    out_beads.extend(self._apply_smooth_problem(prob, final_pos))
        return out_beads
