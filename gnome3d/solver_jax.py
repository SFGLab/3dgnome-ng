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

Pass 1 (this module) batches the FINAL smooth across IBs.  The subanchor
heat-dist estimate inside `_prepare_ib` is still per-IB (itself a small batch);
lifting it cross-IB is a planned follow-on for the rest of the win.

Selected via `gnome3d.util.make_solver` when `settings.jax_region_batch` is set
(opt-in).
"""

from __future__ import annotations

import time
from collections import defaultdict
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
        # Phase 1: prepare every IB (arc MC + contact heatmaps + heat-dist +
        # smooth-problem build).  Serial and cheap relative to the final smooth
        # (arc MC is numba/CPU); each IB owns disjoint clusters, so order-free.
        probs: list[dict[str, Any]] = []
        for _ib_i, ib_idx, ib_label, active_region in work:
            with log.step(LOG, ib_label, "(%d anchors)", len(active_region)):
                prob = self._prepare_ib(ib_idx, active_region, chr_)
            if prob is not None:
                prob["ib_label"] = ib_label
                probs.append(prob)
        if not probs:
            return

        # Phase 2: anneal all IBs' final smooths in batched kernels, grouped so
        # each batch is shape-uniform (same energy terms + size bucket).
        beads = self._batched_final_smooth(probs)
        self.dense_active_regions.setdefault(chr_, []).extend(beads)

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
