"""Batch-grouping key for the JAX-batched IB stages.

A single `mc_*_jax_batch` launch pads every IB up to one common shape bucket and
reads its energy-term flags from `problems[0]` - so a batch must be uniform in
`(energy-term signature, shape-ladder bucket)`.  The executor groups ready
batch-nodes by each stage's `batch_key` (built on `batch_bucket` here) before
dispatching, exactly as the pre-refactor `JaxSolver._dispatch_ib_work` grouped
by `(use_heat, use_orn, _bucket_for(n))`.

Grouping by the *raw* size instead (the regressed behavior) both fragments one
ladder bucket into many launches - each its own XLA compile + serial kernel
launch - and risks mixing heat/no-heat IBs into one (wrong-kernel) batch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gnome3d.settings import Settings


def batch_bucket(n: int, settings: Settings) -> int:
    """The shape-ladder bucket `n` pads up to, matching what the batched kernel
    does: `jax_bucket_for(n)` when `jax_bucket_shapes` is on, else `n`` itself
    (each distinct size its own group, as when bucketing is disabled)."""
    if not bool(settings.mc_executor_jax_bucket_shapes):
        return int(n)

    from gnome3d.mc.jax.util import jax_bucket_for

    return int(jax_bucket_for(int(n)))
