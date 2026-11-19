"""The shape bucket for batch grouping.

One batch launch pads every IB to one shape and reads its energy term flags from
`problems[0]`. A batch must be uniform in the term signature and the bucket.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gnome3d.settings import Settings


def batch_bucket(n: int, settings: Settings) -> int:
    """Returns the shape bucket `n` is padded to, `jax_bucket_for(n)` with `jax_bucket_shapes`
    on and `n` otherwise.
    """
    if not bool(settings.mc_executor_jax_bucket_shapes):
        return int(n)

    from gnome3d.mc.jax.util import jax_bucket_for

    return int(jax_bucket_for(int(n)))
