"""The depth of a hosted contact map and the thinning of deeper contacts to it.

The engine reads every contact map at the depth of the hosted maps. Contacts of a user, from a
file or from a link, are thinned to that depth.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


def thin(counts: np.ndarray, target: int, seed: int) -> np.ndarray:
    """Counts reduced to a target total by sampling contacts without replacement. A map at or under
    the target is kept whole."""
    counts = np.asarray(counts, dtype=np.int64)
    if counts.sum() <= target:
        return counts
    return np.random.default_rng(seed).multivariate_hypergeometric(counts, target)


@lru_cache(maxsize=64)
def hosted_density(singletons: Path, chrom: str) -> float:
    """Contacts per Mb of a hosted map on one chromosome, its total count over its covered span."""
    t = pd.read_csv(singletons, sep=r"\s+", header=None, usecols=[0, 3, 5, 6], comment="#")
    t = t[(t[0] == chrom) & (t[3] == chrom)]
    if t.empty:
        t = pd.read_csv(singletons, sep=r"\s+", header=None, usecols=[0, 3, 5, 6], comment="#")
        t = t[t[0] == t[3]]
        span = t.groupby(0)[5].max().sum()
    else:
        span = t[5].max()
    return float(t[6].sum()) / max(span / 1e6, 1e-6)
