"""The nine trio samples and where each one's files sit in the Drive folder.

Three families, each a father, a mother and a child. The Drive tree spells the family
differently in each subtree, so every spelling is recorded here rather than guessed at the call
site. Loops use CHS, PUR and YRI, the downsampling tree uses Han, Puerto and Yoruban, and the
stripes tree uses CH, PR and YO.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    """One individual. `name` is the id used everywhere downstream as the cell line name."""

    name: str
    role: str
    pop: str
    downsampling_dir: str
    stripes_prefix: str


SAMPLES: tuple[Sample, ...] = (
    Sample("HG00512", "father", "CHS", "Han", "CH_F"),
    Sample("HG00513", "mother", "CHS", "Han", "CH_M"),
    Sample("HG00514", "child", "CHS", "Han", "CH_D"),
    Sample("HG00731", "father", "PUR", "Puerto", "PR_F"),
    Sample("HG00732", "mother", "PUR", "Puerto", "PR_M"),
    Sample("HG00733", "child", "PUR", "Puerto", "PR_D"),
    Sample("GM19239", "father", "YRI", "Yoruban", "YO_F"),
    Sample("GM19238", "mother", "YRI", "Yoruban", "YO_M"),
    Sample("GM19240", "child", "YRI", "Yoruban", "YO_D"),
)

BY_NAME = {s.name: s for s in SAMPLES}

# The factor is spelled one way in the Loops and Peaks trees and another in the downsampling
# tree. Both are needed to build a path.
FACTOR_LOOPS = {"CTCF": "CTCF", "RNAPOL2": "RNAPOL2"}
FACTOR_DOWNSAMPLING = {"CTCF": "CTCF", "RNAPOL2": "RNAP2"}


def resolve(names: str | None) -> list[Sample]:
    """Turn a comma separated sample list into Sample records. Empty means all nine."""
    if not names:
        return list(SAMPLES)
    out: list[Sample] = []
    for n in names.split(","):
        n = n.strip()
        if n not in BY_NAME:
            raise SystemExit(f"unknown sample {n}, expected one of {', '.join(BY_NAME)}")
        out.append(BY_NAME[n])
    return out


def family(pop: str) -> list[Sample]:
    return [s for s in SAMPLES if s.pop == pop]


# --- array sharding -----------------------------------------------------------------------

CHROMS: tuple[str, ...] = tuple(f"chr{i}" for i in range(1, 23)) + ("chrX",)


def chunks_for(n_models: int, per_task: int) -> int:
    return (n_models + per_task - 1) // per_task


def array_length(chroms: list[str], samples: list[Sample], n_models: int, per_task: int) -> int:
    return len(chroms) * len(samples) * chunks_for(n_models, per_task)


def shard(
    task: int, chroms: list[str], samples: list[Sample], n_models: int, per_task: int
) -> tuple[str, Sample, int, int] | None:
    """Map an array index to a chromosome, a sample and a member range.

    Chromosome is the slowest varying dimension, so the first block of tasks covers one
    chromosome across every sample. A trio comparison on that chromosome is then possible
    while the rest of the genome is still queued, which is the point of the ordering.

    Returns None when the index is past the end.
    """
    chunks = chunks_for(n_models, per_task)
    per_chrom = len(samples) * chunks
    ci, rem = divmod(task, per_chrom)
    si, ck = divmod(rem, chunks)
    if ci >= len(chroms):
        return None
    first = ck * per_task
    return chroms[ci], samples[si], first, min(first + per_task - 1, n_models - 1)
