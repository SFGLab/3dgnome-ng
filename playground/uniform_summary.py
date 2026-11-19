"""One table of every arm of the uniform chain trials.

    python playground/uniform_summary.py FIGURES_DIR [OUT.md]

FIGURES_DIR holds the `scores-<tag>` folders of the trials, as
`design/figures/uniform-2026-11-19/`. An arm is listed once per set of rows it was scored on,
with the mean of the pooled scores over those rows and, where a trial has `overlaps.txt`, the
close pairs per thousand beads. The default chain of the same rows is beside each arm.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uniform_arms_figure import ARMS, read_scores  # noqa: E402

LABEL = {a[0]: a[1] for a in ARMS}
COLS = [
    ("raw_scc", "SCC raw"),
    ("bal_scc", "SCC balanced"),
    ("raw_scc_mapped", "SCC raw, mapped"),
    ("bal_scc_mapped", "SCC balanced, mapped"),
    ("raw_sp", "Spearman raw"),
    ("bal_p", "Pearson balanced"),
    ("exponent", "exponent over Hi-C"),
]


def overlaps(path: Path) -> dict[str, tuple[float, float]]:
    """Returns {arm: (every pair, not next to each other)} from the mean lines of a check."""
    out: dict[str, tuple[float, float]] = {}
    if not path.is_file():
        return out
    lines = path.read_text().splitlines()
    if "mean over the rows" not in lines:
        return out
    for line in lines[lines.index("mean over the rows") + 1 :]:
        f = line.split()
        if len(f) == 3:
            out[f[0]] = (float(f[1]), float(f[2]))
    return out


def summary(figures: Path) -> str:
    seen: dict[tuple[str, int], tuple[str, dict[str, float], tuple[float, float] | None]] = {}
    base: dict[int, dict[str, float]] = {}
    for scores in sorted(figures.glob("scores*")):
        data = read_scores(scores)
        if not data:
            continue
        close = overlaps(scores / "overlaps.txt")
        arms = sorted({a for d in data.values() for a in d})
        for arm in arms:
            rows = [d[arm] for d in data.values() if arm in d]
            mean = {
                k: sum(r[k] for r in rows if k in r) / max(sum(1 for r in rows if k in r), 1)
                for k, _ in COLS
            }
            key = (arm, len(rows))
            if arm == "prod":
                base[len(rows)] = mean
            if key not in seen or (seen[key][2] is None and arm in close):
                seen[key] = (scores.name.replace("scores-", "").replace("scores", "first"), mean, close.get(arm))
    head = "| arm | trial | rows | " + " | ".join(c for _, c in COLS) + " | close pairs | not next |"
    lines = [head, "|---|---|---|" + "---|" * (len(COLS) + 2)]
    order = sorted(seen, key=lambda k: (k[1], -seen[k][1]["raw_scc"] - seen[k][1]["bal_scc"]))
    for arm, n in order:
        tag, mean, close = seen[(arm, n)]
        vals = " | ".join(f"{mean[k]:.3f}" for k, _ in COLS)
        cp = f"{close[0]:.1f} | {close[1]:.1f}" if close else " | "
        lines.append(f"| {LABEL.get(arm, arm)} | {tag} | {n} | {vals} | {cp} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    text = summary(Path(sys.argv[1]))
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(text)
    print(text)
