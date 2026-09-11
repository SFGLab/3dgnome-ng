"""Check SETTINGS.md against the loader and the production config.

    python harness/check_settings_doc.py

Every key the loader reads must have a row, every row must be a key the loader reads, the
default column must be the loader's default and the production column must be what
`validation.core.config.CANONICAL` sets, or the default where it sets nothing. Exit status is
the number of problems.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gnome3d.settings import Settings  # noqa: E402
from validation.core.config import CANONICAL  # noqa: E402

MC_SECTIONS = ("simulation_heatmap", "simulation_arcs", "simulation_arcs_smooth", "simulation_ib")


def loader_keys() -> dict[tuple[str, str], str]:
    """(section, key) -> attribute name, from every getter call in the loader."""
    tree = ast.parse((ROOT / "gnome3d" / "settings.py").read_text())
    out: dict[tuple[str, str], str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not (
            isinstance(call.func, ast.Name)
            and call.func.id in ("getf", "getb", "geti", "gets", "getworkers")
        ):
            continue
        a = call.args
        if len(a) < 2 or not (isinstance(a[0], ast.Constant) and isinstance(a[1], ast.Constant)):
            continue
        target = node.targets[0]
        attr = target.attr if isinstance(target, ast.Attribute) else ""
        out[(str(a[0].value), str(a[1].value))] = attr
    return out


def doc_rows() -> dict[tuple[str, str], tuple[str, str]]:
    """(section, key) -> (default, production) from the reference's tables."""
    rows: dict[tuple[str, str], tuple[str, str]] = {}
    sections: list[str] = []
    for line in (ROOT / "SETTINGS.md").read_text().splitlines():
        m = re.match(r"^#{2,3} \[([a-z_]+)\]", line)
        if m:
            sections = [m.group(1)]
            continue
        if line.startswith("## The MC schedule sections"):
            sections = list(MC_SECTIONS)
            continue
        if line.startswith("## ") and not line.startswith("## ["):
            sections = []
            continue
        m = re.match(r"^\| `([A-Za-z_0-9]+)` \| [^|]+ \| ([^|]*) \| ([^|]*) \|", line)
        if not m or not sections:
            continue
        key, default, prod = m.group(1), m.group(2).strip(), m.group(3).strip()
        for sec in sections:
            k = key + "_heatmap" if sec == "simulation_heatmap" and len(sections) > 1 else key
            rows[(sec, k)] = (default, prod)
    return rows


STAGE_WORD = {
    "simulation_arcs": "arcs",
    "simulation_ib": "ib",
    "simulation_arcs_smooth": "smooth",
    "simulation_heatmap": "heatmap",
}


def per_stage(cell: str, section: str) -> str:
    """A shared table cell may spell one value per stage, as in `arcs and ib 100, smooth 50`."""
    word = STAGE_WORD.get(section)
    if word is None or not re.search(r"[a-z]+ [0-9]", cell):
        return cell
    for part in cell.split(","):
        names, _, value = part.strip().rpartition(" ")
        if word in re.split(r" and |, ", names):
            return value
    return cell


def norm(v: object) -> str:
    s = str(v).strip().strip("`")
    if s.lower() in ("yes", "true", "1") and not re.match(r"^\d+\.\d+$", s):
        return "yes" if s.lower() in ("yes", "true") else s
    if s.lower() in ("no", "false"):
        return "no"
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else repr(f)
    except ValueError:
        return s


def main() -> int:
    loader = loader_keys()
    doc = doc_rows()
    defaults = Settings()
    problems: list[str] = []
    for sk in sorted(set(loader) - set(doc)):
        problems.append(f"undocumented: [{sk[0]}] {sk[1]}")
    for sk in sorted(set(doc) - set(loader)):
        problems.append(f"stale row: [{sk[0]}] {sk[1]} is not read by the loader")
    for sk, attr in sorted(loader.items()):
        if sk not in doc or not attr or not hasattr(defaults, attr):
            continue
        d_doc, p_doc = doc[sk]
        d_real = getattr(defaults, attr)
        if (
            d_doc
            and norm(d_doc) != norm(d_real)
            and not isinstance(d_real, bool | type(None))
            or (isinstance(d_real, bool) and d_doc and norm(d_doc) != ("yes" if d_real else "no"))
        ):
            problems.append(f"default: [{sk[0]}] {sk[1]} doc {d_doc!r} vs loader {d_real!r}")
        sec = CANONICAL.get(sk[0], {})
        if sk[1] in sec:
            want = sec[sk[1]]
            if p_doc and norm(per_stage(p_doc, sk[0])) != norm(want):
                problems.append(f"production: [{sk[0]}] {sk[1]} doc {p_doc!r} vs config {want!r}")
        elif p_doc and d_doc and norm(p_doc) != norm(d_doc) and not p_doc.startswith("auto"):
            problems.append(
                f"production: [{sk[0]}] {sk[1]} doc {p_doc!r} but the config sets nothing, default is {d_doc!r}"
            )
    print(
        f"{len(loader)} keys read by the loader, {len(doc)} rows in SETTINGS.md, {len(problems)} problems"
    )
    for p in problems:
        print("  " + p)
    return len(problems)


if __name__ == "__main__":
    raise SystemExit(main())
