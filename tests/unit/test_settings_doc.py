"""Checks SETTINGS.md against the loader and the production config.

    pytest tests/unit/test_settings_doc.py

Every key the loader reads must have a row and every row must be a key the loader reads. The
default column must be the default of the loader. The production column must be the value of
`gnome3d.production.CANONICAL`, or the default where it has none. As a script the exit status
is the number of problems.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from gnome3d.production import CANONICAL
from gnome3d.settings import Settings

ROOT = Path(__file__).resolve().parents[2]

MC_SECTIONS = ("simulation_heatmap", "simulation_arcs", "simulation_arcs_smooth", "simulation_ib")


def loader_keys() -> dict[tuple[str, str], str]:
    """Returns (section, key) -> attribute name from every getter call of the loader."""
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
    """Returns (section, key) -> (default, production) from the tables of SETTINGS.md."""
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
    """Returns the value of one stage from a shared table cell such as `arcs and ib 100, smooth 50`."""
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


def problems() -> tuple[list[str], int, int]:
    """Returns every mismatch, the count of loader keys and the count of rows."""
    loader = loader_keys()
    doc = doc_rows()
    defaults = Settings()
    out: list[str] = []
    for sk in sorted(set(loader) - set(doc)):
        out.append(f"undocumented: [{sk[0]}] {sk[1]}")
    for sk in sorted(set(doc) - set(loader)):
        out.append(f"stale row: [{sk[0]}] {sk[1]} is not read by the loader")
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
            out.append(f"default: [{sk[0]}] {sk[1]} doc {d_doc!r} vs loader {d_real!r}")
        sec = CANONICAL.get(sk[0], {})
        if sk[1] in sec:
            want = sec[sk[1]]
            if p_doc and norm(per_stage(p_doc, sk[0])) != norm(want):
                out.append(f"production: [{sk[0]}] {sk[1]} doc {p_doc!r} vs config {want!r}")
        elif p_doc and d_doc and norm(p_doc) != norm(d_doc) and not p_doc.startswith("auto"):
            out.append(
                f"production: [{sk[0]}] {sk[1]} doc {p_doc!r} but the config sets nothing, default is {d_doc!r}"
            )
    return out, len(loader), len(doc)


def test_settings_doc_matches_the_loader_and_the_production_config() -> None:
    found, _keys, _rows = problems()
    assert not found, "\n".join(found)
