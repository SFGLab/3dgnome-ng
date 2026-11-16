"""The engine settings a user may change on the form, read from SETTINGS.md.

Every other setting keeps the value of the cell type's config. A change is validated against
its range here and written into the job's config.ini before the engine starts.
"""

from __future__ import annotations

import configparser
import re
from dataclasses import dataclass
from pathlib import Path

SETTINGS_MD = Path(__file__).resolve().parents[2] / "SETTINGS.md"
BOOLEAN = ("yes", "no")
_TRUE = {"yes", "true", "1", "on"}
_FALSE = {"no", "false", "0", "off"}


class SettingError(ValueError):
    """A setting change the form does not accept, with the key and the allowed values."""


@dataclass(frozen=True)
class SettingRow:
    section: str
    key: str
    type: str
    default: str
    production: str
    meaning: str


@dataclass(frozen=True)
class Editable:
    section: str
    key: str
    lo: float | None = None
    hi: float | None = None
    step: float | None = None
    choices: tuple[str, ...] | None = None


def _n(section: str, key: str, lo: float, hi: float, step: float) -> Editable:
    return Editable(section, key, lo, hi, step)


def _c(section: str, key: str, *choices: str) -> Editable:
    return Editable(section, key, choices=choices or BOOLEAN)


EDITABLE: dict[str, list[Editable]] = {
    "Distance law": [
        _n("distance", "polymer_exponent", 0, 0.6, 0.01),
        _n("distance", "contact_half_saturation", 0.1, 5, 0.1),
    ],
    "Loops": [
        _n("springs", "stretch_constant", 0.01, 1, 0.01),
        _n("springs", "squeeze_constant", 0.01, 1, 0.01),
        _n("springs", "arc_weight_exponent", 0, 2, 0.1),
        _n("motif_orientation", "weight", 0, 200, 1),
        _c("springs", "loop_dropout"),
        _c("springs", "loop_contact"),
    ],
    "Pairs without a loop": [
        _n("springs", "background_weight", 0, 1, 0.01),
        _n("springs", "background_range_bp", 0, 1_000_000, 10_000),
        _c("springs", "use_contact_background"),
        _c("excluded_volume", "arcs_repulsion"),
        _n("excluded_volume", "arcs_repulsion_cutoff_factor", 0, 3, 0.1),
        _n("springs", "gyration_weight", 0, 10, 0.1),
    ],
    "Stretches without loops": [
        _c("holes", "use_hole_beads"),
        _n("holes", "flank_beads", 0, 20, 1),
    ],
    "Anchor fit": [
        _c("simulation_arcs", "solver", "mc", "lbfgs"),
        _n("simulation_arcs", "solver_tol", 0, 1e-3, 1e-6),
        _c("simulation_arcs", "start", "centroid", "walk", "hilbert", "maxent"),
    ],
    "Chains": [
        _n("main", "noise_smooth", 0.1, 5, 0.1),
        _c("simulation_arcs_smooth", "hard_wall"),
        _c("simulation_arcs_smooth", "start", "line", "coil"),
    ],
    "Excluded volume": [
        _c("excluded_volume", "use_excluded_volume"),
        _n("excluded_volume", "weight", 0, 10, 0.1),
        _n("excluded_volume", "skip_neighbors", 0, 5, 1),
        _n("excluded_volume", "auto_factor_arcs", 0.1, 2, 0.1),
        _c("relax", "use_cross_block_relax"),
        _c("relax", "within_blocks"),
    ],
    "Confinement": [
        _c("confinement", "use_confinement"),
        _n("confinement", "weight", 0, 10, 0.1),
    ],
    "Compartments": [
        _c("compartments", "use_compartments"),
        _n("compartments", "weight", 0, 2, 0.1),
    ],
}

_BY_KEY = {f"{e.section}.{e.key}": e for group in EDITABLE.values() for e in group}


SCHEDULE_SECTIONS = ("simulation_heatmap", "simulation_arcs", "simulation_arcs_smooth", "simulation_ib")


def read_settings_md(path: Path = SETTINGS_MD) -> list[SettingRow]:
    """Reads every row of the settings tables, each with the ini section of its heading. A row of
    the shared schedule table counts once per schedule section, with the _heatmap suffix in
    [simulation_heatmap]. The [data] table has no production column."""
    rows: list[SettingRow] = []
    section = ""
    for line in path.read_text().splitlines():
        heading = re.match(r"#+ \[(\w+)\]", line)
        if heading:
            section = heading.group(1)
        elif line.startswith("## The MC schedule sections"):
            section = "*schedule"
        elif line.startswith("#"):
            section = ""
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")] if line.startswith("| `") else []
        if len(cells) < 4:
            continue
        key, typ = cells[0].strip("`"), cells[1]
        default, production, meaning = (cells[2], "", cells[3]) if len(cells) == 4 else (cells[2], cells[3], " | ".join(cells[4:]))
        meaning = meaning.replace("`", "")
        if section == "*schedule":
            for sec in SCHEDULE_SECTIONS:
                k = f"{key}_heatmap" if sec == "simulation_heatmap" else key
                rows.append(SettingRow(sec, k, typ, default, production, meaning))
        else:
            rows.append(SettingRow(section, key, typ, default, production, meaning))
    return rows


def _format(e: Editable, value: str) -> str:
    """Returns the value in the form the config takes. Raises SettingError outside the allowed values."""
    name = f"{e.section}.{e.key}"
    v = value.strip()
    if e.choices == BOOLEAN:
        if v.lower() in _TRUE:
            return "yes"
        if v.lower() in _FALSE:
            return "no"
        raise SettingError(f"{name} takes yes or no, not {value!r}")
    if e.choices:
        if v not in e.choices:
            raise SettingError(f"{name} takes {', '.join(e.choices)}, not {value!r}")
        return v
    try:
        x = float(v)
    except ValueError as err:
        raise SettingError(f"{name} takes a number from {e.lo:g} to {e.hi:g}, not {value!r}") from err
    assert e.lo is not None and e.hi is not None
    if not e.lo <= x <= e.hi:
        raise SettingError(f"{name} takes a number from {e.lo:g} to {e.hi:g}, not {value}")
    return v


def validate_overrides(raw: dict[str, str]) -> dict[str, str]:
    """Checks setting changes keyed section.key and returns them in the config's form.

    @param raw the changes from the form or an uploaded config
    @return the changes, booleans written as yes or no
    @raises SettingError for a key off the list or a value outside its range
    """
    out = {}
    for name, value in raw.items():
        e = _BY_KEY.get(name)
        if e is None:
            raise SettingError(f"{name} is not a setting users may change")
        out[name] = _format(e, str(value))
    return out


def apply_overrides(config: Path, overrides: dict[str, str]) -> None:
    """Writes the changes into an ini file, replacing a key in its section or adding it there,
    and keeps every other line."""
    lines = config.read_text().splitlines()
    for name, value in sorted(overrides.items()):
        section, key = name.split(".", 1)
        start = next((i for i, ln in enumerate(lines) if ln.strip() == f"[{section}]"), None)
        if start is None:
            lines += ["", f"[{section}]", f"{key} = {value}"]
            continue
        end = next((i for i in range(start + 1, len(lines)) if lines[i].strip().startswith("[")), len(lines))
        hit = next((i for i in range(start + 1, end) if re.match(rf"\s*{re.escape(key)}\s*[=:]", lines[i])), None)
        if hit is None:
            lines.insert(start + 1, f"{key} = {value}")
        else:
            lines[hit] = f"{key} = {value}"
    config.write_text("\n".join(lines) + "\n")


def parse_ini(text: str) -> tuple[dict[str, str], list[str]]:
    """Reads an uploaded config. Returns the changes the form accepts and one message per line
    it refuses."""
    cp = configparser.ConfigParser(interpolation=None)
    try:
        cp.read_string(text)
    except configparser.Error as err:
        return {}, [f"The file is not an ini file: {err}"]
    ok, refused = {}, []
    for section in cp.sections():
        for key, value in cp.items(section):
            try:
                ok.update(validate_overrides({f"{section}.{key}": value}))
            except SettingError as err:
                refused.append(str(err))
    return ok, refused


def schema(cell_config: Path) -> list[dict[str, object]]:
    """The editable settings in groups, each with its type, default, the value of the cell
    type's config, its range or choices and its meaning."""
    rows = {(r.section, r.key): r for r in read_settings_md()}
    cp = configparser.ConfigParser(interpolation=None)
    cp.read(cell_config)
    groups = []
    for name, items in EDITABLE.items():
        settings = []
        for e in items:
            r = rows[(e.section, e.key)]
            server = cp.get(e.section, e.key, fallback=r.default)
            if e.choices == BOOLEAN:
                server = "yes" if server.lower() in _TRUE else "no"
            settings.append(
                {
                    "section": e.section,
                    "key": e.key,
                    "type": r.type,
                    "default": r.default,
                    "server": server,
                    "lo": e.lo,
                    "hi": e.hi,
                    "step": e.step,
                    "choices": list(e.choices) if e.choices else None,
                    "meaning": r.meaning,
                }
            )
        groups.append({"name": name, "settings": settings})
    return groups
