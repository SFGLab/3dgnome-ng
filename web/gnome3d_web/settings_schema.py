"""The engine settings a user may change on the form. Types and defaults are read from
SETTINGS.md and the text shown under a setting is `HELP`.

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

# The text the form shows under each setting, for a reader who wants to know what the setting
# does to a model. SETTINGS.md is the engineering reference and keeps the measurements and the
# history, which do not belong here.
HELP: dict[str, str] = {
    "distance.polymer_exponent": "Exponent of the law that turns genomic separation into 3D distance. 0 measures it from the contact data of the run. A positive value fixes it.",
    "distance.contact_half_saturation": "Loop strength at which a loop pulls its two anchors halfway from their distance without a loop to contact. A smaller value pulls weak loops closer.",
    "springs.stretch_constant": "Stiffness of the bond between neighbouring beads once it is stretched past its target length.",
    "springs.squeeze_constant": "Stiffness of the bond between neighbouring beads once it is shorter than its target length.",
    "springs.arc_weight_exponent": "Makes strong loops stiffer. The spring of a loop is multiplied by its strength to this power. 0 gives every loop the same spring.",
    "motif_orientation.weight": "Weight of the term that turns the CTCF motifs at the two anchors of a loop toward each other. 0 turns it off.",
    "springs.loop_dropout": "Each structure keeps a loop with a probability that grows with the strength of the loop. A strong loop is then present in more structures of the ensemble than a weak one. Off, every structure has every loop.",
    "springs.loop_contact": "A loop that a structure keeps brings its two anchors into contact, in place of a distance set by its strength. Meant for use with loop_dropout.",
    "springs.background_weight": "Strength of a weak spring that holds two nearby anchors without a loop at the distance the law gives for their separation. 0 turns it off.",
    "springs.background_range_bp": "Largest genomic separation, in bp, of an anchor pair that the background_weight spring holds.",
    "springs.use_contact_background": "Beyond background_range_bp, pulls an anchor pair without a loop closer if the contact map has more contacts than expected at its separation.",
    "excluded_volume.arcs_repulsion": "Pushes apart every pair of anchors that has no loop. Off, distant anchors are free to meet and gyration_weight keeps the size of the model.",
    "excluded_volume.arcs_repulsion_cutoff_factor": "Reach of arcs_repulsion, as a multiple of the mean loop distance. Anchors farther apart are not pushed. 0 sets no limit.",
    "springs.gyration_weight": "Holds every stretch of the region at the size the distance law gives for its length. It sets how compact the model is and fixes no pair of anchors at a distance. 0 turns it off.",
    "holes.use_hole_beads": "Fills a stretch that no loop covers with beads. Off, such a stretch is left out of the structure.",
    "holes.flank_beads": "Beads on each side of a filled stretch that its starting shape is fitted to. More beads tie the stretch closer to its neighbours. 0 uses the two anchors alone.",
    "simulation_arcs.solver": "Method that places the loop anchors. lbfgs minimises the energy directly and is fast. mc is simulated annealing on the same energy and is slow.",
    "simulation_arcs.solver_tol": "Stop rule of the lbfgs solver. It stops once a step improves the energy by less than this fraction. 0 runs to the iteration limit.",
    "simulation_arcs.start": "Starting positions of the anchors. centroid puts them at one point. walk is a random walk. hilbert is a compact space filling curve. maxent draws a different start for each structure from the loops and the distance law.",
    "main.noise_smooth": "Size of one trial move of a bead between anchors, as a multiple of the mean bond length. Smaller moves settle the chain more finely and take longer.",
    "simulation_arcs_smooth.hard_wall": "Refuses a move that brings two beads closer than the excluded volume radius. Beads then cannot pass into each other.",
    "simulation_arcs_smooth.start": "Starting shape of the chain between two anchors. line is straight. coil is a compact random path.",
    "excluded_volume.use_excluded_volume": "Keeps beads from overlapping by pushing apart any two that come too close.",
    "excluded_volume.weight": "Strength of the excluded volume push.",
    "excluded_volume.skip_neighbors": "Beads this many places apart along the chain, or fewer, are not pushed apart.",
    "excluded_volume.auto_factor_arcs": "Distance under which two anchors count as overlapping, as a multiple of the mean loop distance.",
    "relax.use_cross_block_relax": "A last pass over the whole structure that moves apart overlapping beads of different loop blocks. The anchors stay in place.",
    "relax.within_blocks": "The last pass also moves apart overlapping beads inside one loop block.",
    "confinement.use_confinement": "Keeps the structure inside a sphere, as a nucleus would.",
    "confinement.weight": "Strength of the pull back into the sphere.",
    "compartments.use_compartments": "Draws beads of the same compartment, A or B, toward each other. The compartments come from the Hi-C map of the cell type. A job takes many times longer with it.",
    "compartments.weight": "Strength of the attraction between beads of one compartment.",
}


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
    """Writes the changes into an ini file, replacing a key in its section or adding it there
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
    type's config, its range or choices and the text of `HELP`."""
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
                    "meaning": HELP[f"{e.section}.{e.key}"],
                }
            )
        groups.append({"name": name, "settings": settings})
    return groups
