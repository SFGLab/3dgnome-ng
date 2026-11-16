"""The settings users may change: the schema from SETTINGS.md, validation, and the config they
are written into.

    python web/tests/test_settings_schema.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from _check import check, done
from fixture import make_data

from gnome3d_web.catalog import Catalog
from gnome3d_web.jobs import JobRequest, request_key
from gnome3d_web.settings_schema import EDITABLE, SettingError, apply_overrides, parse_ini, read_settings_md, schema, validate_overrides

ROOT = Path(__file__).resolve().parents[2]


def refused(raw: dict[str, str]) -> str:
    try:
        validate_overrides(raw)
    except SettingError as e:
        return str(e)
    return ""


def main() -> int:
    rows = read_settings_md(ROOT / "SETTINGS.md")
    by = {(r.section, r.key): r for r in rows}
    check("SETTINGS.md has 186 settings", len(rows) == 186, str(len(rows)))
    check("polymer_exponent is in [distance]", ("distance", "polymer_exponent") in by)
    missing = [f"{e.section}.{e.key}" for group in EDITABLE.values() for e in group if (e.section, e.key) not in by]
    check("every editable setting is in SETTINGS.md with its section", not missing, str(missing))

    check("a value in range passes", validate_overrides({"distance.polymer_exponent": "0.3"}) == {"distance.polymer_exponent": "0.3"})
    msg = refused({"distance.polymer_exponent": "0.9"})
    check("a value out of range is refused with the range", "polymer_exponent" in msg and "0 to 0.6" in msg, msg)
    msg = refused({"simulation_backend.mc_executor_smooth": "serial"})
    check("a key off the list is refused", "not a setting users may change" in msg, msg)
    msg = refused({"springs.loop_dropout": "maybe"})
    check("a boolean takes yes or no", "yes or no" in msg, msg)
    check("a boolean is written as yes or no", validate_overrides({"springs.loop_dropout": "true"}) == {"springs.loop_dropout": "yes"})

    with tempfile.TemporaryDirectory() as d:
        cat = Catalog.load(make_data(Path(d) / "data"))
        cfg = Path(d) / "config.ini"
        shutil.copy(cat.cell_types["GMX"].path("config"), cfg)
        before = cfg.read_text()
        apply_overrides(cfg, {"distance.polymer_exponent": "0.3", "main.noise_smooth": "2"})
        after = cfg.read_text()
        check("the overrides are written under their sections", "[distance]\npolymer_exponent = 0.3" in after and "noise_smooth = 2" in after, after)
        check("the rest of the config is kept", all(line in after for line in before.splitlines() if line.strip()), after)

        groups = schema(cat.cell_types["GMX"].path("config"))
        names = [g["name"] for g in groups]
        check("the groups follow the design", names[:3] == ["Distance law", "Loops", "Pairs without a loop"], str(names))
        pe = next(s for g in groups for s in g["settings"] if s["key"] == "polymer_exponent")
        check("a setting carries its default, server value, range and meaning", pe["default"] == "0.0" and pe["server"] == "0.0" and pe["hi"] == 0.6 and pe["meaning"], str(pe))

        ok, bad = parse_ini("[distance]\npolymer_exponent = 0.2\n[data]\nanchors = x.bed\n")
        check("an uploaded ini keeps the editable keys", ok == {"distance.polymer_exponent": "0.2"}, str(ok))
        check("an uploaded ini lists the refused keys", len(bad) == 1 and "anchors" in bad[0], str(bad))

        a = JobRequest(cell_type="GMX", region="chr1:0-2000000", factors="CTCF", overrides={"distance.polymer_exponent": "0.3", "main.noise_smooth": "2"})
        b = JobRequest(cell_type="GMX", region="chr1:0-2000000", factors="CTCF", overrides={"main.noise_smooth": "2", "distance.polymer_exponent": "0.3"})
        c = JobRequest(cell_type="GMX", region="chr1:0-2000000", factors="CTCF", overrides={"distance.polymer_exponent": "0.31", "main.noise_smooth": "2"})
        check("the order of the overrides does not change the key", request_key(a, cat) == request_key(b, cat))
        check("a changed value changes the key", request_key(a, cat) != request_key(c, cat))
    return done()


if __name__ == "__main__":
    sys.exit(main())
