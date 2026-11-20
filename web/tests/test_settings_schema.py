"""The settings users may change: the schema from SETTINGS.md, validation and the config they
are written into.

    pytest web/tests/test_settings_schema.py
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from fixture import make_data

from gnome3d_web.catalog import Catalog
from gnome3d_web.jobs import JobRequest, request_key
from gnome3d_web.settings_schema import (
    EDITABLE,
    SettingError,
    apply_overrides,
    parse_ini,
    read_settings_md,
    schema,
    validate_overrides,
)

ROOT = Path(__file__).resolve().parents[2]


def refused(raw: dict[str, str]) -> str:
    try:
        validate_overrides(raw)
    except SettingError as e:
        return str(e)
    return ""


def test_settings_md_has_190_settings() -> None:
    rows = read_settings_md(ROOT / "SETTINGS.md")
    by = {(r.section, r.key): r for r in rows}
    assert len(rows) == 190, "SETTINGS.md has 190 settings: " + str(len(rows))
    assert ("distance", "polymer_exponent") in by, "polymer_exponent is in [distance]"
    missing = [f"{e.section}.{e.key}" for group in EDITABLE.values() for e in group if (e.section, e.key) not in by]
    assert not missing, "every editable setting is in SETTINGS.md with its section: " + str(missing)


def test_a_value_in_range_passes() -> None:
    assert validate_overrides({"distance.polymer_exponent": "0.3"}) == {"distance.polymer_exponent": "0.3"}, "a value in range passes"
    msg = refused({"distance.polymer_exponent": "0.9"})
    assert "polymer_exponent" in msg and "0 to 0.6" in msg, "a value out of range is refused with the range: " + str(msg)
    msg = refused({"simulation_backend.mc_executor_smooth": "serial"})
    assert "not a setting users may change" in msg, "a key off the list is refused: " + str(msg)
    msg = refused({"springs.loop_dropout": "maybe"})
    assert "yes or no" in msg, "a boolean takes yes or no: " + str(msg)
    assert validate_overrides({"springs.loop_dropout": "true"}) == {"springs.loop_dropout": "yes"}, "a boolean is written as yes or no"


@pytest.fixture(scope="module")
def after_and_before_and_cat(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    d = str(tmp_path_factory.mktemp("work"))
    cat = Catalog.load(make_data(Path(d) / "data"))
    cfg = Path(d) / "config.ini"
    shutil.copy(cat.cell_types["GMX"].path("config"), cfg)
    before = cfg.read_text()
    apply_overrides(cfg, {"distance.polymer_exponent": "0.3", "main.noise_smooth": "2"})
    after = cfg.read_text()
    return SimpleNamespace(after=after, before=before, cat=cat)


def test_the_overrides_are_written_under_their_sections(after_and_before_and_cat: SimpleNamespace) -> None:
    after, before = after_and_before_and_cat.after, after_and_before_and_cat.before
    assert "[distance]\npolymer_exponent = 0.3" in after and "noise_smooth = 2" in after, "the overrides are written under their sections: " + str(after)
    assert all(line in after for line in before.splitlines() if line.strip()), "the rest of the config is kept: " + str(after)


def test_the_groups_follow_the_design(after_and_before_and_cat: SimpleNamespace) -> None:
    cat = after_and_before_and_cat.cat
    groups = schema(cat.cell_types["GMX"].path("config"))
    names = [g["name"] for g in groups]
    assert names[:3] == ["Distance law", "Loops", "Pairs without a loop"], "the groups follow the design: " + str(names)
    pe = next(s for g in groups for s in g["settings"] if s["key"] == "polymer_exponent")
    assert pe["default"] == "0.0" and pe["server"] == "0.0" and pe["hi"] == 0.6 and pe["meaning"], "a setting carries its default, server value, range and meaning: " + str(pe)
    import re as _re

    from gnome3d_web import settings_schema as _schema

    keys = {f"{e.section}.{e.key}" for group in _schema.EDITABLE.values() for e in group}
    assert keys == set(_schema.HELP), "every editable setting has its own text and no text is left over: " + str(keys ^ set(_schema.HELP))
    shown = [x["meaning"] for g in groups for x in g["settings"]]
    assert all(20 <= len(t) <= 240 and t.endswith(".") for t in shown), "a text is one to three short sentences: " + str([t for t in shown if not (20 <= len(t) <= 240)])
    history = _re.compile(r"\b20\d\d\b|production|byte exact|measured|under measurement|design/|since|idea \d|null|Master switch", _re.I)
    assert not [t for t in shown if history.search(t)], "a text states what the setting does, without history: " + str([t for t in shown if history.search(t)])
    rules = _re.compile(r", so |, and |because|\bwhich\b.*\bwhich\b|^(When|Where|What|Why|How|Which)\b|[;:]|—| - ")
    assert not [t for t in shown if rules.search(t)], "a text keeps to the writing rules: " + str([t for t in shown if rules.search(t)])


def test_an_uploaded_ini_keeps_the_editable_keys() -> None:
    ok, bad = parse_ini("[distance]\npolymer_exponent = 0.2\n[data]\nanchors = x.bed\n")
    assert ok == {"distance.polymer_exponent": "0.2"}, "an uploaded ini keeps the editable keys: " + str(ok)
    assert len(bad) == 1 and "anchors" in bad[0], "an uploaded ini lists the refused keys: " + str(bad)


def test_the_order_of_the_overrides_does_not_change_the_key(after_and_before_and_cat: SimpleNamespace) -> None:
    cat = after_and_before_and_cat.cat
    a = JobRequest(cell_type="GMX", region="chr1:0-2000000", factors="CTCF", overrides={"distance.polymer_exponent": "0.3", "main.noise_smooth": "2"})
    b = JobRequest(cell_type="GMX", region="chr1:0-2000000", factors="CTCF", overrides={"main.noise_smooth": "2", "distance.polymer_exponent": "0.3"})
    c = JobRequest(cell_type="GMX", region="chr1:0-2000000", factors="CTCF", overrides={"distance.polymer_exponent": "0.31", "main.noise_smooth": "2"})
    assert request_key(a, cat) == request_key(b, cat), "the order of the overrides does not change the key"
    assert request_key(a, cat) != request_key(c, cat), "a changed value changes the key"
