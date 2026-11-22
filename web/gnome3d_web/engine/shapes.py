"""The shapes of the answers on the settings a request may change."""

from __future__ import annotations

from typing_extensions import TypedDict


class Setting(TypedDict):
    section: str
    key: str
    type: str
    default: str
    server: str
    lo: float | None
    hi: float | None
    step: float | None
    choices: list[str] | None
    meaning: str


class SettingGroup(TypedDict):
    name: str
    settings: list[Setting]


class IniOverrides(TypedDict):
    overrides: dict[str, str]
    refused: list[str]
