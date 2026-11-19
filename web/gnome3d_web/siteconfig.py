"""The site's own configuration, `config.json` beside the catalog.

    {"features": {"results": true, "method": true, "trios": false}}

`features` turns parts of the site on and off. `results` and `method` are the two pages with
their links and `trios` is the nine people on the Results page. A feature the file leaves out
is on and a site with no file shows everything. The file is read again when it changes. An
edit needs no restart.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# The parts of the site a feature flag can turn off.
FEATURES = ("results", "method", "trios")


class ConfigError(ValueError):
    """A config.json the site cannot read, with the reason as its message."""


def _all_on() -> dict[str, bool]:
    return dict.fromkeys(FEATURES, True)


@dataclass(frozen=True)
class SiteConfig:
    features: dict[str, bool] = field(default_factory=_all_on)

    @property
    def hidden(self) -> tuple[str, ...]:
        """The features turned off, in the order of `FEATURES`."""
        return tuple(f for f in FEATURES if not self.features[f])


def parse(text: str, where: str = "config.json") -> SiteConfig:
    """The configuration a file's text holds. An unknown key or feature and a value that is not
    true or false are errors. A misspelt flag does not pass for a working one."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        raise ConfigError(f"{where} is not JSON: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must hold an object")
    unknown = sorted(set(raw) - {"features"})
    if unknown:
        raise ConfigError(f"{where} has the unknown key {', '.join(unknown)}. It takes features.")
    flags = raw.get("features", {})
    if not isinstance(flags, dict):
        raise ConfigError(f"{where}: features must be an object of true and false")
    unknown = sorted(set(flags) - set(FEATURES))
    if unknown:
        raise ConfigError(f"{where} names the feature {', '.join(unknown)}. It takes {', '.join(FEATURES)}.")
    for name, on in flags.items():
        if not isinstance(on, bool):
            raise ConfigError(f"{where}: the feature {name} must be true or false")
    return SiteConfig(features={**_all_on(), **flags})


class ConfigFile:
    """The configuration of a path, read again when the file changes.

    The first read raises on a file the site cannot read. A site does not start on a broken
    configuration. A later edit that breaks the file is logged and the last good configuration
    stays in use.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stamp: tuple[int, int] | None = None
        self._config = SiteConfig()
        self._read(strict=True)

    def _read(self, strict: bool) -> None:
        try:
            st = self.path.stat()
        except FileNotFoundError:
            self._stamp, self._config = None, SiteConfig()
            return
        stamp = (st.st_mtime_ns, st.st_size)
        if stamp == self._stamp:
            return
        try:
            self._config = parse(self.path.read_text(), str(self.path))
        except ConfigError as e:
            if strict:
                raise
            log.error("%s The site keeps the configuration it had.", e)
        self._stamp = stamp

    def get(self) -> SiteConfig:
        self._read(strict=False)
        return self._config
