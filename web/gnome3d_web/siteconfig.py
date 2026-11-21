"""The configuration of the site, `config.json` beside the catalog.

    {"features": {"results": true, "method": true, "trios": false},
     "limits": {"region_mb": 20, "hicdiffusion_mb": 2}}

`results` and `method` are the two pages with their links and `trios` is the nine people on
the Results page with their RNA-seq in the form. A feature the file leaves out is on.

`region_mb` is the longest region a job may cover and `hicdiffusion_mb` the longest region a
job with a HiCDiffusion background may cover, both in Mb. A limit the file leaves out keeps the
site's default. The file is read again when it changes. The workers of the site, their cards
and their cores are variables of the environment, see settings.py.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from personal3d.hicdiffusion import SIZE as HICDIFFUSION_WINDOW

log = logging.getLogger(__name__)

# The parts of the site a feature flag can turn off.
FEATURES = ("results", "method", "trios")
# The limits of a job the file can set, two lengths in Mb.
LIMITS = ("region_mb", "hicdiffusion_mb")


class ConfigError(ValueError):
    """A config.json the site cannot read, with the reason as its message."""


def _all_on() -> dict[str, bool]:
    return dict.fromkeys(FEATURES, True)


@dataclass(frozen=True)
class SiteConfig:
    features: dict[str, bool] = field(default_factory=_all_on)
    # None keeps the default of the site.
    region_mb: float | None = None
    hicdiffusion_mb: float | None = None

    @property
    def hidden(self) -> tuple[str, ...]:
        """The features turned off, in the order of `FEATURES`."""
        return tuple(f for f in FEATURES if not self.features[f])


def parse(text: str, where: str = "config.json") -> SiteConfig:
    """Returns the configuration a file's text holds. An unknown key or feature and a value that is
    not true or false are errors. A misspelt flag does not pass for a working one."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        raise ConfigError(f"{where} is not JSON: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must hold an object")
    unknown = sorted(set(raw) - {"features", "limits"})
    if unknown:
        raise ConfigError(f"{where} has the unknown key {', '.join(unknown)}. It takes features and limits.")
    flags = raw.get("features", {})
    if not isinstance(flags, dict):
        raise ConfigError(f"{where}: features must be an object of true and false")
    unknown = sorted(set(flags) - set(FEATURES))
    if unknown:
        raise ConfigError(f"{where} names the feature {', '.join(unknown)}. It takes {', '.join(FEATURES)}.")
    for name, on in flags.items():
        if not isinstance(on, bool):
            raise ConfigError(f"{where}: the feature {name} must be true or false")
    limits = raw.get("limits", {})
    if not isinstance(limits, dict):
        raise ConfigError(f"{where}: limits must be an object of numbers")
    unknown = sorted(set(limits) - set(LIMITS))
    if unknown:
        raise ConfigError(f"{where} names the limit {', '.join(unknown)}. It takes {', '.join(LIMITS)}.")
    for name, value in limits.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ConfigError(f"{where}: the limit {name} must be a number of Mb above zero")
    window_mb = HICDIFFUSION_WINDOW / 1e6
    if limits.get("hicdiffusion_mb", 0) > window_mb:
        raise ConfigError(
            f"{where}: the limit hicdiffusion_mb is {limits['hicdiffusion_mb']:g}. HiCDiffusion predicts one window of "
            f"{HICDIFFUSION_WINDOW:,} bp, and a longer region would get the prediction on its first part alone. Give {window_mb:g} or less."
        )
    return SiteConfig(
        features={**_all_on(), **flags},
        region_mb=limits.get("region_mb"),
        hicdiffusion_mb=limits.get("hicdiffusion_mb"),
    )


class ConfigFile:
    """The configuration of a path, read again when the file changes. The first read raises on
    a broken file. A later broken edit is logged and the last good configuration stays.
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
