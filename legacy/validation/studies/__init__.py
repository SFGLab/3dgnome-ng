"""Validation studies. Each study is a small class that registers itself for the CLI and the run
battery. A study declares its extra CLI args and a run method that does the work and prints its
report."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class Context:
    """Shared knobs every study receives. config and data are filled per region by the study or the
    reconstruction helpers. label names the reference output files."""

    cell: str
    data_root: str
    hic: str | None
    quality: str | None
    n: int
    py_arcs: str
    py_workers: int
    ref_workers: int
    fast: bool
    config: Any = None
    data: Any = None
    label: str = "v"


@runtime_checkable
class Study(Protocol):
    name: str
    help: str

    def add_args(self, p: ArgumentParser) -> None: ...
    def run(self, ctx: Context, args: Namespace) -> None: ...


_REGISTRY: dict[str, Study] = {}


def register(study: Study) -> None:
    _REGISTRY[study.name] = study


def all_studies() -> dict[str, Study]:
    return dict(_REGISTRY)
