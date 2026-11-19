"""Logging for the package through one stdlib handler.

A serial run indents each line by its scope. A parallel run prefixes each line with its full
scope path. `output_level` 0, 1 and 2 show STATUS, INFO and DEBUG. A backend that must sync a
device to log a value guards that work with `LOG.isEnabledFor(logging.DEBUG)`.
"""

from __future__ import annotations

import contextvars
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager

_ROOT = "gnome3d"
_SEP = " ▸ "  # joins scope tags

# A level between INFO and WARNING for the run milestones that show at `output_level` 0.
STATUS = 25
logging.addLevelName(STATUS, "STATUS")

# The scope stack of the current thread. A new worker thread starts with an empty stack.
_scope: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "gnome3d_scope", default=()
)

# Set by the main thread before the workers start and restored after they join.
_parallel: bool = False


def get(name: str) -> logging.Logger:
    """Returns the package logger of a short module name such as "mc.jax"."""
    return logging.getLogger(f"{_ROOT}.{name}")


def current() -> str:
    """Returns the active scope path, "" at the top level. The JAX kernels seed from it."""
    return _SEP.join(_scope.get())


def status(logger: logging.Logger, msg: str, *args: object) -> None:
    """Logs a run milestone at STATUS."""
    logger.log(STATUS, msg, *args)


@contextmanager
def scope(tag: str) -> Iterator[None]:
    """Pushes `tag` on the scope stack for the block and logs nothing."""
    token = _scope.set(_scope.get() + (tag,))
    try:
        yield
    finally:
        _scope.reset(token)


@contextmanager
def step(
    logger: logging.Logger,
    tag: str,
    msg: str | None = None,
    *args: object,
    level: int = logging.INFO,
) -> Iterator[None]:
    """Enters the scope `tag` and logs it.

    A serial run logs the tag and `msg` at the parent depth. A parallel run logs `msg` alone
    inside the scope and logs nothing without one.
    """
    enabled = logger.isEnabledFor(level)
    if enabled and not _parallel:
        if msg:
            logger.log(level, "%s  " + msg, tag, *args)
        else:
            logger.log(level, "%s", tag)
    token = _scope.set(_scope.get() + (tag,))
    try:
        if enabled and _parallel and msg:
            logger.log(level, msg, *args)
        yield
    finally:
        _scope.reset(token)


@contextmanager
def parallel(on: bool = True) -> Iterator[None]:
    """Switches to flat lines with scope prefixes for the block."""
    global _parallel
    prev = _parallel
    _parallel = on
    try:
        yield
    finally:
        _parallel = prev


def _short(name: str) -> str:
    """Returns "mc.jax" for "gnome3d.mc.jax"."""
    return name[len(_ROOT) + 1 :] if name.startswith(_ROOT + ".") else name


class _ScopeFilter(logging.Filter):
    """Attaches the scope stack to each record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.scope = _scope.get()  # type: ignore[attr-defined]
        return True


class _AdaptiveFormatter(logging.Formatter):
    """Renders indented lines in a serial run and scope prefixed lines in a parallel run."""

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        sc: tuple[str, ...] = getattr(record, "scope", ())
        if _parallel:
            tag = _SEP.join(sc) if sc else _short(record.name)
            return f"[{tag}] {msg}"
        if not sc:
            return f"[{_short(record.name)}] {msg}"
        return f"{'  ' * len(sc)}{msg}"


class _FileFormatter(logging.Formatter):
    """Renders timestamp, level, module, scope path and message for the log file."""

    def __init__(self) -> None:
        super().__init__(datefmt="%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        sc: tuple[str, ...] = getattr(record, "scope", ())
        path = _SEP.join(sc)
        ts = self.formatTime(record, self.datefmt)
        return f"{ts} {record.levelname:<5} {_short(record.name):<8} [{path}] {record.getMessage()}"


_LEVELS = {0: STATUS, 1: logging.INFO, 2: logging.DEBUG}


def setup(output_level: int, *, log_file: str | None = None) -> None:
    """Configures the one `gnome3d` handler. A second call replaces the handlers.

    `output_level` 0, 1 or 2 sets the console to STATUS, INFO or DEBUG. `log_file` adds a file
    handler at DEBUG.
    """
    level = _LEVELS.get(int(output_level), logging.INFO)
    root = logging.getLogger(_ROOT)
    root.setLevel(logging.DEBUG)  # the handlers filter
    root.propagate = False
    for h in list(root.handlers):
        root.removeHandler(h)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.addFilter(_ScopeFilter())
    console.setFormatter(_AdaptiveFormatter())
    root.addHandler(console)

    if log_file:
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(logging.DEBUG)
        fh.addFilter(_ScopeFilter())
        fh.setFormatter(_FileFormatter())
        root.addHandler(fh)


# A default for an embedder that does not call `setup`.
setup(0)
