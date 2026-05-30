"""
Centralized logging for 3dgnome-ng.

Every diagnostic line in the package flows through the stdlib `logging`
module to a single handler configured once by `setup()`.

Two render styles, chosen automatically by `parallel()`:

  * SERIAL  (the common ``ib_workers=1`` case) — nested 2-space indentation,
    the familiar look::

        [solver] chr1 IB 4/52  (2 anchors)
          arcs 1/1
            mc_arcs: jax  N=2  terms=[arcs+EV]

  * PARALLEL (``ib_workers>1`` or ``n_structures>1``) — no indentation;
    every line is prefixed with its full scope path so interleaved workers
    stay attributable::

        [chr1 IB 4/52 ▸ arcs 1/1] mc_arcs: jax  N=2  terms=[arcs+EV]
        [chr1 IB 7/52 ▸ smooth 1/1] step 1,200,000  score=0.81  [done]

Thread model
------------
The scope stack is a `contextvars.ContextVar`, which is per-thread.
`ThreadPoolExecutor` workers start with an empty stack, so each worker
re-establishes its own root scope (the IB / structure label) at the top of
its task — no inheritance needed, and no cross-thread scope bleed.  The
stdlib handler locks around each `emit`, so whole lines never interleave.

Visibility is driven by log level (mapped from the legacy ``output_level``
config knob), not by passing ``log1``/``log2``/``verbose`` flags down the
call stack:

    output_level 0 -> STATUS    (quiet: run banner + warnings only)
    output_level 1 -> INFO      (milestones + per-phase headers)
    output_level 2 -> DEBUG     (per-batch MC step lines)

Backends that must do real work to log (e.g. a JAX device->host sync to read
a score) should guard that work with ``LOG.isEnabledFor(logging.DEBUG)``.
"""

from __future__ import annotations

import contextvars
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager

_ROOT = "gnome3d"
_SEP = " ▸ "  # ▸  joins scope tags in the parallel/file path prefix

# A tier between INFO and WARNING for the handful of top-level run milestones
# (config banner, output paths, backend coercions) that should survive even
# the quiet ``output_level=0`` default — these always printed unconditionally
# before the migration.  output_level 0 maps the console to this threshold, so
# STATUS shows while solver(INFO) and MC-step(DEBUG) detail stay hidden.
STATUS = 25
logging.addLevelName(STATUS, "STATUS")

# Per-context scope stack.  Empty at the top level and in every fresh worker
# thread; grown by step()/scope().  Read by the filter on every record.
_scope: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "gnome3d_scope", default=()
)

# Render mode.  Toggled by parallel() around a worker pool.  A plain module
# global (not a ContextVar): it is set by the main thread before workers are
# submitted and read by them while they run, then restored after the join.
_parallel: bool = False


def get(name: str) -> logging.Logger:
    """Return the package logger for a short module name (``"solver"``,
    ``"main"``, ``"mc"``, ``"mc.jax"``, ...).  All live under ``gnome3d.``."""
    return logging.getLogger(f"{_ROOT}.{name}")


def current() -> str:
    """The active scope path as a single string (``"chr1 IB 4/52 ▸ smooth 1/1"``),
    or ``""`` at the top level.  Used both as a display fallback and as a
    deterministic RNG-seed source where a per-call label is needed."""
    return _SEP.join(_scope.get())


def status(logger: logging.Logger, msg: str, *args: object) -> None:
    """Emit an always-on run milestone (see `STATUS`)."""
    logger.log(STATUS, msg, *args)


@contextmanager
def scope(tag: str) -> Iterator[None]:
    """Push `tag` onto the scope stack for the duration of the block, without
    emitting an announcement line.  Use `step()` when you also want a header."""
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
    """Enter a scope `tag` and announce it, then run the block one level deeper.

    The announcement renders per mode (this is the one place that knows the
    difference, so call sites stay style-agnostic):

      * SERIAL — emitted at the *parent* depth, with the tag shown inline:
        ``<indent>[module?] <tag>[  <msg>]``.
      * PARALLEL — emitted *inside* the scope (the path prefix already carries
        the tag), so only `msg` is shown; with no `msg`, nothing is emitted —
        the next real line will carry the path anyway.
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
    """Switch the formatter to flat per-line scope tags for the block (used
    around a worker pool).  Restores the previous mode on exit."""
    global _parallel
    prev = _parallel
    _parallel = on
    try:
        yield
    finally:
        _parallel = prev


def _short(name: str) -> str:
    """``gnome3d.mc.jax`` -> ``mc.jax``."""
    return name[len(_ROOT) + 1 :] if name.startswith(_ROOT + ".") else name


class _ScopeFilter(logging.Filter):
    """Attach the live scope stack to each record so formatters can render it
    without reaching into the contextvar themselves."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.scope = _scope.get()  # type: ignore[attr-defined]
        return True


class _AdaptiveFormatter(logging.Formatter):
    """Indented-when-serial / tagged-when-parallel console renderer."""

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        sc: tuple[str, ...] = getattr(record, "scope", ())
        if _parallel:
            tag = _SEP.join(sc) if sc else _short(record.name)
            return f"[{tag}] {msg}"
        # serial: 2-space indent per depth; module bracket only at top level
        if not sc:
            return f"[{_short(record.name)}] {msg}"
        return f"{'  ' * len(sc)}{msg}"


class _FileFormatter(logging.Formatter):
    """Full-detail, parseable line for the optional file sink: timestamp,
    level, module, scope path, message.  Always emitted regardless of the
    console level — the file is the after-the-fact record of a parallel run."""

    def __init__(self) -> None:
        super().__init__(datefmt="%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        sc: tuple[str, ...] = getattr(record, "scope", ())
        path = _SEP.join(sc)
        ts = self.formatTime(record, self.datefmt)
        return f"{ts} {record.levelname:<5} {_short(record.name):<8} [{path}] {record.getMessage()}"


_LEVELS = {0: STATUS, 1: logging.INFO, 2: logging.DEBUG}


def setup(output_level: int, *, log_file: str | None = None) -> None:
    """Configure the single ``gnome3d`` sink.  Idempotent — safe to call again
    (handlers are replaced, not stacked).

    `output_level` maps to the console handler's level (0/1/2 ->
    STATUS/INFO/DEBUG).  `log_file`, if given, adds a DEBUG file handler with
    the structured formatter — handy for reconstructing parallel runs.
    """
    level = _LEVELS.get(int(output_level), logging.INFO)
    root = logging.getLogger(_ROOT)
    root.setLevel(logging.DEBUG)  # handlers do the filtering
    root.propagate = False  # don't bubble to the stdlib root handler
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


# Install a quiet default at import so the package always logs somewhere even
# if an embedder (a script, a test) never calls setup() — STATUS + warnings to
# stdout, matching the output_level=0 default.  cli.setup() replaces this.
setup(0)
