"""Composeable MC energy terms — the shared abstraction.

An energy *term* (excluded volume, confinement, chain bonds, …) is the unit of
composition.  Each term lives in its own module under `gnome3d.mc.terms` and
provides FOUR implementations plus ONE shared parameter bundle:

  * ``Params``   — a ``namedtuple`` (single source of truth for the term's
                   parameters).  numba consumes it as a typed tuple; JAX consumes
                   it as a pytree (namedtuples are pytrees), so it vmaps for free.
  * ``nb_local`` — njit ``(pos, p, prm) -> float``: the term's energy
                   contribution that involves bead ``p`` (the per-step delta unit).
  * ``nb_init``  — njit ``(pos, prm) -> float``: the term's full score.
  * ``jax_local``— ``(pos, p, p_pos, prm, n_active) -> float`` (jnp).
  * ``jax_init`` — ``(pos, prm, n_active) -> float`` (jnp, padding-insensitive scan).

The numba and JAX bodies are necessarily written separately (njit cannot share a
traced body with XLA), but they sit side by side in one module and are pinned
together by per-term parity tests (``nb == jax`` on random inputs), so drift is a
failing test, not a silent statistical bug.

A *recipe* is just an ordered ``tuple[Term, ...]`` — that's what a kernel is
(ARCS, SMOOTH, …).  `compose_init_nb` / `compose_local_nb` turn a recipe into one
njit function by code-generating constant-index calls into the recipe's terms;
this is how numba composes a variable-length, heterogeneous term list without
runtime dispatch (each ``prms[i]`` is a literal index, so numba keeps it typed).
JAX composes the same recipe by unrolling the term tuple at trace time (a static
closure), so the two backends share the recipe definition, not the kernel body.

Uniform signatures (some args ignored by a given term, e.g. confinement ignores
``pos``/``p``/``n_active`` in its local form) are what let the composer treat all
terms identically.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast

from numba import njit as _njit  # type: ignore[reportMissingTypeStubs]

if TYPE_CHECKING:
    from collections.abc import Sequence

# Typed njit wrapper (mirrors gnome3d.mc.numba.njit) so the term helpers compile
# with identical decorator options and pyright keeps the original signatures.
F = TypeVar("F", bound=Callable[..., Any])


def njit(**kwargs: Any) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        return cast(F, _njit(**kwargs)(fn))

    return decorator


# Standard decorator options shared by every term helper — same as numba.py's,
# so an extracted helper is byte-identical to the inline original it replaced.
NJIT = {"cache": True, "fastmath": True, "nogil": True}


@dataclass(frozen=True)
class Term:
    """One composeable energy term: its name, its four backend implementations,
    its parameter ``namedtuple`` type, and its global delta convention.

    ``delta_factor`` is how the term's local change scales into the global score
    in the MC accept step (2.0 = double-counted pairwise term, 1.0 = per-bead /
    single-counted).  It is applied by the driver to ``(local_new - local_prev)``,
    NOT to the local score itself — so the composer sums raw locals and the driver
    weights the deltas.
    """

    name: str
    params: type
    nb_local: Callable[..., Any]
    nb_init: Callable[..., Any]
    jax_local: Callable[..., Any]
    jax_init: Callable[..., Any]
    delta_factor: float = 1.0
    # Stateful terms (orientation) carry a mutable cache through the MC loop and
    # can't express their delta as ``local(new) - local(old)``.  When set,
    # ``jax_step(pos, p, old_p, new_p, prm, state, n_active) -> (delta, new_state)``
    # replaces the ``jax_local`` pair in the driver and ``new_state`` is committed
    # on accept.  ``None`` (the default) marks a stateless term.
    jax_step: Callable[..., Any] | None = None


def compose_init_nb(terms: Sequence[Term]) -> Callable[..., Any]:
    """Code-generate an njit ``(pos, prms) -> float`` that sums each term's full
    init score, where ``prms[i]`` is term ``i``'s param namedtuple.

    Constant-index access (``prms[0]``, ``prms[1]``, …) keeps each element typed
    for numba even though the tuple is heterogeneous — this is the mechanism that
    lets a variable-length recipe compile to one njit kernel with no runtime
    dispatch.  Generated functions use ``cache=False`` (no stable module for the
    disk cache); the per-term helpers they call keep ``cache=True``.
    """
    inits = tuple(t.nb_init for t in terms)
    body = " + ".join(f"f{i}(pos, prms[{i}])" for i in range(len(inits))) or "0.0"
    ns: dict[str, Any] = {f"f{i}": fn for i, fn in enumerate(inits)}
    src = f"def _composed_init(pos, prms):\n    return {body}\n"
    exec(src, ns)  # noqa: S102 - trusted codegen from a static recipe
    return cast(Callable[..., Any], _njit(cache=False, fastmath=True, nogil=True)(ns["_composed_init"]))


def compose_local_nb(terms: Sequence[Term]) -> Callable[..., Any]:
    """Code-generate an njit ``(pos, p, prms) -> float`` summing each term's local
    energy at bead ``p``.  Same constant-index codegen trick as `compose_init_nb`.
    (The driver tracks per-term scores for the delta-factor convention; this
    helper is the raw local sum used in tests and the single-counted fast path.)
    """
    locals_ = tuple(t.nb_local for t in terms)
    body = " + ".join(f"f{i}(pos, p, prms[{i}])" for i in range(len(locals_))) or "0.0"
    ns: dict[str, Any] = {f"f{i}": fn for i, fn in enumerate(locals_)}
    src = f"def _composed_local(pos, p, prms):\n    return {body}\n"
    exec(src, ns)  # noqa: S102 - trusted codegen from a static recipe
    return cast(Callable[..., Any], _njit(cache=False, fastmath=True, nogil=True)(ns["_composed_local"]))
